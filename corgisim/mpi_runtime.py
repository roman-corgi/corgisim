import traceback

from mpi4py import MPI

from corgisim.parallel_executor import _memory_report, install_pool_guards

log = logging.getLogger(__name__)

TASK_RUN = "RUN"
TASK_STOP = "STOP"
TASK_RESULT = "RESULT"
TASK_ERROR = "ERROR"


def initialize_mpi_comm():
    """
    Initialize MPI with rank 0 as the manager and all other ranks as workers.

    The manager receives the communicator. Workers enter worker_loop
    and only exit after receiving a stop message.

    Requires at least twon ranks: one manager and at least one worker.

    Raises:
        ValueError: MPI was started with fewer than two ranks.
        SystemExit: A worker rank exits after the manager sends its stop message.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        worker_loop(comm)
        raise SystemExit(0)
    if comm.Get_size() < 2:
        raise ValueError("MPI mode requires at least 2 ranks (1 manager + >=1 worker)")
    return comm


def worker_loop(comm):
    """
    Persistent loop for worker ranks.

    Wait for a message from rank 0, execute the requested task, and send back either the result or an error traceback. A STOP message ends the loop.

    Workers remain alive across visits and rounds, processing new tasks (PSF calculation or detector frames) as they arrive.
    """
    install_pool_guards()

    while True:
        message = comm.recv(source=0)

        if message["message_type"] == TASK_STOP:
            break

        task = message["task"]

        try:
            result = task["func"](*task["args"])
        except BaseException:
            # Add this rank's memory use, so it appears in rank 0's log with the error.
            comm.send(
                {
                    "message_type": TASK_ERROR,
                    "job_id": task["job_id"],
                    "traceback": (
                        f"{traceback.format_exc()}"
                        f"Memory on this worker rank: {_memory_report()}\n"
                    ),
                },
                dest=0,
            )
            continue
        comm.send(
            {
                "message_type": TASK_RESULT,
                "job_id": task["job_id"],
                "result": result,
            },
            dest=0,
        )


def run_tasks_mpi(comm, func, args_list, max_workers=None):
    """
    Manager-side dynamic task queue: func(*args) for every args in args_list.

    Sends one task to each active worker rank, then as soon as any
    worker returns a result, it immediately sends that now-free worker the
    next pending task, until every task has been dispatched and collected. 
    
    Blocks until all dispatched tasks have completed.

    If a task raises on a worker, no new tasks are sent, the tasks still running are
    collected (so no worker is left blocked sending its result), and a RuntimeError with
    the worker's traceback is raised here. The caller should then call shutdown_workers.

    Args:
        comm: MPI communicator, from initialize_mpi_comm().
        func: module-level function to run on the workers (sent by reference).
        args_list (list of tuple): one positional argument tuple per task: one PSF
            (_compute_psf, round 1) or one frame (_apply_detector, or _generate_one_frame
            with zindex), see observation.prepare_frame_tasks. generate_observation_sequence
            calls this once per round: twice per visit, or once with zindex.
        max_workers (int or None): optional cap on active worker ranks,
            same role as run_parallel's n_jobs.

    Returns:
        list: func results, ordered the same as args_list.

    Raises:
        RuntimeError: if a task raised on a worker rank (message includes its traceback).
    """
    # Rank 0 manages tasks; ranks 1 through size - 1 can do the work.
    n_workers = comm.Get_size() - 1

    if max_workers is not None and max_workers < 1:
        raise ValueError(f"max_workers must be >= 1 or None, got {max_workers}")

    # Nothing to do, for example when a segment contains no frames.
    if not args_list:
        return []

    if max_workers is not None:
        n_workers = min(n_workers, max_workers)
    n_workers = min(n_workers, len(args_list))

    # Keep each task's input position so results can be reordered on arrival.
    tasks = [
        {"job_id": task_id, "func": func, "args": args}
        for task_id, args in enumerate(args_list)
    ]
    results = [None] * len(tasks)
    n_dispatched = 0
    n_received = 0
    first_error = None  # (rank, task_id, traceback)

    # Start with one task per active worker rank.
    for rank in range(1, n_workers + 1):
        comm.send({"message_type": TASK_RUN, "task": tasks[n_dispatched]}, dest=rank)
        n_dispatched += 1

    # Receive each response, then refill that worker if tasks remain.
    status = MPI.Status()

    while n_received < n_dispatched:
        # ANY_SOURCE lets the first worker to finish continue with the next task.
        message = comm.recv(source=MPI.ANY_SOURCE, status=status)
        n_received += 1
        rank = status.Get_source()

        if message["message_type"] == TASK_ERROR:
            # Preserve the first traceback; it is usually the most informative.
            if first_error is None:
                first_error = (rank, message["job_id"], message["traceback"])
        else:
            task_id = message["job_id"]
            results[task_id] = message["result"]

        # Once an error occurs, let active tasks finish but start no new tasks.
        if first_error is None and n_dispatched < len(tasks):
            comm.send(
                {"message_type": TASK_RUN, "task": tasks[n_dispatched]},
                dest=rank,
            )
            n_dispatched += 1

    # All workers that received tasks are now idle and can be shut down safely.
    if first_error is not None:
        rank, task_id, worker_traceback = first_error
        function_name = getattr(func, "__name__", repr(func))
        message = (
            f"Task {task_id} ({function_name}) failed on MPI rank {rank}:\n"
            f"{worker_traceback}"
        )
        # Also log on rank 0 so the error reaches the manager's log file.
        log.error("%s", message)
        raise RuntimeError(message)

    return results


def shutdown_workers(comm):
    """Tell every worker rank to leave worker_loop and exit cleanly."""
    if comm is None:
        return
    for rank in range(1, comm.Get_size()):
        comm.send({"message_type": TASK_STOP}, dest=rank)
