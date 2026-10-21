import traceback

from mpi4py import MPI

TASK_RUN = "RUN"
TASK_STOP = "STOP"
TASK_RESULT = "RESULT"
TASK_ERROR = "ERROR"


def initialize_mpi_comm():
    """
    Enter the MPI runtime and assign rank roles.

    Rank 0 returns MPI.COMM_WORLD and continues into the caller's own
    manager code. Every other rank immediately enters worker_loop() and
    never returns -- same split as corgihowfsc's mpi_runtime.initialize_mpi_comm.

    Raises:
        ValueError: if MPI is started with fewer than two ranks.
        SystemExit: on worker ranks, after worker_loop() returns.
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

    Wait for a message from rank 0: a RUN task runs task['func'](*task['args'])
    and sends the result back; STOP ends the loop. One incoming task always produces
    exactly one outgoing message, same shape as corgihowfsc's mpi_runtime.worker_loop:
    the result, or the traceback if the task raised, so the manager never waits for a
    rank that has died. The loop persists across visits and rounds, so a worker rank
    simply picks up the next task (a PSF or a frame) when it arrives.
    """
    while True:
        message = comm.recv(source=0)

        if message['message_type'] == TASK_STOP:
            break

        task = message['task']
        try:
            result = task['func'](*task['args'])
        except Exception:
            comm.send({'message_type': TASK_ERROR, 'job_id': task['job_id'],
                       'traceback': traceback.format_exc()}, dest=0)
            continue
        comm.send({'message_type': TASK_RESULT, 'job_id': task['job_id'], 'result': result}, dest=0)


def run_tasks_mpi(comm, func, args_list, max_workers=None):
    """
    Manager-side dynamic task queue: func(*args) for every args in args_list.

    Sends one task to each active worker rank, then as soon as any
    worker returns a result, it immediately sends that now-free worker the
    next pending task, until every task has been dispatched and collected. 
    
    Blocks until all results are back (barrier). Same scheme as
    corgihowfsc's mpi_runtime._run_manager_task_queue, and the same
    (func, args_list) call shape as parallel_executor.run_parallel, so
    functools.partial(run_tasks_mpi, comm) can be passed as observation.generate_observation_sequence's frame_runner.

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
    # Nothing to do (e.g. a segment with no frames): return without touching the workers
    if not args_list:
        return []

    # Rank 0 is this manager, so the workers are ranks 1 .. size-1. Use at most max_workers of them, and never more than there are tasks (the extra ranks just stay idle in worker_loop).
    n_workers = comm.Get_size() - 1
    
    if max_workers is not None:
        n_workers = min(n_workers, max_workers)
    n_workers = min(n_workers, len(args_list))

    # job_id is the task's position in args_list: workers may answer in any order, and the
    # job_id in each answer tells where its result goes, so results come back in input order
    tasks = [{'job_id': i, 'func': func, 'args': args} for i, args in enumerate(args_list)]
    results = [None] * len(tasks)
    n_sent = 0          # tasks sent so far; they go out in order, so tasks[n_sent] is the next one
    n_answered = 0      # results or errors received so far
    error = None        # (rank, job_id, traceback) of the first failed task

    # Fill the worker pool: one initial task per active worker rank (1, 2, ..., n_workers).
    # n_workers <= len(tasks), so there is always a task for each of them.
    for rank in range(1, n_workers + 1):
        comm.send({'message_type': TASK_RUN, 'task': tasks[n_sent]}, dest=rank)
        n_sent += 1

    completed = 0
    status = MPI.Status()

    while completed < len(tasks):
        message = comm.recv(source=MPI.ANY_SOURCE, status=status)
        rank = status.Get_source()

        ordered_results[message['job_id']] = message['result']
        completed += 1

        if next_task_index < len(tasks):
            comm.send({'message_type': TASK_RUN, 'task': tasks[next_task_index]}, dest=rank)
            next_task_index += 1

    return ordered_results


def shutdown_workers(comm):
    """Tell every worker rank to leave worker_loop and exit cleanly."""
    if comm is None:
        return
    for rank in range(1, comm.Get_size()):
        comm.send({'message_type': TASK_STOP}, dest=rank)
