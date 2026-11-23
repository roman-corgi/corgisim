import logging
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import threading
import time
import warnings
import weakref
import psutil

log = logging.getLogger(__name__)


class NoDaemonProcess(mp.get_context("spawn").Process):
    """
    A Process subclass that is never daemonic.

    By default, multiprocessing.Pool marks all worker processes as daemon=True,
    which prevents them from spawning their own child processes. This class
    overrides the daemon property to always return False, regardless of what
    Pool internals try to set it to.

    This is required when workers need to spawn their own child processes —
    in our case, each frame worker calls PROPER which internally spawns
    its own multiprocessing.Pool(NCPUS).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        # Ignore attempts by Pool internals to set daemon=True
        pass


class NoDaemonContext(type(mp.get_context("spawn"))):
    """
    A multiprocessing context that uses NoDaemonProcess instead of the
    default Process class. Passed to NestablePool so all workers are
    spawned as non-daemonic processes.

    Always 'spawn', never the 'fork': forked workers inherit
    the parent's numpy RNG state, so frames drawn in different workers would
    get identical detector noise (same random seed for emccd.detect)
    """
    Process = NoDaemonProcess


class NestablePool(mpp.Pool):
    """
    A Pool subclass that allows workers to spawn their own child processes.

    Standard multiprocessing.Pool workers are daemonic and cannot spawn
    children — attempting to do so raises:
        AssertionError: daemonic processes are not allowed to have children

    NestablePool injects NoDaemonContext so all workers are non-daemonic,
    enabling nested process pools. This is required for the corgisim, where
    each worker calls PROPER's internal multiprocessing.Pool.

    Usage:
        Use run_parallel(..., allow_nesting=True) to enable
        worker monitoring and timeout handling.

    Warning:
        Only use when nested pools are genuinely required. Non-daemonic
        workers are not automatically terminated if the parent crashes —
        always use as a context manager to ensure cleanup.
    """
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)


class WorkerProcessError(RuntimeError):
    """A worker could not start or stopped unexpectedly before completing its
    task."""


def run_task(task):
    """
    Run a function with the given arguments provided for one task.

    Pool.map() passes each task as a single argument. This function separates
    the function and its arguments, then calls the function with the provided
    arguments.

    It is defined outside the run_parallel() function so that multiprocessing
    can send it to worker process when using the 'spawn' start method.
    """
    func, args = task
    return func(*args)


def memory_report():
    """
    Return a summary of memory usage without interrupting the simulation.

    Collecting memory information can fail, especially when it is runnign out
    of memory. If this happens, return a short message instead of raising
    another error.
    """
    try:
        return collect_memory_report()
    except Exception:
        return "RAM: unavailable"


def collect_memory_report():
    """
    Report the memory used by this process and its subprocesses.

    The report includes memory used by the main Python process and all its
    child processes, including those started by multiprocessing (e.g. PROPER
    workers). 

    On a Linux cluster, the report also includes the job's memory limit,
    current usage, and remaining memory when this information is available.
    Otherwise, it reports the available memory on the whole machine.

    Memory values are reported in GB.
    """

    process = psutil.Process()

    # Memory used by the current Python process. 
    main = process.memory_info().rss
    worker_memory = 0

    # Add the memory used by all child processes, including nested workers.
    for child_process in process.children(recursive=True):
        try:
            worker_memory += child_process.memory_info().rss
        except psutil.Error:   
            # A child may have finished before its memory could be measured.
            pass

    report = f"RAM: main {main / 1e9:.2f} GB, subprocesses {worker_memory / 1e9:.2f} GB"

    # On a cluster, the job's memory limit is more relevant than the avaliable
    # memory on the whole nodes. 
    job_memory = get_job_memory()
    if job_memory is not None:
        limit, used, source = job_memory
        if used is None:
            # The job's memory limit is known, but its usage is unavailable.
            return f"{report}, job limit {limit / 1e9:.2f} GB ({source})"

        # Report the memory used and remaining within the selected job limit.
        return (f"{report}, job used {used / 1e9:.2f} of {limit / 1e9:.2f} GB,"
                f"job available {(limit - used) / 1e9:.2f} GB ({source})")

    # Without a job-specific limit, report the avaliable memory of the machine.
    available_memory = psutil.virtual_memory().available
    return f"{report}, device available {available_memory / 1e9:.2f} GB"


def get_job_memory():
    """
    Find how much memory a job is allowed to use and how much it is using.

    On a cluster, each job is usually assigned a memory limit. This can be
    much smaller than the total memory avaliable on the node. Linux keeps
    track of these limits and the memory used by each job through a system
    called cgroups.

    A job can have serveral levels of memory limits (for examples, one for the
    whole job and one for each process). This function checks these levels and
    returns the smallest memory limit found, together with the memory usage at
    that level.

    The reported memory usage also includes memory used for file caching, so
    it may be higher than the memory used directly by the Python processes.

    On macOS, windows, or Linux system without a job-specific memory limit
    (cgroup), this function returns None. The caller can then use the
    avalialbe memory on the whole machine instead.

    Returns:
        tuple or None:
            (limit, used, source), where:
            - limit: Maximum memory allowed, in bytes.
            - used: Current memory usage, in bytes, or None if unavailable.
            - source: The cgroup from which the memory information was read.

            Returns None if no job-specific memory limit is found.
    """

    def read_integer(path):
        """Read a number from a file, or return None if unavailable."""
        try:
            with open(path) as f:
                contents = f.read().strip()
        except OSError:
            return None


        # cgroup v2 uses "max" to mean there is no limit at this level.
        if contents.isdigit():
            return int(contents)
        return None
        

    try:
        with open("/proc/self/cgroup") as f:
            cgroup_lines = f.read().splitlines()
    except OSError:
        # macOS and Windows do not provide Linux cgroup information.
        cgroup_lines = []

    smallest_limit_info = None

    for line in cgroup_lines:
        _, controllers, cgroup_path = line.split(":", 2)
        if controllers == "":
            # cgroup v2: these files contain the memory limit and usage.
            cgroup_root = "/sys/fs/cgroup"
            limit_file, usage_file, version = "memory.max", "memory.current", "v2"
        elif "memory" in controllers.split(","):
            # cgroup v1: the same information is stored under different names.
            cgroup_root = "/sys/fs/cgroup/memory"
            limit_file = "memory.limit_in_bytes"
            usage_file, version = "memory.usage_in_bytes", "v1"
        else:
            continue

        # Check this group and its parent groups, since the whole job may have 
        # a different memory limit from an individual job step.
        while True:
            directory = os.path.join(cgroup_root, cgroup_path.lstrip("/"))
            limit = read_integer(os.path.join(directory, limit_file))

            # ignore missing or unlimited memory values. In cgroup v1, 
            # unlimited memory is represented by a very large number. 
            if limit is None:
                is_real_limit = False
            else:
                is_real_limit = limit < psutil.virtual_memory().total

            if smallest_limit_info is None:
                is_tighter_limit = True
            elif limit is None:
                is_tighter_limit = False
            else:
                is_tighter_limit = limit < smallest_limit_info[0]

            # keep the smallest memory limit found and its usage.
            if is_real_limit and is_tighter_limit:
                used = read_integer(os.path.join(directory, usage_file))
                source = f"cgroup {version} {cgroup_path}"
                smallest_limit_info = (limit, used, source)

            # Stop after checking the top of the cgroup hierarchy.
            if cgroup_path in ("", "/"):
                break

            # Move up one level to check the parent's memory limit.
            cgroup_path = os.path.dirname(cgroup_path.rstrip("/")) or "/"

    return smallest_limit_info


def raise_if_pool_cannot_start():
    """
    Prevent Pool.map() from hanging if a worker cannot start.

    Normally, a multiprocessing pool starts several worker processes to run
    tasks. If Python cannot start one of these workers. (for example, because 
    there is no enough memory), the thread responsible for managing the
    workers may crash.

    In this situation, Pool.map() can wait forever for results that will never
    arrive. This function installs an exception handler that detects such
    failures and raises WorkerProcessError in the waiting caller,
    allowing the simulation to stop with a clear error message instead of
    hanging indefinitely.

    The recovery relies on internal parts of Python's multiprocessing
    implementation. If these change and recovery is no longer possible, the
    process is terminated to avoid waiting forever.

    The handler is installed only once per process. If another exeception
    handler was already installed, it will be still called.
    """
    if getattr(threading.excepthook, "_corgisim_pool_guard", False):
        return  # the handler is already installed in this process.

    previous_hook = threading.excepthook

    def handle_thread_exception(exception_info):
        # Let the previous handler report the original exception.
        # If that handler fails, use Python's default handler instead.
        try:
            previous_hook(exception_info)
        except Exception:
            threading.__excepthook__(exception_info)

        traceback = exception_info.exc_traceback
        while traceback is not None:
            current_frame = traceback.tb_frame

            if (current_frame.f_code.co_name == "_handle_workers"
                    and current_frame.f_globals.get("__name__") == "multiprocessing.pool"):

                try:
                    # Find the tasks still waiting for results and
                    # the queue used by the pool to manage tasks.
                    pending_tasks = current_frame.f_locals["cache"]
                    task_queue = current_frame.f_locals["taskqueue"]

                    waiting_results = []
                    for result in list(pending_tasks.values()):
                        if hasattr(result, "_event"):
                            waiting_results.append(result)
                    

                    if not waiting_results:
                        raise RuntimeError("No pending map/apply results found")

                    worker_error = WorkerProcessError(
                        f"Could not start a pool worker ({exception_info.exc_value})."
                        "Request more memory or reduce processes per node."
                    )

                    # Notify the waiting tasks immediately so Pool.map() can
                    # raise an Error instead of hanging.
                    for result in waiting_results:
                        result._success = False
                        result._value = worker_error
                        result._event.set()

                    # Signal the pool's task handling thread to stop.
                    task_queue.put(None)

                except BaseException:
                    # If recovery fails, terminate this process rather than
                    # leave the simulation waiting indefinitely.
                    try:
                        os.write(
                            2,
                            b"FATAL: Could not recover failed pool handler. "
                            b"Terminating process.\n"
                        )
                    finally:
                        os._exit(1)

                # Mark the handler so it is not installed again in this process.
                handle_thread_exception._corgisim_pool_guard = True
                # Report memory usage if possible, without risking another failure during error handling. 
                try:
                    log.error("%s %s", worker_error, _memory_report())
                except Exception:
                    pass
                return

            traceback = traceback.tb_next

    # Mark the handler so it is not installed again in this process.
    handle_thread_exception._corgisim_pool_guard = True
    threading.excepthook = handle_thread_exception


def raise_if_pool_worker_dies(poll_interval=0.5):
    """
    Prevent Pool.map() from hanging when a worker dies unexpectedly.

    When a worker process stops unexpectedly (for example, because it runs out
    of memory), Python may automatically start a new worker to replace it.
    However, the task being processed by the dead worker can be lost, leaving
    Pool.map() waiting forever for its result.

    This function installs a handler that checks for failed worker and
    notifies waiting tasks so the simualtions can stop with an error instead
    of hanging indefinitely.

    The handler is started automatically for each new process poll and is
    installed only once per Python process.
    """
    if getattr(mpp.Pool, "_watch_workers", False):
        return  # already installed in this process

    original_init = mpp.Pool.__init__ # save the original Pool.__init__ method to call it later

    def initialize_pool(self, *args, **kwargs):
        # create the pool normally before starting the watcher.
        original_init(self, *args, **kwargs)

        # only monitor real processes, not thread pools.
        if isinstance(self, mpp.ThreadPool):
            return  # thread workers are not processes; nothing to watch

        # remember the initial workers so when a worker dies, we can detect it
        # even if Pool replaces it with a new worker.
        initial_workers = {}

        for worker in list(self._pool):
            if worker.pid is not None:
                initial_workers[worker.pid] = worker

        # start a background thread to monitor this pool.
        # Use a weak reference to monitor the pool without keeping it alive (or caching it).
        threading.Thread(
            target=watch_pool_workers,
            args=(weakref.ref(self), poll_interval, initial_workers),
            daemon=True,
        ).start()
    
    # automatically start a watcher whenever a new process pool is created.
    mpp.Pool.__init__ = initialize_pool
    mpp.Pool._watch_workers = True


def watch_pool_workers(pool_ref, poll_interval, initial_workers=None):
    """
    Monitor a process pool for unexpected worker deaths.

    If a worker dies while a Pool.map() is waiting for results,
    notify the waiting tasks and terminal the pool.

    Stop monitoring when the pool is no longer avaliable or its worker handler
    thread stops.
    """
    known_workers = dict(initial_workers or {})

    while True:
        # Access the pool through a weak reference so the pool can be garbage
        # collected when no longer needed.
        pool = pool_ref()

        if pool is None:
            return
        if not pool._worker_handler.is_alive():
            return

        # find workers that stopped with a non-zero exit code.
        dead_workers = []

        for process_id, exit_code in stopped_workers(pool, known_workers):
            if exit_code != 0:
                dead_workers.append((process_id, exit_code))

        # find waiting Pool.map() results that can be notified.
        # imap results are monitored separately in _run_tasks_with_monitoring()
        pending_results = []
        for result in list(pool._cache.values()):
            if hasattr(result, "_event"):
                pending_results.append(result)

        if dead_workers and pending_results:
            # build a description of the dead workers and their exit codes,
            # including a note if the exit code indicates a SIGKILL (possible
            # memory issue).
            failure_message = []
            for process_id, exit_code in dead_workers:
                message = f"process {process_id} exited with code {exit_code}"
                
                if exit_code == -9:
                    message += " (SIGKILL; possible memory issue)"
                failure_message.append(message)
            
            failure_details = ", ".join(failure_message)

            worker_error = WorkerProcessError(
                f"A pool worker stopped unexpectedly (e.g. a PROPER process): {failure_details}. "
                "Request more memory or reduce processes per node."
            )

            # Notify waiting operations before collecting memory information,
            # which may be slow when the system is running out of memory.
            for result in pending_results:
                result._success = False
                result._value = worker_error
                result._event.set()

            try:
                log.error("%s %s", worker_error, memory_report())
            finally:
                pool.terminate()
            return

        # Release the local reference before sleeping so the pool can be
        # garbage collected if no longer needed.
        del pool
        time.sleep(poll_interval)


def install_pool_guards():
    """
    Prevent multiprocessing pools from hanging when a worker cannot start or
    dies unexpectedly.

    A worker may fail to start or die unexpectedly while runnign a task.
    In either case, Pool.map() can wait forever for a result that will never arrive.

    This function installs two exception handlers to detect these failures and
    raise WorkerProcessError instead of leaving the simulation hanging.
    The handlers are installed only once per Python process.
    """
    # Prevent Pool.map() from hanging if a worker cannot start.
    raise_if_pool_cannot_start()

    # Prevent Pool.map() from hanging when a worker dies unexpectedly.
    raise_if_pool_worker_dies() 


def stopped_workers(pool, known_workers):
    """
    Find workers that have stoppped running, including those have been replaced. 

    When a worker process dies unexpectedly, multiprocessing.Pool may
    automatically replace it with a new worker. The original worker is then
    removed from the pool's list, making it difficult to detect that the worker died.

    This function keeps track of all workers observed in the pool, including
    those that have been replaced, and check whether they have exited.

    Args:
        pool: A multiprocessing.Pool whose workers are being monitored.
        known_workers: A dictionary mapping worker process IDs (PIDs) to their
            multiprocessing.Process objects. Updated with any new workers found in the pool.

    Returns:
        list: Tuple (process_id, exit_code) for each worker that has exited.
        A exit code of 0 means normal termination, while a non-zero exit code
        indicates an error or signal termination (e.g. -9 for SIGKILL).
    """

    # remember new workers, including those that replaced dead workers.
    for worker in list(pool._pool):
        if worker.pid is not None:
            known_workers.setdefault(worker.pid, worker)

    # Check all known workers, including those no longer in the pool.
    stopped_workers = []
    for process_id, worker in known_workers.items():
        if worker.exitcode is not None:
            stopped_workers.append((process_id, worker.exitcode))

    return stopped_workers


def run_tasks_with_monitoring(pool, func, args_list, poll_interval=5.0, log_interval=60.0):
    """
    Collect ordered task results while monitoring the worker pool.

    Periodically report progress and memory use. Stop with an error if a worker
    exits unexpectedly or the pool can no longer start workers.
    """
    if poll_interval <= 0 or log_interval <= 0:
        raise ValueError("Monitoring intervals must be positive.")

    function_name = getattr(func, "__name__", repr(func))
    n_tasks = len(args_list)

    # Remember the original workers so we can detect their deaths
    # even if Python replaces them.
    known_workers = {}
    for worker in pool._pool:
        if worker.pid is not None:
            known_workers[worker.pid] = worker

    # imap returns results in input order and allows timed waiting.
    tasks = ((func, args) for args in args_list)
    results_iter = pool.imap(run_task, tasks, chunksize=1)

    results = []
    start_time = time.perf_counter()
    last_log_time = start_time

    log.info("%s: started %d tasks. %s", function_name, n_tasks, memory_report())

    while len(results) < n_tasks:
        # Wait up to poll_interval for the next result, then check the workers' status.
        try:
            results.append(results_iter.next(timeout=poll_interval))
        except mp.TimeoutError:
            pass
        except Exception:
            # Log the task failure and memory usage before raising the error.
            log.exception("%s: task %d/%d failed. %s", function_name, len(results) + 1, n_tasks,
                          memory_report())
            raise

        # Check whether any known workers have stopped.
        dead_workers = []

        for process_id, exit_code in stopped_workers(pool, known_workers):
            if exit_code != 0:
                dead_workers.append((process_id, exit_code))

        if dead_workers:
            memory = memory_report()
            pool.terminate()

            # Build a description of the dead worker.
            failure_message = []
            for process_id, exit_code in dead_workers:
                message = f"process {process_id} exited with code {exit_code}"
                if exit_code == -9:
                    message += " (SIGKILL; possible memory issue)"
                failure_message.append(message)

            failure_details = ", ".join(failure_message)

            raise WorkerProcessError(
                f"{function_name}: a simulation worker stopped unexpectedly. "
                f"{len(results)}/{n_tasks} results received. "
                f"{failure_details}. {memory}. "
                "Check the job logs, reduce n_jobs, or request more memory."
            )

        # The handler thread also dies, the new worker cannot be started.
        # In that case, waiting for the remaining imap results would hang.
        if not pool._worker_handler.is_alive():
            memory = memory_report()
            pool.terminate()
            raise WorkerProcessError(
                f"{function_name}: could not start a simulation worker (see the traceback above, e.g. "
                f"'Cannot allocate memory'). {len(results)}/{n_tasks} results received. {memory}. "
                "Reduce n_jobs or request more memory."
            )

        # Report progress at regular intervals.
        current_time = time.perf_counter()

        if current_time - last_log_time >= log_interval:
            log.info(
                "%s: %d/%d results received after %.0f s. %s",
                function_name,
                len(results),
                n_tasks,
                current_time - start_time,
                memory_report(),
            )
            last_log_time = current_time

    log.info(
        "%s: all %d tasks completed in %.0f s. %s",
        function_name,
        n_tasks,
        time.perf_counter() - start_time,
        memory_report(),
    )

    return results


def log_progress_until_stopped(function_name, n_tasks, results, start_time, log_interval, stop_event):
    """
    Log progress and memory usage at every ``log_interval`` seconds until
    ``stop_event`` is set.

    Runs in a background thread so messages continue while a long task
    (e.g. PROPER propagation) runs in the main thread.
    """
    # wait between progress reports, or stop when the simulation finishes.
    while not stop_event.wait(log_interval):
        try:
            log.info(
                "%s: %d/%d results received after %.0f s. %s",
                function_name,
                len(results),
                n_tasks,
                time.perf_counter() - start_time,
                memory_report(),
            )
        except Exception:
            pass  # Logging failtures should not interrupt the simulation.


def run_tasks_serially(func, args_list, log_interval=60.0):
    """
    Run tasks one after another in this process, with progress messages.
    A background thread periodically reports progress and memory usage,
    even while a long task is running.

    The messages match those of _run_tasks_with_monitoring().
    """
    if log_interval <= 0:
        raise ValueError("Monitoring intervals must be positive.")

    function_name = getattr(func, "__name__", repr(func))
    n_tasks = len(args_list)
    results = []
    start_time = time.perf_counter()

    log.info("%s: started %d tasks in this process. %s", function_name, n_tasks, memory_report())

    # start a background thread to log progress until the simulation finishes or fails.
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=log_progress_until_stopped,
        args=(function_name, n_tasks, results, start_time, log_interval, stop_event),
        daemon=True,
    )
    progress_thread.start()
    try:
        # Run each task sequentially and collect its results.
        for args in args_list:
            try:
                results.append(func(*args))
            except Exception:
                # Log the task 
                log.exception("%s: task %d/%d failed. %s", function_name, len(results) + 1, n_tasks,
                              memory_report())
                raise
    finally:
        # stop the progress thread, even if a task fails.
        stop_event.set() # signal the progress thread to stop
        progress_thread.join() # waitu until the progress thread has stopped.

    log.info(
        "%s: all %d tasks completed in %.0f s. %s",
        function_name,
        n_tasks,
        time.perf_counter() - start_time,
        memory_report(),
    )
    return results


def run_parallel(
    func,
    args_list,
    n_jobs=1,
    allow_nesting=True,
    start_method="spawn",
    poll_interval=5.0,
    log_interval=60.0,
):
    """
    Run a function sequentially or in parallel for each set of arguments.

    For example, ``[(frame_a,), (frame_b,)]`` runs ``func(frame_a)`` and
    ``func(frame_b)``. Results follow the input order, even when tasks finish
    in a different order. The function waits until every task are completed.

    Args:
        func: Function to run for each task. It must be importable by workers
              when using multiprocessing.
        args_list: One tuple of positional arguments per task.
        n_jobs: Number of worker processes to create. When n_jobs=1, task run
        sequentially in the current process.
        allow_nesting: Allow workers to start their own pools, as required by
            PROPER. Defaults to True.
        start_method: Multiprocessing start method. "spawn" is the
            default and avoids inheriting the caller's process state.
        poll_interval: Seconds between checks for worker failures.
        log_interval: Seconds between progress messages. With n_jobs=1
            they also appear while a long task is still running.

    Returns:
        list: Task results in the same order as args_list

    Raises:
        WorkerProcessError: If a worker dies or cannot be started.
    """
    if not args_list:
        return []

    # Install safeguards for all multiprocessing pools created in this process.
    install_pool_guards()

    # Run tasks sequentially when one process is requested.
    if n_jobs == 1:
        return run_tasks_serially(func, args_list, log_interval)

    # Install the same guard in each worker process.
    pool_initializer = {"initializer": install_pool_guards}

    if allow_nesting:
        # Use a custom Pool subclass that allows workers to spawn their own
        # child processes.
        with NestablePool(processes=n_jobs, **pool_initializer) as pool:
            return run_tasks_with_monitoring(
                pool, func, args_list, poll_interval, log_interval
            )
    else:
        raise ValueError(
            "Nested multiprocessing is required for PROPER, but allow_nesting=False was requested."
            )


def available_cpus():
    """
    Find the number of CPUs avaiable to current process.

    On a shared computing cluster, a job may be assigned with fewer CPUs than
    the total number of CPUs on the node. This function checks the CPU
    allocation using the following methods, in order:
    
    1. Python's process CPU count (os.process_cpu_count())
    2. CPU affinity on Linux, which identifies the CPUs avaliable to the
    current process.
    3. CPU allocation reported by job schedulers (e.g. SLURM)
    4. Total CPUs on the machine, if no other information is available.
    
    Returns:
        tuple: (n_cpus, source), where:
            - n_cpus: Number of CPUs reported as avaliable.
            - source: Method used to determine this number.
    """
    # Prefer the number of CPUs available to the current process.
    if hasattr(os, "process_cpu_count"): 
        process_cpu_count = os.process_cpu_count()
        if process_cpu_count: 
            return process_cpu_count, "os.process_cpu_count()"

    # On Linux CPU affinity specifies which CPUs the current process can use.
    if hasattr(os, "sched_getaffinity"):
        allowed_cpus = os.sched_getaffinity(0)
        return len(allowed_cpus), "os.sched_getaffinity(0)"

    # Check CPU allocation provided by cluster schedulers if there is one.
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NP"):
        value = os.environ.get(name, "")
        if value.isdigit() and int(value) > 0:
            return int(value), name

    # Fall back to the total number of CPUs on the machine.
    total_cpus = os.cpu_count()
    if not total_cpus:
        total_cpus = 1

    return total_cpus, "os.cpu_count()"


def get_cpu_allocation(n_jobs=1, nlam=1, n_psf_states=1, zindex=False,
    n_frames=None, warn=True,):
    """
    Estimate the peak CPU usage of an observation sequence and try to detect
    oversubscription.

    A sequence normally has two rounds.

    1. PSF calculations: multiple PSF states can run in parallel. Each PSF
    calculatation can start a PROPER pool with one process per wavelength.
    Different optical state (e.g. applying satellite spots, speckle-variation,
    or jitter) may run as seperate tasks. The four polarization calculations
    within a task run sequentially, so they do not multiply the peak.

    2. Detector calculations: each frame is processed in parallel, with one
    worker per frame.

    When zindex=True, each frame includes its own time-varying PSF calculation,
    so each active frame worker may start a PROPER pool.

    PROPER uses os.cpu_count() to size its wavelength pool. On a shared
    cluster node, that can exceed the CPUs assigned to this job.
    
    This function estimates the number of processes used and compares it with
    the available CPU allocation to identify potential oversubscription.

    Args:
        n_jobs: Number of frame tasks processed concurrently by local workers
            or MPI ranks.
        nlam: Number of wavelengths in the optics model.
        n_psf_states: Number of distinct PSF states (usually one, or four with
            satellite spots).
        zindex: Whether each frame has its own time-varying PSF calculation.
        n_frames: Number of frames in the visit, used to flag idle workers.
        warn: Issue a ``UserWarning`` for each detected problem when ``True``.

    Returns:
        dict: CPU allocation information containing:
            available_cpu_count: Number of CPUs available to this process.
            cpu_source: Method used to determine the CPU allocation.
            proper_processes_per_psf: PROPER processes started per PSF.
            peak_cpu_count: Estimated maximum simultaneous process count during PSF calculations.
            detector_peak: Number of detector workers requested.
            suggested_n_jobs: Suggested worker count for this allocation.
            problems: List of CPU oversubscription or idle-worker messages.

    Raises:
        ValueError: If n_jobs, nlam, n_psf_states or n_frames is not a positive integer.
    """
    # Check that all task and process counts are positive integers.
    parameters = {
        "n_jobs": n_jobs,
        "nlam": nlam,
        "n_psf_states": n_psf_states,
    }
    if n_frames is not None:
        parameters["n_frames"] = n_frames

    for name, value in parameters.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    # Find the available CPUs and estimate the size of each PROPER pool.
    available_cpu_count, cpu_source = available_cpus()
    node_cpu_count = os.cpu_count() or available_cpu_count
    proper_processes_per_psf = min(nlam, node_cpu_count) 

    if zindex:
        # Every active frame worker may start a PROPER pool.
        peak_cpu_count = n_jobs * proper_processes_per_psf
        suggested_n_jobs = max(1, available_cpu_count // proper_processes_per_psf)
    else:
        # The PSF round runs one task per worker, limited by the number of PSF states.
        if n_jobs > 1: 
            simultaneous_psfs = min(n_jobs, n_psf_states)
        else:
            simultaneous_psfs = 1

        peak_cpu_count = simultaneous_psfs * proper_processes_per_psf

        # Detector tasks use one CPU each; workers beyond n_frames are idle.
        if n_frames:
            suggested_n_jobs = min(n_frames, available_cpu_count)
        else:
            suggested_n_jobs = available_cpu_count

    detector_peak = n_jobs

    # Check whether one PROPER pool already exceed the available CPUs.abs
    problems = []

    if proper_processes_per_psf > available_cpu_count:
        problems.append(
            f"PROPER opens min(nlam, os.cpu_count()) = {proper_processes_per_psf} processes per PSF, "
            f"more than the {available_cpu_count} CPUs allocated ({cpu_source}): every PSF is oversubscribed, "
            "whatever n_jobs is."
        )

    # Estimate how many PSFs or frames run concurrently at the peak.
    n_psf_at_once = peak_cpu_count // proper_processes_per_psf

    if peak_cpu_count > available_cpu_count and n_psf_at_once > 1:
        if zindex:
            activity = "Every frame"
            recommendation = f" Use n_jobs <= {suggested_n_jobs}."
        else:
            activity = f"The PSF round ({n_psf_at_once} PSFs at once)"
            recommendation = (
                "The PSF round is short compared with the detector round, "
                "so this is usually acceptable."
            )
        problems.append(
            f"{activity}: {n_psf_at_once} x {proper_processes_per_psf} PROPER processes = "
            f"{peak_cpu_count} > {available_cpu_count} CPUs ({cpu_source}), "
            f"{peak_cpu_count / available_cpu_count:.1f}x oversubscribed.{recommendation}"
        )

    # The detector round uses one CPU per worker.
    if detector_peak > available_cpu_count:
        problems.append(
            f"The detector round: n_jobs = {n_jobs} workers > {available_cpu_count} CPUs "
            f"({cpu_source}). Use n_jobs <= {available_cpu_count} if possible."
        )

    # Check whether there are more workers than frames, which would leave some workers idle.
    if n_frames is not None and n_jobs > n_frames:
        problems.append(
            f"n_jobs = {n_jobs} > {n_frames} frames in the visit: "
            f"{n_jobs - n_frames} workers will sit idle."
        )

    if warn:
        for problem in problems:
            warnings.warn(problem, UserWarning, stacklevel=2)

    # Return the CPU allocation information for logging or further checks.
    return {
        "available_cpu_count": available_cpu_count,
        "cpu_source": cpu_source,
        "proper_processes_per_psf": proper_processes_per_psf,
        "peak_cpu_count": peak_cpu_count,
        "detector_peak": detector_peak,
        "suggested_n_jobs": suggested_n_jobs,
        "problems": problems,
    }
