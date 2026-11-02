import logging
import multiprocessing as mp
import multiprocessing.pool as mpp
import time
import psutil
import os
import warnings
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
    enabling nested process pools. This is required for the corgisim
    backend, where each worker calls PROPER's internal multiprocessing.Pool.

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


class WorkerDiedError(RuntimeError):
    """A pool worker exited while tasks were pending (e.g. killed by the OOM killer)."""

def _run_task(task):
    """func(*args) for pool.imap, which passes one argument per task. Module level, so it pickles under spawn."""
    func, args = task
    return func(*args)


def _memory_report():
    """Report approximate memory usage of this process and its subprocesses, together with avaliable memory."""

    process = psutil.Process() 

    main = process.memory_info().rss
    worker_memory = 0

    for child in process.children(recursive=True):
        try:
            worker_memory += child.memory_info().rss
        except psutil.Error:   # child exited between listing and reading
            pass
    
    available_memory = psutil.virtual_memory().available
    return (
        f"RAM: main {main / 1e9:.2f} GB, subprocesses {worker_memory / 1e9:.2f} GB, device available {available_memory / 1e9:.2f} GB"
    )


def _dead_workers(pool, known_workers):
    """
    Return workers that have exited unexpectedly. 
    Remember workers because Pool may replace them after a crash. 
    """

    for worker in list(pool._pool):
        if worker.pid is not None:
            known_workers.setdefault(worker.pid, worker)

    return [
        (pid, worker.exitcode)
        for pid, worker in known_workers.items()
        if worker.exitcode is not None
    ]        


def _run_tasks_with_monitoring(pool, func, args_list, poll_interval=5.0, log_interval=60.0, timeout=None):
    """
    Run tasks in parallel while monitoring their execution. 
    
    Report progress and memory usage periodically. Raise an error if a worker dies or the time limit is exceeded,
    rather than waiting indefinitely.

    Results are returned in the same order as the input tasks.
    """
    if poll_interval <= 0 or log_interval <= 0:
        raise ValueError("Monitoring intervals must be positive.")

    name = getattr(func, "__name__", repr(func))
    n_tasks = len(args_list)

    # Remember the original workers so we can detect their deaths, even if Pool replaces them.
    known_workers = {
        worker.pid: worker
        for worker in pool._pool
        if worker.pid is not None
    }

    # imap returns results in input order and supports timed waiting.
    tasks = ((func, args) for args in args_list)
    results_iter = pool.imap(_run_task, tasks, chunksize=1)

    results = []
    start = last_log = time.perf_counter()

    log.info(
        "%s: started %d tasks. %s",
        name, n_tasks, _memory_report()
    )

    while len(results) < n_tasks:
        # Check the overall time limit.
        elapsed = time.perf_counter() - start

        if timeout is not None and elapsed >= timeout:
            pool.terminate()
            raise TimeoutError(
                f"{name}: simulation exceeded the {timeout}s "
                f"time limit ({len(results)}/{n_tasks} results received)."
            )

        # Do not wait beyond the remaining time limit.
        wait = poll_interval
        if timeout is not None:
            wait = min(wait, timeout - elapsed)

        # Wait for the next result.
        try:
            results.append(results_iter.next(timeout=wait))
        except mp.TimeoutError:
            pass

        # Check whether a worker has died.
        dead = _dead_workers(pool, known_workers)

        if dead:
            memory = _memory_report()
            pool.terminate()

            details = ", ".join(
                f"process {pid} exited with code {code}"
                + (" (SIGKILL; possible memory issue)" if code == -9 else "")
                for pid, code in dead
            )

            raise WorkerDiedError(
                f"{name}: a simulation worker stopped unexpectedly. "
                f"{len(results)}/{n_tasks} results received. "
                f"{details}. {memory}. "
                "Check the job logs, reduce n_jobs, or request more memory."
            )

        # Check elapsed time and report progress.
        now = time.perf_counter()
        elapsed = now - start

        if now - last_log >= log_interval:
            log.info(
                "%s: %d/%d results received after %.0f s. %s",
                name, len(results), n_tasks, elapsed,
                _memory_report()
            )
            last_log = now

    log.info(
        "%s: all %d tasks completed in %.0f s. %s",
        name, n_tasks, time.perf_counter() - start,
        _memory_report()
    )

    return results

def run_parallel(func, args_list, n_jobs=1, allow_nesting=True, start_method="spawn", poll_interval=5.0, log_interval=60.0, timeout=None):
    """
    Run func over args_list in parallel using multiprocessing.Pool.
    Blocks until all jobs finish (barrier).

    Args:
        func:          top-level picklable callable
        args_list:     list of tuples, each unpacked as func(*args)
        n_jobs:        number of worker processes.
                       On the cluster — never hardcode or use cpu_count()
        allow_nesting: if True (default), use NestablePool so workers can
                       spawn their own child processes (e.g. PROPER multirun,
                       which corgisim frames with their own optics state run).
        start_method:  process start method. 'spawn' is safest for
                       nested/process-heavy workloads.
        poll_interval: seconds between checks that all workers are still alive.
        log_interval:  seconds between progress reports (cannot be None).
        timeout:       optional total time limit in seconds (None = no limit).

    Returns:
        list of results in the same order as args_list

    Raises:
        WorkerDiedError: if a worker dies (e.g. killed for exceeding the memory limit).
        TimeoutError:    if timeout is set and exceeded.
    """
    if not args_list:
        return []

    if n_jobs == 1:
        return [func(*a) for a in args_list] # serial, no multiprocessing apart from PROPER's internal Pool

    if allow_nesting:
        with NestablePool(processes=n_jobs) as pool:
            return _run_tasks_with_monitoring(pool, func, args_list, poll_interval, log_interval, timeout)

    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=n_jobs) as pool:
        return _run_tasks_with_monitoring(pool, func, args_list, poll_interval, log_interval, timeout)

def available_cpus():
    """
    Number of CPUs this process may use, and where that number comes from.

    Checks: 
        1. os.process_cpu_count() or os.sched_getaffinity(0) (Linux). On Linux these follow the CPU affinity set by a scheduler such as SLURM/PBS when it pins the job to its allocated CPUs. 
        2. SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE / PBS_NP environment variables 
        3. os.cpu_count(), the whole machine, which can overestimate on a shared node.
    
    Returns:
        tuple: (n_cpus (int), source (str)).
    """
    if hasattr(os, 'process_cpu_count') and os.process_cpu_count():
        return os.process_cpu_count(), 'os.process_cpu_count()'
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0)), 'os.sched_getaffinity(0)'
    for name in ['SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'PBS_NP']:
        if os.environ.get(name, '').isdigit() and int(os.environ[name]) > 0:
            return int(os.environ[name]), name
    return os.cpu_count() or 1, 'os.cpu_count()'
