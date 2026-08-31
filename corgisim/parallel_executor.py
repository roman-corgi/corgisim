import multiprocessing as mp
import multiprocessing.pool as mpp


class NoDaemonProcess(mp.Process):
    """
    A Process subclass that is never daemonic.

    By default, multiprocessing.Pool marks all worker processes as daemon=True,
    which prevents them from spawning their own child processes. This class
    overrides the daemon property to always return False, regardless of what
    Pool internals try to set it to.

    This is required when workers need to spawn their own child processes —
    in our case, each imager worker calls PROPER which internally spawns
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


class NoDaemonContext(type(mp.get_context())):
    """
    A multiprocessing context that uses NoDaemonProcess instead of the
    default Process class. Passed to NestablePool so all workers are
    spawned as non-daemonic processes.
    """
    Process = NoDaemonProcess


class NestablePool(mpp.Pool):
    """
    A Pool subclass that allows workers to spawn their own child processes.

    Standard multiprocessing.Pool workers are daemonic and cannot spawn
    children — attempting to do so raises:
        AssertionError: daemonic processes are not allowed to have children

    NestablePool injects NoDaemonContext so all workers are non-daemonic,
    enabling nested process pools. This is required for corgisim backend
    where each imager worker calls PROPER's internal multiprocessing.Pool.

    Usage:
        with NestablePool(processes=n) as pool:
            results = pool.starmap(func, args_list)

    Warning:
        Only use when nested pools are genuinely required. Non-daemonic
        workers are not automatically terminated if the parent crashes —
        always use as a context manager to ensure cleanup.
    """
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)


def run_parallel(
        func, args_list, n_jobs=1, allow_nesting=False,
        start_method="spawn"):
    """
    Run func over args_list in parallel using multiprocessing.Pool.
    Blocks until all jobs finish (barrier).

    Args:
        func:          top-level picklable callable
        args_list:     list of tuples, each unpacked as func(*args)
        n_jobs:        number of worker processes.
                       on the cluster — never hardcode or use cpu_count()
        allow_nesting: if True, use NestablePool so workers can spawn
                       their own child processes (e.g. PROPER multirun).
        start_method:  process start method. 'spawn' is safest for
                       nested/process-heavy workloads.

    Returns:
        list of results in the same order as args_list
    """
    if not args_list:
        return []

    if n_jobs == 1:
        return [func(*a) for a in args_list]

    if allow_nesting:
        with NestablePool(processes=n_jobs) as pool:
            return pool.starmap(func, args_list)

    # standard multiprocessing.Pool with specified start method but we should not use it
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=n_jobs) as pool:
        return pool.starmap(func, args_list)
