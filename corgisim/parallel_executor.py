import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import warnings

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


def run_parallel(func, args_list, n_jobs=1, allow_nesting=True, start_method="spawn"):
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

    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=n_jobs) as pool:
        return pool.starmap(func, args_list)


def available_cpus():
    """
    Number of CPUs this process may use, and where that number comes from.

    Checks: 
        1. os.process_cpu_count() or os.sched_getaffinity(0) (Linux). 
            On Linux these follow the CPU affinity set by a scheduler such as SLURM/PBS when it pins the job to its allocated CPUs. 
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


def get_cpu_allocation(n_jobs=1, nlam=1, n_psf_states=1, zindex=False, n_frames=None, warn=True):
    """
    Check a frame_runner's worker count against the CPUs this process may use, for one corgisim observation sequence, and warn about oversubscription.

    A sequence runs in up to two parallel rounds (see observation.prepare_frame_tasks):
      - PSF: one task per distinct optics state (n_psf_states: 1, or 3 with satellite
        spots: background, positive, negative). Each task is one PSF, i.e. one
        proper.prop_run_multi, which opens its own pool of min(nlam, os.cpu_count())
        processes. 
            Peak: min(n_jobs, n_psf_states) * that pool (n_jobs = 1: the PSFs run in this process, one pool at a time). 
            Polarimetry runs its 4 PROPER calls one after another inside a task, so it does not raise the peak.

      - Detector: n_jobs workers, one CPU each (EMCCD noise, no PROPER).
    
    With zindex (a time series with speckle variation) every frame task is a PSF plus the detector step,
    so the peak is PROPER process * n_jobs in one visit. 

    PROPER sizes its pool from os.cpu_count(), not from the allocation, since
    corgisim does not pass NCPUS. On a cluster node shared with other jobs that pool can be
    larger than this process is allowed, which is also reported.

    Args:
        n_jobs (int): number of frame_runner workers (run_parallel's n_jobs, or the number of
            MPI worker ranks on this node).
        nlam (int): number of wavelengths of the optics (CorgiOptics.nlam).
        n_psf_states (int): distinct optics states in the sequence: 1, or 3 with satellite spots.
        zindex (bool): True if the sequence has a time series with speckle variation (one PSF per frame).
        n_frames (int, optional): frames in the largest segment (nominally the detector segment), to report idle workers.
        warn (bool): if True (default), issue a UserWarning for each problem found.

    Returns:
        dict: 'cpus', 'cpus_source', 'proper_pool' (processes per PSF), 'peak_cpu_count',
        'detector_peak', 'suggested_n_jobs', and 'problems' (list of str, empty if none).

    Raises:
        ValueError: if n_jobs, nlam, n_psf_states or n_frames is not a positive integer.
    """
    for name, val in [('n_jobs', n_jobs), ('nlam', nlam), ('n_psf_states', n_psf_states)] + \
                     ([('n_frames', n_frames)] if n_frames is not None else []):
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} must be a positive integer, got {val!r}")

    cpus, source = available_cpus()
    node_cpus = os.cpu_count() or cpus
    proper_pool = min(nlam, node_cpus)

    if zindex: # speckle variation: every frame is a PSF plus detector, so the peak is n_jobs * PROPER pool
        peak_cpu_count = n_jobs * proper_pool
        suggested = max(1, cpus // proper_pool)
    else:
        # Up to n_psf_states PSFs run at once, one per worker
        peak_cpu_count = (min(n_jobs, n_psf_states) if n_jobs > 1 else 1) * proper_pool
        # the detector only needs one CPU per worker; more workers than frames sit idle
        suggested = min(cpus, n_frames) if n_frames else cpus
    detector_peak = n_jobs

    # No warning when only os.cpu_count() is available (e.g. macOS, Windows): that is the allocation
    # On Linux clusters sched_getaffinity / the scheduler variables are read instead.
    # 'cpus_source' in the returned dict says which method was used to get the CPU count.
    problems = []
    if proper_pool > cpus:
        problems.append(f"PROPER opens min(nlam, os.cpu_count()) = {proper_pool} processes per PSF, more than the "
                        f"{cpus} CPUs allocated ({source}): every PSF is oversubscribed, whatever n_jobs is.")
    
    # how many PSFs run in parallel at the peak (1 if n_jobs=1, or min(n_jobs, n_psf_states) if n_jobs>1)
    n_psf_at_once = peak_cpu_count // proper_pool 
    if peak_cpu_count > cpus and n_psf_at_once > 1:   
        what = 'Every frame' if zindex else f'The PSF round ({n_psf_at_once} PSFs at once)'
        problems.append(f"{what}: {n_psf_at_once} x {proper_pool} PROPER processes = {peak_cpu_count} > "
                        f"{cpus} CPUs ({source}), {peak_cpu_count / cpus:.1f}x oversubscribed."
                        + (f" Use n_jobs <= {suggested}." if zindex else
                           " The PSF round is short compared with the detector round, so this is usually acceptable."))
    # The detector round is one CPU per worker, so oversubscription is only a problem if n_jobs > cpus.
    if detector_peak > cpus: 
        problems.append(f"The detector round: n_jobs = {n_jobs} workers > {cpus} CPUs ({source}). Use n_jobs <= {cpus} if possible.")
    # workers > frames in the largest segment, so some workers will sit idle.
    if n_frames is not None and n_jobs > n_frames: 
        problems.append(f"n_jobs = {n_jobs} > {n_frames} frames in the largest segment: "
                        f"{n_jobs - n_frames} workers will sit idle.")

    if warn:
        for problem in problems:
            warnings.warn(problem, UserWarning, stacklevel=2)

    # returning this dict is useful for logging in the calling script, e.g. generate_visits_from_cpgs_example.py
    # logging is needed when running these jobs, as the user may not see the warnings in the terminal.
    return {'cpus': cpus, 'cpus_source': source, 'proper_pool': proper_pool, 'peak_cpu_count': peak_cpu_count,
            'detector_peak': detector_peak, 'suggested_n_jobs': suggested, 'problems': problems}

