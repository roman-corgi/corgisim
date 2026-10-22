"""
Generate an observation scenario from a CPGS file, one visit at a time, with each visit's frames generated in parallel.

This is observation.generate_observation_scenario_from_cpgs with a frame_runner. The default --cpgs
is test/test_data/cpgs_ref_wfov_satspots.xml: wide FOV SPC in band 4, a target and a reference
star, 7 visits at two roll angles, satellite spots in every visit (28 PSFs, 107 frames).
generate_visits_example.py writes the same observation out as Python settings. For a quick check
use --cpgs test/test_data/cpgs_short_sequence.xml (2 visits, 8 PSFs, 18 frames; its satellite spot
frames are 0 s at gain 1, so they show no spots).

WARNING: this script is not meant to be edited. To simulate another observation, pass its CPGS file
with --cpgs (or use generate_visits_example.py to write the observation in Python).

Output: each visit's frames are written as full-frame L1 FITS files to a V<visit_id> subfolder of
--output-dir (default: examples/output), and the progress is logged to the console and to run.log
in the output dir. The PROPER prescription files are copied into the current directory, so run the
script from a directory you can write to.

If you would like to enable parallelization, you can use the --option mp or --option mpi flags. The -n flag specifies the number of workers to use for parallelization. For every visit, the workers compute the PSFs of its segments at the same time (round 1: satellite spot background, positive, negative and science, 4 PSFs, or just science without satellite spots) and then all of its frames (round 2); the manager collects the frames, writes them to disk and frees them before starting the next visit.

    mp  : (one machine: a workstation or one cluster node) local multiprocessing (run_parallel) with -n worker processes.
    mpi : (one or more cluster nodes; needs mpi4py and an MPI library) rank 0 is the manager and ranks 1..n are workers,
          fed PSF and frame tasks through a dynamic queue (see corgisim.mpi_runtime). mpirun must start n + 1 ranks;
          the script stops if it doesn't. Worker ranks never get past initialize_mpi_comm().

Choosing -n: 
    each PSF opens its own pool of nlam PROPER processes (nlam = 7 for the default file),
    and round 1 runs up to 4 PSFs at once (1 without satellite spots), so it briefly needs up to
    min(n, 4) x nlam CPUs. 
    Round 2 needs one CPU per worker, and more workers than frames in a visit sit idle. Before the run, the script reads the CPGS file, logs the CPUs it may use and a suggested -n for that file, and warns if -n does not fit (corgisim.parallel_executor.get_cpu_allocation). With MPI the check is per rank: give each rank at least nlam CPUs.

You can also run the script in serial mode (no parallelization) with --option serial. This is useful for testing and debugging, as well as for comparing the results of parallel runs.

    serial: every PSF and frame one after another in this process (-n is ignored),
            e.g. as a baseline to compare the parallel runs with (use the same --seed).

Pick how to run with --option; -n is the number of workers for mp and mpi:
    python generate_visits_from_cpgs_example.py --option serial                # one process, no parallelization
    python generate_visits_from_cpgs_example.py --option mp -n 4               # 4 worker processes
    mpirun -n 5 python generate_visits_from_cpgs_example.py --option mpi -n 4  # 4 worker ranks + 1 manager
    python generate_visits_from_cpgs_example.py --option mp -n 4 --cpgs my_file.xml --seed 1 --output-dir ./output

TODO - need to attach a shell script to run the simulation in a multinode environment, can provide an example using the lam example but user will really need to reach out to their system admin to get the correct way to run the simulation in a multinode environment, 

--seed makes the run reproducible: each visit gets its own sub-seed and each frame its own
seed, so the frames are the same for serial, -n workers and MPI. Without --seed the noise pattern
changes from run to run.

Speckle variation is not supported from a CPGS file yet: generate_observation_scenario_from_cpgs
has no zindex (see generate_observation_sequence for the zindex / zval_m_i time series).

User can change where to place the scene onto the detector with LOC_X and LOC_Y (pixels). The default is (512, 512).
"""
import argparse
import functools
import logging
import os
import time

from corgisim import inputs, observation
from corgisim.parallel_executor import run_parallel, get_cpu_allocation

import roman_preflight_proper

log = logging.getLogger("generate_visits_from_cpgs_example")
LOC_X, LOC_Y = 512, 512  # star location on the detector (pixels)


def setup_logging(output_dir):
    """Log to the console and to <output_dir>/run.log."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(output_dir, "run.log"))])


def logged_frame_runner(frame_runner):
    """Wrap a frame_runner to log how long each stage of a visit takes.

    generate_observation_sequence calls frame_runner twice per visit: once for the
    PSFs (_compute_psf: satellite-spot background/positive/negative and science, or one
    PSF without satellite spots) and once for all of the visit's detector frames
    (_apply_detector).
    """
    def run(frame_func, args_list):
        start = time.perf_counter()
        frames = frame_runner(frame_func, args_list)
        log.info("  %d task(s) [%s] in %.1f s", len(args_list), frame_func.__name__,
                 time.perf_counter() - start)
        return frames
    return run


def check_cpus(filepath, n_workers, mpi):
    """
    Log whether n_workers fits the CPUs this job may use, for this CPGS file.

    Reads the CPGS file with inputs.load_cpgs_data (no PROPER run) to get the number of
    wavelengths, whether there are satellite spots (4 PSFs per visit instead of 1) and the
    largest number of frames in one visit (satellite spot frames included: the detector
    round runs all of a visit's frames in one call). 
    With MPI every worker rank is its own process with its own CPUs (e.g. SLURM --cpus-per-task) running one task at a time, so the check is per rank, against this (manager) rank's allocation.
    """
    loaded = inputs.load_cpgs_data(filepath)
    if len(loaded) == 8:   # target and reference
        optics, visits, sat_target, sat_reference = loaded[2], loaded[5], loaded[6], loaded[7]
    else:                  # target only
        optics, visits, sat_target, sat_reference = loaded[1], loaded[3], loaded[4], None
    nlam = optics.nlam

    # With satellite spots a visit has 4 PSFs (background, positive, negative, science), otherwise 1
    n_psf_states = 4 if (sat_target or sat_reference) else 1

    # The detector round runs all of a visit's frames at once: check against the largest visit.
    # Frames in a visit = 3 satellite spot segments of that visit's star (if any) + science frames.
    largest_visit = 0
    for visit in visits:
        sat = sat_reference if visit['isReference'] else sat_target
        n_sat_frames = 3 * sat['satellite_spots_number_of_frames'] if sat else 0
        largest_visit = max(largest_visit, visit['number_of_frames'] + n_sat_frames)

    # With MPI every worker rank is its own process with its own CPUs, so check one rank (n_jobs=1)
    n_jobs = 1 if mpi else n_workers

    report = get_cpu_allocation(n_jobs=n_jobs, nlam=nlam, n_psf_states=n_psf_states,
                                n_frames=largest_visit, warn=False)
    log.info("CPUs: %d available (%s); per PSF PROPER opens %d processes (nlam=%d); "
             "peak %d processes in the PSF round, %d in the detector round%s",
             report['cpus'], report['cpus_source'], report['proper_pool'], nlam,
             report['peak_cpu_count'], report['detector_peak'],
             " per MPI rank" if mpi else "")

    if not mpi:
        log.info("Suggested -n for this file: %d", report['suggested_n_jobs'])
    
    for problem in report['problems']:
        log.warning(problem)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["serial", "mp", "mpi"], default="mp",
                        help="serial (one process, -n ignored), local multiprocessing, or MPI (launch with mpirun)")
    parser.add_argument("-n", type=int, default=1,
                        help="number of workers: worker processes (mp) or worker ranks (mpi, "
                             "launched with mpirun -n <n+1>)")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'),
                        help="where each visit's frames are written, one V<visit_id> subfolder per visit")
    parser.add_argument("--cpgs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test',
                                                       'test_data', 'cpgs_ref_wfov_satspots.xml'),
                        help="CPGS XML file to simulate")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for the detector noise; the same seed gives the same frames "
                             "with any --option / -n (default: not reproducible)")
    args = parser.parse_args()
    if args.option == "serial":
        args.n = 1   # serial: one process, so -n is ignored

    comm = None
    if args.option == "mpi":
        from mpi4py import MPI
        from corgisim.mpi_runtime import initialize_mpi_comm, run_tasks_mpi, shutdown_workers
        # Every rank checks, so a mismatch stops all of them before any work starts
        if MPI.COMM_WORLD.Get_size() != args.n + 1:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"-n {args.n} expects {args.n + 1} MPI ranks (1 manager + {args.n} workers), "
                      f"got {MPI.COMM_WORLD.Get_size()}: run with mpirun -n {args.n + 1}")
            raise SystemExit(1)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Copying the PROPER prescription files to the working directory (rank 0)...")
            roman_preflight_proper.copy_here()
        MPI.COMM_WORLD.Barrier()
        comm = initialize_mpi_comm()
    else:
        roman_preflight_proper.copy_here()

    # Only the manager gets here (MPI worker ranks stay in worker_loop)
    setup_logging(args.output_dir)
    filepath = os.path.abspath(args.cpgs)
    log.info("option=%s, n=%d workers, seed=%s, cpgs=%s, output_dir=%s", args.option, args.n, args.seed, filepath, args.output_dir)
    # Warn before the run if -n does not fit the CPUs this job may use
    check_cpus(filepath, args.n, mpi=args.option == "mpi")
    run_start = time.perf_counter()

    # How each visit's PSFs and frames are run: on the MPI worker ranks, on -n local worker
    # processes, or (serial, n_jobs=1) one after another in this process
    if args.option == "mpi":
        frame_runner = functools.partial(run_tasks_mpi, comm)
    else:
        frame_runner = functools.partial(run_parallel, n_jobs=args.n)
    frame_runner = logged_frame_runner(frame_runner) # wrap to log how long each visit's frame stage takes (PSFs + frames)

    try:
        # save_as_list=False (default): frames are written per visit and not kept,
        # so only one visit's frames are in memory at a time. Writing as L1 needs
        # full-frame images.
        observation.generate_observation_scenario_from_cpgs(filepath, save_as_fits=True, output_dir=args.output_dir, 
                                                            full_frame=True, loc_x=LOC_X, loc_y=LOC_Y,
                                                            frame_runner=frame_runner, seed=args.seed)
    finally:
        if args.option == "mpi":
            shutdown_workers(comm)

    log.info("Done: visits written to %s in %.1f s", args.output_dir, time.perf_counter() - run_start)
