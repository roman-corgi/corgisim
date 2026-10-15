"""
Generate an observation scenario from a CPGS file, one visit at a time,
with each visit's frames generated in parallel (same --option / -n as
generate_visits_example.py; -n is the number of workers in both):
    python generate_visits_from_cpgs_example.py --option mp -n 4               # 4 worker processes (TODO explain whats going on here)
    mpirun -n 5 python generate_visits_from_cpgs_example.py --option mpi -n 4  # 4 worker ranks + 1 manager (TODO explain whats going on here)

This is observation.generate_observation_scenario_from_cpgs with a
frame_runner: 
for every visit (including its satellite-spot background/
positive/negative sequences, if configured) the manager computes the PSF,
generates the frames in parallel, collects them, writes them to disk and
frees them before starting the next visit.

mp  : local multiprocessing (run_parallel) with -n worker processes.
mpi : rank 0 is the manager and ranks 1..n are workers, fed frames through a
      dynamic queue (see corgisim.mpi_runtime). mpirun must start n + 1 ranks;
      the script stops if it doesn't. Worker ranks never get past initialize_mpi_comm().

TODO - explains what is going on in here. 
TODO - write the output dir and a bunch of argumens in here, also whether have log or not. 


Usage of this script is norminally calling just in the cmd, with a specific CPGS file, and the output dir, and the number of workers. The script will run the simulation in parallel for each visit, and write the output to the specified output dir. The script will also log the progress of the simulation to a log file in the output dir.
    python generate_visits_from_cpgs_example.py --option mp -n 4               # 4 worker processes (TODO explain whats going on here)
    mpirun -n 5 python generate_visits_from_cpgs_example.py --option mpi -n 4  # 4 worker ranks + 1 manager (TODO explain whats going on here)

with the multinode thing, TODO - need to attach a shell script to run the simulation in a multinode environment, can provide an example using the lam example but user will really need to reach out to their system admin to get the correct way to run the simulation in a multinode environment, 

Drift / speckle variation is not yet implemented. 


"""
import argparse
import functools
import logging
import os
import time

from corgisim import observation
from corgisim.parallel_executor import run_parallel

import roman_preflight_proper

log = logging.getLogger("generate_visits_from_cpgs_example")


def setup_logging(output_dir):
    """Log to the console and to <output_dir>/run.log."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(output_dir, "run.log"))])


def logged_frame_runner(frame_runner):
    """Wrap a frame_runner to log how long each sequence's frame stage takes.

    generate_observation_scenario_from_cpgs runs up to 4 sequences per visit
    (satellite-spot background/positive/negative, then science), each one
    frame_runner call.
    """
    def run(frame_func, args_list):
        start = time.perf_counter()
        frames = frame_runner(frame_func, args_list)
        log.info("  %d frame(s) [%s] in %.1f s", len(args_list), frame_func.__name__,
                 time.perf_counter() - start)
        return frames
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["mp", "mpi"], default="mp",
                        help="local multiprocessing or MPI (launch with mpirun)")
    parser.add_argument("-n", type=int, default=1,
                        help="number of workers: worker processes (mp) or worker ranks (mpi, "
                             "launched with mpirun -n <n+1>)")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'),
                        help="where each visit's frames are written, one V<visit_id> subfolder per visit")
    parser.add_argument("--cpgs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test',
                                                       'test_data', 'cpgs_short_sequence.xml'),
                        help="CPGS XML file to simulate")
    cli = parser.parse_args()

    comm = None
    if cli.option == "mpi":
        from mpi4py import MPI
        from corgisim.mpi_runtime import initialize_mpi_comm, run_tasks_mpi, shutdown_workers
        # Every rank checks, so a mismatch stops all of them before any work starts
        if MPI.COMM_WORLD.Get_size() != cli.n + 1:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"-n {cli.n} expects {cli.n + 1} MPI ranks (1 manager + {cli.n} workers), "
                      f"got {MPI.COMM_WORLD.Get_size()}: run with mpirun -n {cli.n + 1}")
            raise SystemExit(1)
        if MPI.COMM_WORLD.Get_rank() == 0:
            roman_preflight_proper.copy_here()
        MPI.COMM_WORLD.Barrier()
        comm = initialize_mpi_comm()
    else:
        roman_preflight_proper.copy_here()

    # Only the manager gets here (MPI worker ranks stay in worker_loop)
    setup_logging(cli.output_dir)
    filepath = os.path.abspath(cli.cpgs)
    log.info("option=%s, n=%d workers, cpgs=%s, output_dir=%s", cli.option, cli.n, filepath, cli.output_dir)
    run_start = time.perf_counter()

    # How each visit's frames are run in parallel: on the MPI worker ranks, or
    # on -n local worker processes
    if cli.option == "mpi":
        frame_runner = functools.partial(run_tasks_mpi, comm)
    else:
        frame_runner = functools.partial(run_parallel, n_jobs=cli.n)
    frame_runner = logged_frame_runner(frame_runner)

    try:
        # save_as_list=False (default): frames are written per visit and not kept,
        # so only one visit's frames are in memory at a time. Writing as L1 needs
        # full-frame images.
        observation.generate_observation_scenario_from_cpgs(filepath, save_as_fits=True, output_dir=cli.output_dir,
                                                            full_frame=True, loc_x=512, loc_y=512,
                                                            frame_runner=frame_runner)
    finally:
        if cli.option == "mpi":
            shutdown_workers(comm)

    log.info("Done: visits written to %s in %.1f s", cli.output_dir, time.perf_counter() - run_start)
