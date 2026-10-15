"""
Simulate several visits (one CorgiOptics state each), one visit at a time,
with each visit's frames generated in parallel.

Pick the parallel backend with --option; -n is the number of workers in both:
    python generate_visits_example.py --option mp -n 4               # 4 worker processes
    mpirun -n 5 python generate_visits_example.py --option mpi -n 4  # 4 worker ranks + 1 manager

For every visit the manager computes the PSF, generates that visit's frames
in parallel, collects them, writes them to disk and frees them before
starting the next visit.

mp  : local multiprocessing (run_parallel) with -n worker processes.
mpi : rank 0 is the manager and ranks 1..n are workers, fed frames through a
      dynamic queue (see corgisim.mpi_runtime). mpirun must start n + 1 ranks;
      the script stops if it doesn't. Worker ranks never get past initialize_mpi_comm().
    
TODO - populate the cases and explain what is going on in here. 

TODO - things are visit based, meaning setup the optics for each visits, including the roll angle and point source and e.t.c, also host star properties if you like to change the host star properties for each visit. Then, run the simulation in parallel for each visit. This is a simplified example, and you can refer to the generate_visits_from_cpgs_example.py for a more complete
example.

Usage of this script is norminally calling just in the cmd, the output dir, and the number of workers. The script will run the simulation in parallel for each visit, and write the output to the specified output dir. The script will also log the progress of the simulation to a log file in the output dir.
    python generate_visits_example.py --option mp -n 4               # 4 worker processes (TODO explain whats going on here)
    mpirun -n 5 python generate_visits_example.py --option mpi -n 4  # 4 worker ranks + 1 manager

with the multinode thing, TODO - need to attach a shell script to run the simulation in a multinode environment, can provide an example using the lam example but user will really need to reach out to their system admin to get the correct way to run the simulation in a multinode environment, 

Drift / speckle variation is not yet implemented. 

"""
## import packages
import argparse
import functools
import logging
import os
import time
from corgisim import scene
from corgisim import instrument
from corgisim import outputs
from corgisim import observation
from corgisim.parallel_executor import run_parallel
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper


## Setup the optics in here, and simulate different visits with stars and planets



log = logging.getLogger("generate_visits_example")


def setup_logging(output_dir):
    """Log to the console and to <output_dir>/run.log."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(output_dir, "run.log"))])


def logged_frame_runner(frame_runner):
    """Wrap a frame_runner to log how long each sequence's frame stage takes."""
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
    log.info("option=%s, n=%d workers, output_dir=%s", cli.option, cli.n, cli.output_dir)
    run_start = time.perf_counter()

    # --- Host Star Properties ---
    Vmag = 8                            # V-band magnitude of the host star
    sptype = 'G0V'                      # Spectral type of the host star
    ref_flag = False                    # if the target is a reference star or not, default is False
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': False}

    # --- Companion Properties ---
    # Define two companions
    mag_companion = [25, 25]           # List of magnitudes for each companion

    # Define their positions relative to the host star, in milliarcseconds (mas)
    # For reference: 1 λ/D at 550 nm with a 2.3 m telescope is ~49.3 mas
    # We're placing them at a separation of 3 λ/D
    dx = 3 * 49.3           # X position relative to host star in sky coordinates (dRA) in mas for the companion
    dy = 3 * 49.3           # Y position relative to host star in sky coordinates (dDec) in mas for the companion

    # Construct a list of dictionaries for all companion point sources
    point_source_info = [
        {
            'Vmag': mag_companion[0],
            'magtype': 'vegamag',
            'position_x': dx,
            'position_y': dy
        },

    ]

    # create a base scene with the host star and companions
    base_scene = scene.Scene(host_star_properties, point_source_info)

    # stellar spectrum of the host star, which is used to compute the flux of the host star and companions
    sp_star = base_scene.stellar_spectrum
    # companion spectrum of the companions, which is used to compute the flux of the companions
    sp_comp = base_scene.off_axis_source_spectrum

    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1F'


    cases = ['3e-8']
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    ##  Define the polarizer axis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10
    # output_dim define the size of the output image
    output_dim = 51
    # roll_angle in degree, define the roll angle of the telescope.
    # roll_angle is defined as the rotation angle of the excam coordinates (X, Y) relative to the sky coordinates(RA,DEC), positive is counter-clockwise
    # Default is 0 degrees, corresponding to North up, East left in the sky coordinates.
    roll_angle = 0

    ### define a dictinatary to pass keywarod to proper
    ### optics_keywords are the keyword arguments to the internal functions for package proper, which define the optics of coronagraph
    # use_dm1/use_dm2: if use dm
    # use_fpm: if use focal plane mask
    # use_lyot_stop: if use lyot stop
    # use_field_stop: if use field stop
    # other paramters that could pass to Proper defined by CgiSim
    nd_filter = 0
    ##options for nd_filter:
    ##nd_filter = 1: ND 2.25 @ FPAM
    ##nd_filter = 2: ND 4.75 @ FPAM
    ##nd_filter = 3: ND 4.75 @ FSAM

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 ,"nd":nd_filter}

    ##visit_type and visit_id are populated into the headers
    visit_type = 'CGIVST_TDD_OBS'
    visit_id  = '0020001001001901001'

    visits = ['0020001001001901001', '0020001001001901002', '0020001001001901003', '0020001001001901004']

    gain =100
    emccd_keywords ={'em_gain':gain}
    detector = instrument.CorgiDetector(emccd_keywords, photon_counting = True)

    exptime = 100
    n_frames = 2

    # How each visit's frames are run in parallel: on the MPI worker ranks, or
    # on -n local worker processes
    if cli.option == "mpi":
        frame_runner = functools.partial(run_tasks_mpi, comm)
    else:
        frame_runner = functools.partial(run_parallel, n_jobs=cli.n)
    frame_runner = logged_frame_runner(frame_runner)

    try:
        for visit_id in visits:
            log.info("Visit %s: %d frame(s), exptime %s s", visit_id, n_frames, exptime)
            visit_start = time.perf_counter()
            # One CorgiOptics instance is one fixed optics state, so each visit gets its own
            optics_i = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords,
                                              roll_angle=roll_angle, visit_type=visit_type,
                                              visit_id=visit_id, if_quiet=True)

            # Frames run in parallel, come back to this (manager) process and are
            # written here; writing as L1 needs full-frame images
            visit_dir = os.path.join(cli.output_dir, 'V' + visit_id)
            os.makedirs(visit_dir, exist_ok=True)
            observation.generate_observation_sequence(base_scene, optics_i, detector, exptime, n_frames,
                                                      vistype=visit_type, visit_id=visit_id,
                                                      save_as_fits=True, output_dir=visit_dir,
                                                      full_frame=True, loc_x=512, loc_y=512,
                                                      frame_runner=frame_runner)
            # The returned frames are not kept, so they are freed before the next visit
            log.info("Visit %s done in %.1f s (PSF + frames + save)", visit_id, time.perf_counter() - visit_start)
    finally:
        if cli.option == "mpi":
            shutdown_workers(comm)

    log.info("Done: %d visits written to %s in %.1f s", len(visits), cli.output_dir, time.perf_counter() - run_start)
