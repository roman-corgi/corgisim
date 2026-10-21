"""
Simulate visits from settings written in this script (no CPGS file), one visit at a time,
with each visit's frames generated in parallel.

All the simulation settings are in setup_observation(). They reproduce test/test_data/cpgs_ref_wfov_satspots.xml:
wide FOV SPC in band 4, a target star (Eps Eri) and a reference star (alf Col), 7 visits that
alternate between the two stars and between two roll angles, and satellite spots in every visit
(28 PSFs and 107 frames in total, so a serial run is slow). Each visit is one observation sequence
with its own star, roll angle, detector and satellite spot settings. With the same --seed, the
frames should match generate_visits_from_cpgs_example.py run on that file. Edit setup_observation()
to simulate something else.

WARNING: apart from setup_observation(), this script is not meant to be edited.

Output: each visit's frames are written as full-frame L1 FITS files to a V<visit_id> subfolder of
--output-dir (default: examples/output), and the progress is logged to the console and to run.log
in the output dir. The PROPER prescription files are copied into the current directory, so run the
script from a directory you can write to.

If you would like to enable parallelization, you can use the --option mp or --option mpi flags. 
The -n flag specifies the number of workers to use for parallelization. For every visit, the workers compute the PSFs of its segments at the same time (round 1: satellite spot background, positive, negative and science, 4 PSFs) and then all of its frames (round 2); the manager collects the frames, writes them to disk and frees them before starting the next visit.

    mp  : (one machine: a workstation or one cluster node) local multiprocessing (run_parallel) with -n worker processes.
    mpi : (one or more cluster nodes; needs mpi4py and an MPI library) rank 0 is the manager and ranks 1..n are workers,
          fed PSF and frame tasks through a dynamic queue (see corgisim.mpi_runtime). mpirun must start n + 1 ranks;
          the script stops if it doesn't. Worker ranks never get past initialize_mpi_comm().

Choosing -n: 
    each PSF opens its own pool of nlam PROPER processes (nlam = 7 here), and round 1 runs
    up to 4 PSFs at once (if there are satellite spots and no speckle variation), so it briefly needs up to min(n, 4) x nlam CPUs. 
    Round 2 needs one CPU per worker, and more workers than frames in a visit sit idle. 
    Before the run, the script logs the CPUs it may use and warns if -n does not fit (corgisim.parallel_executor.get_cpu_allocation). 
    With MPI the check is per rank: give each rank at least nlam CPUs.

You can also run the script in serial mode (no parallelization) with --option serial. 
This is useful for testing and debugging, as well as for comparing the results of parallel runs.

    serial: every PSF and frame one after another in this process (-n is ignored),
            e.g. as a baseline to compare the parallel runs with (use the same --seed).

Pick how to run with --option; -n is the number of workers for mp and mpi:
    python generate_visits_example.py --option serial                # one process, no parallelization
    python generate_visits_example.py --option mp -n 4               # 4 worker processes
    mpirun -n 5 python generate_visits_example.py --option mpi -n 4  # 4 worker ranks + 1 manager (hence -n 5 for mpirun)
    python generate_visits_example.py --option mp -n 4 --seed 1 --output-dir ./output   # reproducible, chosen output dir

TODO - need to attach a shell script to run the simulation in a multinode environment, can provide an example using the lam example but user will really need to reach out to their system admin to get the correct way to run the simulation in a multinode environment, 

--seed makes the run reproducible: SeedSequence(seed).spawn gives each visit its own sub-seed
(reusing one int for every visit would repeat the same noise in every visit), and each frame
gets its own seed from it, so the frames are the same for serial, -n workers and MPI. Without
--seed the noise pattern changes from run to run.
 
Speckle variation (one PSF per frame) is supported by generate_observation_sequence through
zindex / zval_m_i, but not fully supported with corgisim. 
"""
## import packages
import argparse
import functools
import logging
import os
import time
from corgisim import scene
from corgisim import instrument
from corgisim import observation
from corgisim.parallel_executor import run_parallel, get_cpu_allocation
import numpy as np
import proper
import roman_preflight_proper


log = logging.getLogger("generate_visits_example")


def setup_observation():
    """
    All the simulation settings in one place: edit this function to change the observation, e.g. the stars, visits, roll angles, satellite spots, coronagraph mask, etc.

    The values reproduce test/test_data/cpgs_ref_wfov_satspots.xml the way inputs.load_cpgs_data
    and observation.generate_observation_scenario_from_cpgs read it, except output_dim (51 here,
    201 by default there).

    Returns:
        dict: 'cgi_mode', 'bandpass', 'optics_keywords' (shared by all visits), 'stars' (per star:
        scene, science detector and satellite spot settings) and 'visits' (in the order they are taken).
    """
    # Optics, shared by every visit (coronagraph_mask 2 = wide FOV SPC, howfsc_cfam_pos 4F) 
    cgi_mode = 'excam'
    bandpass = '4F'
    cor_type = 'spc-wide_band4'

    # DM solution closest to initial_target_planet_star_flux_ratio (7e-8), as load_cpgs_data picks it
    rootname = 'spc-wide_ni_5e-9'
    dm1 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')
    
    # polaxis 0 and prism 'None' are the CPGS defaults for imaging without polarimetry;
    # output_dim is the size of the simulated image before it is placed on the detector
    optics_keywords = {'cor_type': cor_type, 'polaxis': 0, 'output_dim': 51, 'prism': 'None',
                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2}

    # Satellite spots: pair id 1 for both stars = 2 pairs at 13 lambda/D, 0 and 90 deg, contrast 1e-7
    satspot_keywords = {'num_pairs': 2, 'sep_lamD': 13, 'angle_deg': [0, 90], 'contrast': 1e-7}

    # Stars: each has its own scene, science detector and satellite spot settings 
    # No companions in this CPGS file. To add one to a star, pass e.g.
    # [{'Vmag': 25, 'magtype': 'vegamag', 'position_x': 3 * 49.3, 'position_y': 3 * 49.3}]  (mas)
    # as the second argument of scene.Scene.
    stars = {
        'target': {   # cpgs_input target_*, satellite_spots_*_targ
            'scene': scene.Scene({'Vmag': 3.73, 'spectral_type': 'K2', 'magtype': 'vegamag',
                                  'ref_flag': False, 'target_name': 'Eps Eri'}),
            'detector': instrument.CorgiDetector({'em_gain': 300.0, 'fast_gain_mode': 'auto', 'gain_CIC_Q': 'roman'},
                                                 photon_counting=True),
            'satspots': {'satspots_are_present': True,
                         'satspots_detector': instrument.CorgiDetector({'em_gain': 300.0}, photon_counting=False),
                         'satspots_exptime': 0.5, 'satspots_number_of_frames': 3,
                         'satspot_keywords': satspot_keywords},
        },
        'reference': {   # cpgs_input reference_*, satellite_spots_* (no _targ)
            'scene': scene.Scene({'Vmag': 2.65, 'spectral_type': 'B9', 'magtype': 'vegamag',
                                  'ref_flag': True, 'target_name': 'alf col'}),
            'detector': instrument.CorgiDetector({'em_gain': 200.0, 'fast_gain_mode': 'auto', 'gain_CIC_Q': 'roman'},
                                                 photon_counting=True),
            'satspots': {'satspots_are_present': True,
                         'satspots_detector': instrument.CorgiDetector({'em_gain': 200.0}, photon_counting=False),
                         'satspots_exptime': 0.1, 'satspots_number_of_frames': 2,
                         'satspot_keywords': satspot_keywords},
        },
    }

    # Visits, in the order they are taken: one visit = one observation sequence 
    # visit_id = default L1 header prefix + visit number; roll_angle is the visit's position_angle (as load_cpgs_data reads it); science frames and exposure time come from the visit's cgi_excam.
    visits = [
        {'visit_id': '0200001001001001001', 'star': 'reference', 'roll_angle': 88.92, 'n_frames': 3,  'exp_time': 0.1,  'vistype': 'CGIVST_TDD_SETUP_HOWFSC'},
        {'visit_id': '0200001001001001002', 'star': 'target',    'roll_angle': 94.53, 'n_frames': 11, 'exp_time': 10.0, 'vistype': 'CGIVST_TDD_OBS'},
        {'visit_id': '0200001001001001003', 'star': 'target',    'roll_angle': 90.54, 'n_frames': 11, 'exp_time': 10.0, 'vistype': 'CGIVST_TDD_OBS'},
        {'visit_id': '0200001001001001004', 'star': 'reference', 'roll_angle': 85.03, 'n_frames': 3,  'exp_time': 0.1,  'vistype': 'CGIVST_TDD_OBS'},
        {'visit_id': '0200001001001001005', 'star': 'target',    'roll_angle': 90.56, 'n_frames': 11, 'exp_time': 10.0, 'vistype': 'CGIVST_TDD_OBS'},
        {'visit_id': '0200001001001001006', 'star': 'target',    'roll_angle': 94.57, 'n_frames': 11, 'exp_time': 10.0, 'vistype': 'CGIVST_TDD_OBS'},
        {'visit_id': '0200001001001001007', 'star': 'reference', 'roll_angle': 89.07, 'n_frames': 3,  'exp_time': 0.1,  'vistype': 'CGIVST_TDD_OBS'},
    ]

    return {'cgi_mode': cgi_mode, 'bandpass': bandpass, 'optics_keywords': optics_keywords,
            'stars': stars, 'visits': visits}


def frames_in_visit(visit, star):
    """Frames in one visit: satellite spot background, positive and negative (if any), then science."""
    satspots = star['satspots']
    n_sat = 3 * satspots['satspots_number_of_frames'] if satspots['satspots_are_present'] else 0
    return n_sat + visit['n_frames']


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
