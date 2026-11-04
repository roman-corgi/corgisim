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
    python generate_visits_example.py --option serial --output-dim 201  # one process, no parallelization
    python generate_visits_example.py --option mp -n 4 --output-dim 201  # 4 worker processes
    mpirun -n 5 python generate_visits_example.py --option mpi -n 4 --output-dim 201  # 4 worker ranks + 1 manager (hence -n 5 for mpirun)
    python generate_visits_example.py --option mp -n 4 --output-dim 201 --seed 1 --output-dir ./output   # reproducible, chosen output dir

To simulate disk observations, need to pass additional arguments: the disk-model FITS file, matching PRF cube, and pixel-value unit. 

The disk is attached to the target star only; the reference star has no disk. The disk must be square and --output-dim must equal its size (padding is not implemented); the disk must be at least as large as the minimum for the coronagraph mode (51 pixels for HLC band 1, 201 for SPC-wide band 4). Use a PRF cube generated for the observation's bandpass and optics.

The disk is convolved once per visit, at that visit's roll angle, in the main
process (rank 0 for MPI), and added to every satellite-spot and science frame
before detector noise. Only the main process reads the two files.
The supplied PRF cube stays fixed across satellite-spot DM and wavefront aberrations.

    python generate_visits_example.py --option mp -n 2 --output-dim 201 --disk-model disk.fits \
        --prf-cube prf_cube.fits --surface-brightness-unit contrast/pixel \
        --output-dir ./disk_output

The same with MPI (2 workers + 1 manager = 3 ranks). Pass the same arguments to
every rank (mpirun does); only rank 0 reads the disk and PRF files:

    mpirun -n 3 python generate_visits_example.py --option mpi -n 2 --output-dim 201 --disk-model disk.fits \
        --prf-cube prf_cube.fits --surface-brightness-unit contrast/pixel \
        --output-dir ./disk_output_mpi

--seed makes the run reproducible: SeedSequence(seed).spawn gives each visit its own sub-seed
(reusing one int for every visit would repeat the same noise in every visit), and each frame
gets its own seed from it, so the frames are the same for serial, -n workers and MPI. Without
--seed the noise pattern changes from run to run.

Speckle variation (one PSF per frame) is supported by generate_observation_sequence through
zindex / zval_m_i, but not fully supported with corgisim. 

User can change where to place the scene onto the detector with loc_x and loc_y (pixels). The default is (512, 512). It is at the beginning of setup_observation() and passed to generate_observation_sequence().
"""
## import packages
import argparse
import functools
import logging
import os
import time
from astropy.io import fits
from corgisim import scene
from corgisim import instrument
from corgisim import observation
from corgisim.parallel_executor import run_parallel, get_cpu_allocation
import numpy as np
import proper
import roman_preflight_proper


log = logging.getLogger("generate_visits_example")


def setup_observation(output_dim, twoD_scene_info=None):
    """
    All the simulation settings in one place: edit this function to change the observation, e.g. the stars, visits, roll angles, satellite spots, coronagraph mask, etc.

    The values reproduce test/test_data/cpgs_ref_wfov_satspots.xml the way inputs.load_cpgs_data
    and observation.generate_observation_scenario_from_cpgs read it.
    
    output_dim is the size of the simulated image (pixels) before it is placed on the detector, given with --output-dim.

    With a disk it must equal the disk size, which must be at least the minimum for the coronagraph mode
    (51 for HLC band 1, 201 for SPC-wide band 4). 
    twoD_scene_info contains disk_model_path, prf_path, surface_brightness_unit
    and optional interpolate_prfs, as in disk_sim_demo.ipynb.
    The disk is attached only to the target star, not the reference.

    Returns:
        dict: 'cgi_mode', 'bandpass', 'optics_keywords' (shared by all visits), 'stars' (per star:
        scene, science detector and satellite spot settings) and 'visits' (in the order they are taken).
    """
    LOC_X, LOC_Y = 512, 512  # star location on the detector (pixels)

    # Optics, shared by every visit (coronagraph_mask = wide FOV SPC, howfsc_cfam_pos 4F)
    # Adjust to match the coronagraph mode and bandpass you want to simulate.
    cgi_mode = 'excam'
    bandpass = '4F'
    cor_type = 'spc-wide_band4'

    # DM solution closest to initial_target_planet_star_flux_ratio (7e-8), as load_cpgs_data picks it
    rootname = 'spc-wide_ni_5e-9'
    dm1 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')
    
    # polaxis 0 and prism 'None' are the CPGS defaults for imaging without polarimetry;
    # output_dim is the size of the simulated image before it is placed on the detector
    optics_keywords = {'cor_type': cor_type, 'polaxis': 0, 'output_dim': output_dim, 'prism': 'None',
                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2}

    # Satellite spots: pair id 1 for both stars = 2 pairs at 13 lambda/D, 0 and 90 deg, contrast 1e-7
    satspot_keywords = {'num_pairs': 2, 'sep_lamD': 13, 'angle_deg': [0, 90], 'contrast': 1e-7}

    # Stars: each has its own scene, science detector and satellite spot settings 
    # No companions in this CPGS file. To add one to a star, pass e.g.
    # [{'Vmag': 25, 'magtype': 'vegamag', 'position_x': 3 * 49.3, 'position_y': 3 * 49.3}]  (mas)
    # as the second argument of scene.Scene. See scene.Scene.__init__() for the full list of parameters (host star, companions, 2D scene, etc.).
    stars = {
        'target': {   # cpgs_input target_*, satellite_spots_*_targ
            'scene': scene.Scene({'Vmag': 3.73, 'spectral_type': 'K2', 'magtype': 'vegamag',
                                  'ref_flag': False, 'target_name': 'Eps Eri'},
                                 twoD_scene_info=twoD_scene_info),
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
            'stars': stars, 'visits': visits, 'loc_x': LOC_X, 'loc_y': LOC_Y}


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
    """Wrap a frame_runner to log how long each sequence's frame stage takes (the PSF, then the frames)."""
    def run(frame_func, args_list):
        start = time.perf_counter()
        frames = frame_runner(frame_func, args_list)
        log.info("  %d frame(s) [%s] in %.1f s", len(args_list), frame_func.__name__,
                 time.perf_counter() - start)
        return frames
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["serial", "mp", "mpi"], default="mp",
                        help="serial (one process, -n ignored), local multiprocessing, or MPI (launch with mpirun)")
    parser.add_argument("-n", type=int, default=1,
                        help="number of workers: worker processes (mp) or worker ranks (mpi, "
                             "launched with mpirun -n <n+1>)")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'),
                        help="where each visit's frames are written, one V<visit_id> subfolder per visit")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for the detector noise; the same seed gives the same frames "
                             "with any --option / -n (default: noise pattern not reproducible)")
    parser.add_argument("--output-dim", type=int, required=True,
                        help="size (pixels) of the simulated image before it is placed on the detector; with a disk it must equal the disk size")
    parser.add_argument("--disk-model", help="2D disk-model FITS file")
    parser.add_argument("--prf-cube", help="matching field-dependent PRF-cube FITS file")
    parser.add_argument("--surface-brightness-unit",
                        choices=["mJy/arcsec^2", "W.m^-2.pixel^-1", "contrast/pixel"],
                        help="unit of the disk-model pixel values")
    parser.add_argument("--interpolate-prfs", action=argparse.BooleanOptionalAction, default=True,
                        help="interpolate between neighbouring PRFs (default); --no-interpolate-prfs uses the nearest PRF")
    
    args = parser.parse_args()
    disk_args = (args.disk_model, args.prf_cube, args.surface_brightness_unit) # group the three disk-related arguments to check if they are all given or none

    # Check that either all three disk-related arguments are given or none of them are given
    if any(disk_args) and not all(disk_args):
        parser.error("--disk-model, --prf-cube, and --surface-brightness-unit must be given together")

    # Setup the twoD_scene_info dictionary if all disk-related arguments are provided, otherwise set it to None
    if all(disk_args):
        disk_path, prf_path = os.path.abspath(args.disk_model), os.path.abspath(args.prf_cube)
        twoD_scene_info = {'disk_model_path': disk_path, 'prf_path': prf_path,
                           'surface_brightness_unit': args.surface_brightness_unit,
                           'interpolate_prfs': args.interpolate_prfs}
        header = fits.getheader(disk_path)
        if header['NAXIS1'] != header['NAXIS2']:
            parser.error(f"The disk must be square, got {header['NAXIS1']} x {header['NAXIS2']}")
        if args.output_dim != header['NAXIS1']:
            parser.error(f"--output-dim ({args.output_dim}) must equal the disk size ({header['NAXIS1']}); padding the disk is not implemented")
    else:
        twoD_scene_info = None
        
    if args.option == "serial":
        args.n = 1   # serial: one process, so -n is ignored

    comm = None # MPI communicator, if MPI is used; here for scope, not used in serial or mp
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
            roman_preflight_proper.copy_here()
        MPI.COMM_WORLD.Barrier()
        comm = initialize_mpi_comm()
    else:
        roman_preflight_proper.copy_here()

    # Only the manager gets here (MPI worker ranks stay in worker_loop)
    setup_logging(args.output_dir)
    log.info("option=%s, n=%d workers, seed=%s, output_dir=%s", args.option, args.n, args.seed, args.output_dir)
    run_start = time.perf_counter()

    obs = setup_observation(args.output_dim, twoD_scene_info=twoD_scene_info) # setup the observing scenario: optics, stars and visits
    visits, stars = obs['visits'], obs['stars']

    # How each visit's PSFs and frames are run: on the MPI worker ranks, on -n local worker processes, or (serial, n_jobs=1) one after another in this process
    if args.option == "mpi":
        frame_runner = functools.partial(run_tasks_mpi, comm)
    else:
        frame_runner = functools.partial(run_parallel, n_jobs=args.n)
    frame_runner = logged_frame_runner(frame_runner) # wrap to log how long each visit's frame stage takes (PSFs + frames)
 
    # Warn before the run if -n does not fit the CPUs this job may use.
    optics_check = instrument.CorgiOptics(obs['cgi_mode'], obs['bandpass'],
                                          optics_keywords=obs['optics_keywords'], if_quiet=True)

    nlam = optics_check.nlam # number of PROPER processes per PSF (7 here, from the coronagraph mask and bandpass)

    # With satellite spots a visit has 4 PSFs (background, positive, negative, science), otherwise 1
    any_satspots = any(star['satspots']['satspots_are_present'] for star in stars.values())
    n_psf_states = 4 if any_satspots else 1

    # The detector round runs all of a visit's frames at once: check against the largest visit
    largest_visit = max(frames_in_visit(visit, stars[visit['star']]) for visit in visits)

    # With MPI every worker rank is its own process with its own CPUs, so check one rank (n_jobs=1)
    is_mpi = args.option == "mpi"
    n_jobs = 1 if is_mpi else args.n

    report = get_cpu_allocation(n_jobs=n_jobs, nlam=nlam, n_psf_states=n_psf_states,
                                n_frames=largest_visit, warn=False)
    log.info("CPUs: %d available (%s); per PSF PROPER opens %d processes (nlam=%d); "
             "peak %d processes in the PSF round, %d in the detector round%s",
             report['cpus'], report['cpus_source'], report['proper_pool'], nlam,
             report['peak_cpu_count'], report['detector_peak'],
             " per MPI rank" if is_mpi else "")
    for problem in report['problems']:
        log.warning(problem)

    # One sub-seed per visit, so visits don't repeat each other's noise (None: noise pattern not reproducible)
    if args.seed is not None:
        visit_seeds = np.random.SeedSequence(args.seed).spawn(len(visits))
    else:
        visit_seeds = [None] * len(visits)

    try:
        for visit, visit_seed in zip(visits, visit_seeds):
            star = stars[visit['star']]
            visit_id = visit['visit_id']
            log.info("Visit %s: %s star, roll %s deg, %d frame(s) (%d science, exptime %s s)",
                     visit_id, visit['star'], visit['roll_angle'], frames_in_visit(visit, star),
                     visit['n_frames'], visit['exp_time'])
            visit_start = time.perf_counter()
            # One CorgiOptics instance per visit: the roll angle and visit_id are set on it
            optics_i = instrument.CorgiOptics(obs['cgi_mode'], obs['bandpass'], optics_keywords=obs['optics_keywords'],
                                              roll_angle=visit['roll_angle'], visit_type=visit['vistype'],
                                              visit_id=visit_id, if_quiet=True)

            # Frames run in parallel, come back to this (manager) process and are
            # written here; writing as L1 needs full-frame images
            visit_dir = os.path.join(args.output_dir, 'V' + visit_id)
            os.makedirs(visit_dir, exist_ok=True)
            observation.generate_observation_sequence(star['scene'], optics_i, star['detector'],
                                                      visit['exp_time'], visit['n_frames'],
                                                      **star['satspots'],
                                                      vistype=visit['vistype'], visit_id=visit_id,
                                                      save_as_fits=True, output_dir=visit_dir,
                                                      full_frame=True, loc_x=obs['loc_x'], loc_y=obs['loc_y'],
                                                      frame_runner=frame_runner, seed=visit_seed)
            # The returned frames are not kept, so they are freed before the next visit
            log.info("Visit %s done in %.1f s (PSFs + frames + save)", visit_id, time.perf_counter() - visit_start)
    finally:
        if args.option == "mpi":
            shutdown_workers(comm)

    log.info("Done: %d visits written to %s in %.1f s", len(visits), args.output_dir, time.perf_counter() - run_start)
