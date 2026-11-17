### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 
import corgisim
import os
from corgisim import scene, instrument, inputs, outputs
from corgisim.parallel_executor import run_parallel
from corgidrp import mocks

import copy
import warnings
from datetime import datetime, timedelta

# TODO - add a helper function to help user determine how many cores they should assign for the runs.
# TODO - add an example.sh to demostrate how it is being launhced (mpi mode.)

def _apply_detector(sim_scene, detector, exp_time, full_frame, loc_x, loc_y):
    """Generate emccd frames on already defined simulated scene and optics.

    Args:
        sim_scene: output of optics.get_host_star_psf()/inject_point_sources().
        detector (corgisim.instrument.CorgiDetector): detector model to apply.
        exp_time (float): exposure time in seconds.
        full_frame (bool): if True, place the result in a full detector frame.
        loc_x, loc_y (int or None): sub-array/full-frame placement, see
            `generate_observation_sequence`.

    Returns:
        corgisim.scene.SimulatedImage: the generated frame.
    """
    if full_frame:
        sim_image = detector.generate_detector_image(sim_scene, exp_time, full_frame=True, loc_x=loc_x, loc_y=loc_y)
    else:
        sim_image = detector.generate_detector_image(sim_scene, exp_time)

    return copy.deepcopy(sim_image)


def _generate_one_frame(scene, optics, detector, exp_time, full_frame, loc_x, loc_y, zindex, zval_m_i):
    """Generate a frame that needs updated optics keywords (e.g. a distinct
    zernike/drift value for this frame), which triggers get_host_star_psf after the update.
    This is a unit dispatched to `run_parallel` when frames don't share the same optics keywords.
    See `_apply_detector` for the fixed optics keywords case. 
    
    Args:
        scene (corgisim.scene.Scene): output astrophysical scene.
        optics (corgisim.instrument.CorgiOptics): base optics, before this
            frame's zindex/zval_m override. Not modified.
        detector (corgisim.instrument.CorgiDetector): detector instance. 
        exp_time (float): exposure time in seconds.
        full_frame (bool): see `generate_observation_sequence`.
        loc_x, loc_y (int or None): see `generate_observation_sequence`.
        zindex (list of int): zernike noll indices for this frame's.
        zval_m_i (numpy 1d array): zernike values (meters) for this frame.

    Returns:
        corgisim.scene.SimulatedImage: the generated frame.
    """

    optics_keywords = optics.optics_keywords.copy()
    if zindex is not None:
        optics_keywords.update({'zindex': zindex, 'zval_m': zval_m_i})

    # Copy the optics rather than rebuilding it: CorgiOptics.__init__ rejects its
    # own internal keywords (e.g. final_sampling_m), and a copy keeps the rest of
    # the state (SATSPOTS, visit_id, ...)
    optics = copy.copy(optics)
    optics.optics_keywords = optics_keywords

    sim_scene = optics.get_host_star_psf(scene)
    if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
        sim_scene = optics.inject_point_sources(scene, sim_scene)

    return _apply_detector(sim_scene, detector, exp_time, full_frame, loc_x, loc_y)

def prepare_frame_tasks(scene, optics, detector, exp_time, n_frames, vistype = 'CGIVST_TDD_OBS', visit_id= '0200001001001001001', full_frame= False, loc_x=None, loc_y=None, zindex=None, zval_m_i=None):
    """Build the per-frame work for one observation sequence, without running it.

    If all frames share one optics state (`zindex is None`), the host star PSF is
    computed here, once, and every task is only the detector step (`_apply_detector`).
    Otherwise every task is one PROPER run plus detector step (`_generate_one_frame`).

    Args:
        See `generate_observation_sequence`.

    Returns:
        tuple: (frame_func, args_list), where frame_func(*args_list[i]) returns frame i
        as a corgisim.scene.SimulatedImage. frame_func is a module-level function, so the
        tasks can be sent to worker processes or MPI ranks.
    """
    optics.visit_id = visit_id
    optics.visit_type = vistype

    if zindex is None:
        # All frames share one optics state: compute the PSF once, only the detector noise changes per frame
        sim_scene = optics.get_host_star_psf(scene)
        if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
            sim_scene = optics.inject_point_sources(scene, sim_scene)

        args_list = [(sim_scene, detector, exp_time, full_frame, loc_x, loc_y) for i in range(n_frames)]
        return _apply_detector, args_list

    # Each frame has its own optics state (zval_m_i[i]), so each frame is its own PROPER run
    args_list = [(scene, optics, detector, exp_time, full_frame, loc_x, loc_y, zindex, zval_m_i[i]) for i in range(n_frames)]
    return _generate_one_frame, args_list

def _run_frames_local(frame_func, args_list):
    """Default frame runner: frames run one after another in this process, no parallel processing. Used if frame_runner is None."""
    return run_parallel(frame_func, args_list, n_jobs=1)

def generate_observation_sequence(scene, optics, detector, exp_time, n_frames,
                                  satspots_are_present=False, satspots_detector=None, satspots_exptime=None, satspots_number_of_frames=None, satspot_keywords=None,
                                  vistype = 'CGIVST_TDD_OBS', visit_id= '0200001001001001001', save_as_fits= False, output_dir=None, full_frame= False, loc_x=None, loc_y=None, zindex=None, zval_m_i=None, frame_runner=None):
    """
    Generates a sequence of simulated observations and places them on a detector.

    This function setups the simulation of a given astrophysical scene through
    the instrument optics and then onto the detector. It first generates the host star's PSF, then injects any defined off-axis point sources into the simulated scene. 
    
    Finally, it uses the detector model to create a detector image, optionally
    generating either a sub-array or a full-frame image for each exposure. Each observation sequence corresponds to a single visit at a specific roll angle.

    A `CorgiOptics` instance is one fixed optics state (DM settings, zernike/drift
    value, etc.) -- a different state usually requires a new instance. So frames
    either all share one state (e.g. `zindex is None`: the PSF is identical for
    every frame, computed once) or each needs its own (e.g. `zindex is not None`: a distinct `zval_m_i[i]` per frame).

    Parallelisation: the sequence is split into segments, one per optics state:
        satspot: positive, negative, background; only if `satspots_are_present`
        science: the rest of frames are science frames. 
        
    Segments run one after another, because add_satspot/remove_satspot change `optics` between them. Within a segment, `prepare_frame_tasks` builds one task per frame and `frame_runner` runs these tasks in parallel:
        - `zindex is None`: the PSF is computed once, in this process, and
          only the detector step (`_apply_detector`) runs in parallel.
        - `zindex is not None`: every task is a PROPER run plus the detector
          step (`_generate_one_frame`), so the PROPER runs are parallel too.
    
    The frames of all segments are collected in order (satspots first, then
    science) and, if `save_as_fits`, written to disk before this function
    returns, so a caller looping over visits only holds one visit's frames at a time.

    Args:
        scene (corgisim.scene.Scene): The scene object containing information about
            the host star and any specified point sources.
        optics (corgisim.instrument.CorgiOptics): The optics object defining the
            instrument configuration, including the telescope and coronagraph.
        detector (corgisim.instrument.CorgiDetector): The detector object defining
            the detector characteristics and noise properties.
        exp_time (float): The exposure time for each individual frame in seconds.
        n_frames (int): The number of frames to generate in this observation sequence, satellite spots frames not included.
        satspots_present (bool): If yes, the sequence will generate 3xsatspots_number_of_frames satellite spots frames 
        satspots_detector(corgisim.instrument.CorgiDetector): The detector object defining
            the detector characteristics and noise properties used for satellite spots
        satspots_exptime (float): The exposure time for each individual satspots frame in seconds.
        satspots_number_of_frames (int): The number of satellite spots frames OF EACH TYPE (backgroup, positive, negative) to generate; The sequence will contain 3*satspots_number_of_frames
        satspot_keywords:  A dictionary with the keywords that are used to add satellite spots. See instruments. py add_satspot for the keywords.
        save_as_fits (bool): wether or not to save the generated images as fits file
        vistype (str): A string indicating the type of visit  for populating header VISTYPE.
        visit_id (str): A string indicating the visit ID for populating header VISITID.
        output_dir: if saving as fits file, where to save them 
        full_frame (bool, optional): If True, a full-frame detector image will be generated.
            If False (default), a sub-array image is generated.
        loc_x (int, optional): The x-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            x-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.
        loc_y (int, optional): The y-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            y-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.
        zindex (list of int, optional): Zernike Noll indices as `zindex` in the
            optics keywords. Default None (all frames share the input optics keywords).
        zval_m_i (array, optional): Zernike values in meters, one row per frame
            (shape n_frames x len(zindex)), passed as `zval_m`. Required if `zindex` is given.
        frame_runner (callable, optional): frame_runner(frame_func, args_list) runs the frame
            tasks and returns the frames in order. Default None runs them one after another
            in this process (serial). For n local worker processes pass
            functools.partial(corgisim.parallel_executor.run_parallel, n_jobs=n, allow_nesting=True);
            for MPI pass functools.partial(corgisim.mpi_runtime.run_tasks_mpi, comm).

    Returns:
        list[corgisim.scene.SimulatedImage]: A list of :py:class:`corgisim.scene.SimulatedImage` objects,
        where each object represents a single generated observation frame with its image data
        and associated FITS header information.
    """
    if save_as_fits == False: 
        warnings.warn(
                    "save_as_fits is set to False, so files are not automatically written to the disk. "
                    "If you want to re-use these files later, save them using outputs.save_hdu_to_fits ",
                    UserWarning,
                )
    if satspots_are_present and (satspots_detector is None or satspots_exptime is None or satspots_number_of_frames is None or satspot_keywords is None):
        raise ValueError('Satellite spots are presents but arguments are missing. Please specify satspot keywords, detector, exposure time and number of frames' )

    if frame_runner is None:
        # Serial frame generation in this process, no parallel processing. 
        frame_runner = _run_frames_local

    def run_segment(seg_detector, seg_exp_time, seg_n_frames):
        # Prepare the per-frame tasks (function + args) for one optics state, then run them and collect the results.
        # Each segment is run before the optics state is changed for the next one (e.g. add/remove_satspot). 
        frame_func, args_list = prepare_frame_tasks(scene, optics, seg_detector, seg_exp_time, seg_n_frames, vistype=vistype, visit_id=visit_id, full_frame=full_frame, loc_x=loc_x, loc_y=loc_y, zindex=zindex, zval_m_i=zval_m_i)
        return frame_runner(frame_func, args_list)


    # Exposure time of each frame in simulatedImage_list, used to give each saved frame its own FTIMEUTC
    # satspot and science frames can have different exposure times
    simulatedImage_list = [] 
    frame_exp_times = [] # satspot and science frames can have different exposure times
    if satspots_are_present:
        # For each frame, we take background (no satellite spots), positive and negative
        # no satspot frames
        optics.SATSPOTS = 1
        simulatedImage_list.extend(run_segment(satspots_detector, satspots_exptime, satspots_number_of_frames))
        frame_exp_times.extend([satspots_exptime] * satspots_number_of_frames)

        for sign in ["positive", "negative"]:
            satspot_keywords["sign"] = sign
            optics.add_satspot(satspot_keywords=satspot_keywords)
            simulatedImage_list.extend(run_segment(satspots_detector, satspots_exptime, satspots_number_of_frames))
            frame_exp_times.extend([satspots_exptime] * satspots_number_of_frames)
            optics.remove_satspot(satspot_keywords=satspot_keywords) # also resets optics.SATSPOTS to 0

    simulatedImage_list.extend(run_segment(detector, exp_time, n_frames))
    frame_exp_times.extend([exp_time] * n_frames)

    if save_as_fits:
        outdir = output_dir

        if outdir is None:
            local_path = corgisim.lib_dir
            outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
            print("No output directory specified. FITS files saved in ", outdir)

        # Frames generated in parallel are stamped with the same wall-clock FTIMEUTC,
        # which would give them the same L1 filename. Give frame i the start time of
        # the i-th exposure of the sequence instead: t0 plus the sum of the exposure times of
        # the frames before it (save_hdu_to_fits also updates FILENAME and the frame's own header).
        t0 = datetime.fromisoformat(simulatedImage_list[0].image_on_detector[1].header['FTIMEUTC'])
        elapsed = 0.
        for sim_image, frame_exp_time in zip(simulatedImage_list, frame_exp_times):
            ftimeutc = (t0 + timedelta(seconds=elapsed)).isoformat()
            elapsed += frame_exp_time
            outputs.save_hdu_to_fits(sim_image.image_on_detector, outdir=outdir, write_as_L1=True,
                                     overwrite_ext_keywords={'FTIMEUTC': ftimeutc})

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath, save_as_fits= False, save_as_list=False, output_dir=None, full_frame=False, loc_x=None, loc_y=None, point_source_info=None, output_dim=201, polaxis=0, fast_gain_mode='auto', gain_CIC_Q='roman', frame_runner=None):
    """Generates an observation scenario by loading instrument, scene, and visit
    information from a CPGS file.

    This function attempts to load both target and reference star information.
    If only target information is available, it proceeds with that.

    Args:
        - filepath (str): The path to the CPGS XML file.
        - save_as_fits (bool): wether or not to save the generated images as fits file
        - save_as_list (bool): wether or not to save the generated images as list of file. In order to save memory, default behaviour is not to save as a list if we're saving as fits
        - output_dir: if saving as fits file, where to save them 
        - full_frame (bool, optional): If True, a full-frame detector image will be generated.
        - loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - point_sources_info (list): A list of dictionaries, each representing an off-axis point source in the scene.
        - frame_runner (callable, optional): runs each visit's frames in parallel, see
          `generate_observation_sequence`. Visits are still run one after another, and each
          visit's frames are saved (if save_as_fits) before the next visit starts.

    Returns:
        - list[corgisim.scene.SimulatedImage]: A list of SimulatedImage objects, representing the complete observation scenario across all visits defined in the CPGS file.
    """
    # Get the detector, scene and optics used in generate obeservation sequence from CPGS file
    simulatedImage_list = []
    # Try to get target and reference
    try:
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list,satellite_dict_target, satellite_dict_reference = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list, satellite_dict_target = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)

    if point_source_info is not None:
        host_star_properties = {'Vmag': scene_target._host_star_Vmag, 'spectral_type': scene_target._host_star_sptype, 'magtype': scene_target._host_star_magtype, 'ref_flag': False}
        scene_target = scene.Scene(host_star_properties, point_source_info)

    # Satellite spot configuration
    satspots_are_present = (satellite_dict_target is not None) 
    if satspots_are_present:
        contrast1 = 1e-7
        contrast2 = 1e-5
        sep1 = 6.25
        sep2 = 13
        match satellite_dict_target['satellite_spot_conf']:
            case 0:
                sep_lamD = sep1
                angle_deg = [0,90]
                contrast = contrast1
            case 1:
                sep_lamD = sep2
                angle_deg = [0,90]
                contrast = contrast1
            case 2:
                sep_lamD = sep1
                angle_deg = [45,135]
                contrast = contrast1
            case 3:
                sep_lamD = sep2
                angle_deg = [45,135]
                contrast = contrast1
            case 4:
                sep_lamD = sep1
                angle_deg = [0,90]
                contrast = contrast2
            case 5:
                sep_lamD = sep2
                angle_deg = [0,90]
                contrast = contrast2
            case 6:
                sep_lamD = sep1
                angle_deg = [45,135]
                contrast = contrast2
            case 7:
                sep_lamD = sep2
                angle_deg = [45,135]
                contrast = contrast2
            case _:
                raise KeyError('Unknown satellite spots configuration')

        satspot_keywords = {'num_pairs':2, 'sep_lamD': sep_lamD, 'angle_deg': angle_deg, 'contrast': contrast}

        detector_satspots_target = instrument.CorgiDetector(emccd_keywords={'em_gain':satellite_dict_target['satellite_spots_gain']}, photon_counting=False) 
        if 'satellite_dict_reference' in locals():
            detector_satspots_reference = instrument.CorgiDetector(emccd_keywords={'em_gain':satellite_dict_reference['satellite_spots_gain']}, photon_counting=False) 

    prihdr, exthdr = mocks.create_default_L1_headers()
    visit_id_root = prihdr['PROGNUM']+ prihdr['EXECNUM']+prihdr['CAMPAIGN']+prihdr['SEGMENT']+prihdr['OBSNUM']
    for visit in visit_list:
        visit_id = visit_id_root + visit['visit_num']
        # make a new subfolder for each visit 
        subfolder_name = 'V' + visit_id
        subfolder_path =  os.path.join(str(output_dir or ''), subfolder_name)
        if save_as_fits:
            os.makedirs(subfolder_path, exist_ok=True)

        optics.roll_angle = visit['roll_angle']
        simulatedImage_visit = []
        if visit['isReference']:
            simulatedImage_visit = generate_observation_sequence(scene_reference, optics, detector_reference,  visit['exp_time'], visit['number_of_frames'], 
                                                                satspots_are_present = satspots_are_present, satspots_detector = detector_satspots_reference, satspots_exptime = satellite_dict_reference['satellite_spots_frame_time'], satspots_number_of_frames = satellite_dict_reference['satellite_spots_number_of_frames'], satspot_keywords = satspot_keywords,
                                                                visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y, frame_runner=frame_runner)

        else:
            simulatedImage_visit = generate_observation_sequence(scene_target, optics, detector_target,  visit['exp_time'], visit['number_of_frames'], 
                                                                satspots_are_present = satspots_are_present, satspots_detector = detector_satspots_target, satspots_exptime = satellite_dict_target['satellite_spots_frame_time'], satspots_number_of_frames = satellite_dict_target['satellite_spots_number_of_frames'], satspot_keywords = satspot_keywords,
                                                                visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y, frame_runner=frame_runner)


        if not save_as_fits or (save_as_fits and save_as_list): # If we are writing the files, we are not storing the images unless explicitly told to do so
            simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list

    