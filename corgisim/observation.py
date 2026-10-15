### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 
import corgisim
import os
from corgisim import scene, instrument, inputs, outputs
from corgisim.parallel_executor import run_parallel
from corgidrp import mocks

import copy
from datetime import datetime, timedelta

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
    """Default frame runner: frames run one after another in this process."""
    return run_parallel(frame_func, args_list, n_jobs=1)

def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, vistype = 'CGIVST_TDD_OBS', visit_id= '0200001001001001001', save_as_fits= False, output_dir=None, full_frame= False, loc_x=None, loc_y=None, zindex=None, zval_m_i=None, frame_runner=None):
    """
    Generates a sequence of simulated observations and places them on a detector.

    This function orchestrates the simulation of a given astrophysical scene through
    the instrument optics and then onto the detector. It first generates the host star's
    PSF, then injects any defined off-axis point sources into the simulated scene.
    Finally, it uses the detector model to create a detector image, optionally
    generating either a sub-array or a full-frame image for each exposure.
    Each observation sequence corresponds to a single visit at a specific roll angle.

    Args:
        scene (corgisim.scene.Scene): The scene object containing information about
            the host star and any specified point sources.
        optics (corgisim.instrument.CorgiOptics): The optics object defining the
            instrument configuration, including the telescope and coronagraph.
        detector (corgisim.instrument.CorgiDetector): The detector object defining
            the detector characteristics and noise properties.
        exp_time (float): The exposure time for each individual frame in seconds.
        n_frames (int): The total number of frames to generate in this observation sequence.
        save_as_fits (bool): wether or not to save the generated images as fits file
        output_dir: if saving as fits file, where to save them 
        full_frame (bool, optional): If True, a full-frame detector image will be generated.
            If False (default), a sub-array image is generated.
        loc_x (int, optional): The x-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            x-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.
        loc_y (int, optional): The y-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            y-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.        

    Returns:
        list[corgisim.scene.SimulatedImage]: A list of :py:class:`corgisim.scene.SimulatedImage` objects,
        where each object represents a single generated observation frame with its image data
        and associated FITS header information.
    """
    optics.visit_id = visit_id
    optics.visit_type = vistype
    sim_scene = optics.get_host_star_psf(scene)
    if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
        sim_scene = optics.inject_point_sources(scene,sim_scene)
    
    simulatedImage_list = []
    
    if full_frame == False :
        for i in range(0, n_frames):
            sim_image = detector.generate_detector_image(sim_scene,exp_time)
            simulatedImage_list.append(copy.deepcopy(sim_image))
    else:
        if save_as_fits:
            # Save the images as fits in output_dir if specified, in corgisim/test/testdata if not
            # Simulation needs to be full frame to be written as L1
            if output_dir == None:
                local_path = corgisim.lib_dir
                outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
                print("No output directory specified. FITS files saved in ", outdir)
            else:
                outdir = output_dir

        for i in range(0, n_frames):
            sim_image = detector.generate_detector_image(sim_scene,exp_time,full_frame=True,loc_x=loc_x, loc_y=loc_y)
            simulatedImage_list.append(copy.deepcopy(sim_image))

            if save_as_fits:
                outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=True)

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath, save_as_fits= False, save_as_list=False, output_dir=None, full_frame=False, loc_x=None, loc_y=None, point_source_info=None, output_dim=201, polaxis=0, fast_gain_mode='auto', gain_CIC_Q='roman' ):
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
                             
    Returns:
        - list[corgisim.scene.SimulatedImage]: A list of SimulatedImage objects, representing the complete observation scenario across all visits defined in the CPGS file.
    """
    # Get the detector, scene and optics used in generate obeservation sequence from CPGS file
    simulatedImage_list = []
    # Try to get target and reference
    try:
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list,satellite_dict_target, satellite_dict_reference, = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list, satellite_dict_target = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)

    if point_source_info is not None:
        host_star_properties = {'Vmag': scene_target._host_star_Vmag, 'spectral_type': scene_target._host_star_sptype, 'magtype': scene_target._host_star_magtype, 'ref_flag': False}
        scene_target = scene.Scene(host_star_properties, point_source_info)

    #Satellit spot configuration
    satspots_are_present = (satellite_dict_target is not None) 
    if satellite_dict_target is not None:
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

    if satspots_are_present: 
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
             # Generate satellite spot images, if any
            if satspots_are_present:
                # For each frame, we take background (no satellite spots), positive and negative
                # Background 
                optics.SATSPOTS = 1
                simulatedImage_visit_satspots = generate_observation_sequence(scene_reference, optics,detector_satspots_reference, satellite_dict_reference['satellite_spots_frame_time'], satellite_dict_reference['satellite_spots_number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
                simulatedImage_visit.extend(simulatedImage_visit_satspots)

                for sign in ["positive", "negative"]:
                    satspot_keywords["sign"] = sign
                    optics.add_satspot(satspot_keywords=satspot_keywords)
                    simulatedImage_visit_satspots = generate_observation_sequence(scene_reference, optics,detector_satspots_reference, satellite_dict_reference['satellite_spots_frame_time'], satellite_dict_reference['satellite_spots_number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
                    simulatedImage_visit.extend(simulatedImage_visit_satspots)
                    optics.remove_satspot(satspot_keywords=satspot_keywords)

            simulatedImage_visit_sci = generate_observation_sequence(scene_reference, optics, detector_reference, visit['exp_time'], visit['number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
            simulatedImage_visit.extend(simulatedImage_visit_sci)
        else:
            # Generate satellite spot images, if any
            if satspots_are_present:
                # For each frame, we take background (no satellite spots), positive and negative
                # Background 
                optics.SATSPOTS = 1
                simulatedImage_visit_satspots = generate_observation_sequence(scene_target, optics,detector_satspots_target, satellite_dict_target['satellite_spots_frame_time'], satellite_dict_target['satellite_spots_number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
                simulatedImage_visit.extend(simulatedImage_visit_satspots)

                for sign in ["positive", "negative"]:
                    satspot_keywords["sign"] = sign
                    optics.add_satspot(satspot_keywords=satspot_keywords)
                    simulatedImage_visit_satspots = generate_observation_sequence(scene_target, optics,detector_satspots_target, satellite_dict_target['satellite_spots_frame_time'], satellite_dict_target['satellite_spots_number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
                    simulatedImage_visit.extend(simulatedImage_visit_satspots)
                    optics.remove_satspot(satspot_keywords=satspot_keywords)

            simulatedImage_visit_sci = generate_observation_sequence(scene_target, optics, detector_target, visit['exp_time'], visit['number_of_frames'], visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y  )
            simulatedImage_visit.extend(simulatedImage_visit_sci)

        if not save_as_fits or (save_as_fits and save_as_list): # If we are writing the files, we are not storing the images
            simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list

    