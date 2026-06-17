### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 
import corgisim
import os
from corgisim import scene, instrument, inputs, observation, outputs
import copy 

def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, save_as_fits= False, output_dir=None, full_frame= False, loc_x=None, loc_y=None, n_satspot_frames=None):
    """Generates a sequence of simulated observations and places them on a detector.

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
        n_satspot_frames (int, optional): Number of frames at the beginning of
            the sequence that should include satellite spots. This must be a
            multiple of 3, because each set of satellite-spot frames uses
            [negative, positive, no sign override]. To use this, set
            `optics.satspot_keywords` before calling this function. If None,
            the sequence uses the optics object as configured.
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
    def generate_sim_scene():
        sim_scene = optics.get_host_star_psf(scene)
        if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
            sim_scene = optics.inject_point_sources(scene,sim_scene)
        return sim_scene

    if n_satspot_frames is not None:
        if not isinstance(n_satspot_frames, int):
            raise TypeError("n_satspot_frames must be an integer.")
        if n_satspot_frames < 0:
            raise ValueError("n_satspot_frames cannot be negative.")
        if n_satspot_frames > n_frames:
            raise ValueError("n_satspot_frames cannot exceed n_frames.")
        if n_satspot_frames % 3 != 0:
            raise ValueError("n_satspot_frames must be a multiple of 3.")

    original_has_dm1_v = 'dm1_v' in optics.optics_keywords
    original_dm1_v = optics.optics_keywords.get('dm1_v')
    original_satspots = optics.SATSPOTS

    if n_satspot_frames is None:
        regular_sim_scene = generate_sim_scene()
        satspot_sim_scenes = {}
    else:
        if original_satspots == 1:
            raise ValueError(
                "n_satspot_frames requires an optics object that starts without "
                "satellite spots. Set optics.satspot_keywords after creating "
                "the optics object, then pass n_satspot_frames."
            )

        satspot_sim_scenes = {}
        regular_sim_scene = None
        satspot_signs = [None, "positive", "negative"]

        try:
            if n_satspot_frames > 0:
                satspot_keywords = getattr(optics, 'satspot_keywords', None)
                if satspot_keywords is None:
                    raise ValueError(
                        "n_satspot_frames was set, but optics.satspot_keywords is None."
                    )
                if optics.optics_keywords.get('use_dm1') != 1:
                    raise KeyError('ERROR: use_dm1 in optics_keywords is not set 1')

                for sign in satspot_signs:
                    satspot_keywords_for_frame = satspot_keywords.copy()
                    
                    if sign is None:
                        optics.optics_keywords['dm1_v'] = original_dm1_v
                    else:
                        satspot_keywords_for_frame['sign'] = sign
                        optics.optics_keywords['dm1_v'] = original_dm1_v
                        optics.optics_keywords['dm1_v'] = optics.add_satspot(
                            satspot_keywords=satspot_keywords_for_frame )
                    optics.SATSPOTS = int(1)
                    satspot_sim_scenes[sign] = generate_sim_scene()

            if n_satspot_frames < n_frames: 
                optics.optics_keywords['dm1_v'] = original_dm1_v
                optics.SATSPOTS = int(0)
                regular_sim_scene = generate_sim_scene()
        finally:
            #changing optics.optics_keywords['dm1_v'] and optics.SATSPOTS back to original values
            if original_has_dm1_v:
                optics.optics_keywords['dm1_v'] = original_dm1_v
            else:
                optics.optics_keywords.pop('dm1_v', None)
            optics.SATSPOTS = original_satspots
    
    simulatedImage_list = []
    
    if full_frame == False :
        for i in range(0, n_frames):
            if n_satspot_frames is not None and i < n_satspot_frames:
                frame_scene = satspot_sim_scenes[satspot_signs[i % len(satspot_signs)]]
            else:
                frame_scene = regular_sim_scene
            sim_image = detector.generate_detector_image(frame_scene,exp_time)
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
            if n_satspot_frames is not None and i < n_satspot_frames:
                frame_scene = satspot_sim_scenes[satspot_signs[i % len(satspot_signs)]]
            else:
                frame_scene = regular_sim_scene
            sim_image = detector.generate_detector_image(frame_scene,exp_time,full_frame=True,loc_x=loc_x, loc_y=loc_y)
            simulatedImage_list.append(copy.deepcopy(sim_image))

            if save_as_fits:
                outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir, write_as_L1=True)

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath, save_as_fits= False, output_dir=None, full_frame=False, loc_x=None, loc_y=None, point_source_info=None):
    """Generates an observation scenario by loading instrument, scene, and visit
    information from a CPGS file.

    This function attempts to load both target and reference star information.
    If only target information is available, it proceeds with that.

    Args:
        - filepath (str): The path to the CPGS XML file.
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
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(filepath)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list = inputs.load_cpgs_data(filepath)

    if point_source_info is not None:
        host_star_properties = {'Vmag': scene_target._host_star_Vmag, 'spectral_type': scene_target._host_star_sptype, 'magtype': scene_target._host_star_magtype, 'ref_flag': False}
        scene_target = scene.Scene(host_star_properties, point_source_info)

    for visit in visit_list:
        optics.roll_angle = visit['roll_angle']
        if visit['isReference']:
            simulatedImage_visit = generate_observation_sequence(scene_reference, optics, detector_reference, visit['exp_time'], visit['number_of_frames'],save_as_fits= save_as_fits, output_dir=output_dir, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y )
        else:
            simulatedImage_visit = generate_observation_sequence(scene_target, optics, detector_target, visit['exp_time'], visit['number_of_frames'],save_as_fits= save_as_fits, output_dir=output_dir, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y  )

        simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list

    
