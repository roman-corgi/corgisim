### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 

from corgisim import scene, instrument, inputs, observation

def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, full_frame= False, loc_x=None, loc_y=None):
    '''
    """
    Generates a sequence of observations for a given scene, instrument configuration,
    and detector configuration.

    One observation sequence represents a single visit at a specific roll angle.

    :param scene: The scene object containing information about the host star and point sources.
    :type scene: corgisim.scene.Scene
    :param optics: The optics object defining the instrument configuration.
    :type optics: corgisim.instrument.CorgiOptics
    :param detector: The detector object defining the detector characteristics.
    :type detector: corgisim.instrument.CorgiDetector
    :param exp_time: The exposure time for each frame in seconds.
    :type exp_time: float
    :param n_frames: The number of frames to generate in the sequence.
    :type n_frames: int
    :param full_frame: If True, generate a full-frame detector image. Defaults to False.
    :type full_frame: bool, optional
    :param loc_x: The x-coordinate for the center of the sub-array if `full_frame` is False.
                  Required if `full_frame` is True.
    :type loc_x: int, optional
    :param loc_y: The y-coordinate for the center of the sub-array if `full_frame` is False.
                  Required if `full_frame` is True.
    :type loc_y: int, optional
    :return: A list of SimulatedImage objects, each representing a generated observation frame.
    :rtype: list[corgisim.scene.SimulatedImage]
    '''
    sim_scene = optics.get_host_star_psf(scene)
    if hasattr(scene, 'point_source_x'):
        sim_scene = optics.inject_point_sources(scene,sim_scene)
    
    simulatedImage_list = []
    
    if full_frame == False :
        for i in range(0, n_frames):
            sim_image = detector.generate_detector_image(sim_scene,exp_time)
            simulatedImage_list.append(sim_image)
    else:
        for i in range(0, n_frames):
            sim_image = detector.generate_detector_image(sim_scene,exp_time,full_frame=True,loc_x=loc_x, loc_y=loc_y)
            simulatedImage_list.append(sim_image)

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath):
    """
    Generates an observation scenario by loading instrument, scene, and visit
    information from a CPGS file.

    This function attempts to load both target and reference star information.
    If only target information is available, it proceeds with that.

    :param filepath: The path to the CPGS XML file.
    :type filepath: str
    :return: A list of SimulatedImage objects, representing the complete observation
             scenario across all visits defined in the CPGS file.
    :rtype: list[corgisim.scene.SimulatedImage]
    """
    # Get the detector, scene and optics used in generate obeservation sequence from CPGS file
    simulatedImage_list = []
    # Try to get target and reference
    try:
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(filepath)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list = inputs.load_cpgs_data(filepath)


    for visit in visit_list:
        #optics.roll_angle = visit['roll_angle'] Commented out for now
        if visit['isReference']:
            simulatedImage_visit = generate_observation_sequence(scene_reference, optics, detector_reference, visit['exp_time'], visit['number_of_frames'] )
        else:
            simulatedImage_visit = generate_observation_sequence(scene_target, optics, detector_target, visit['exp_time'], visit['number_of_frames'] )

        simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list

    