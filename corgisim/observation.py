### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 

from corgisim import scene, instrument, inputs, observation

def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, full_frame= False, loc_x=None, loc_y=None):
    '''
    Function that generates a sequence of observations for a given scene, instrument configuration, and detector configuration. 
    One observation sequence is one visit at one roll angle
    Arguments: 
    scene

    return list of SimulatedImage
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

    