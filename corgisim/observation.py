### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 


def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, full_frame= False, loc_x=None, loc_y=None):
    '''
    Function that generates a sequence of observations for a given scene, instrument configuration, and detector configuration. 
    One observation sequence is one visit at one roll angle
    Arguments: 
    scene

    return list of HDUList
    '''
    sim_scene = optics.get_host_star_psf(scene)
    if hasattr(scene, 'point_source_x'):
        sim_scene = optics.inject_point_sources(base_scene,sim_scene)
    
    hdu_list = []
    if full_frame == False :
        sim_image = detector.generate_detector_image(sim_scene,exp_time)
    else:
        sim_image = detector.generate_detector_image(sim_scene,exptime,full_frame=True,loc_x=loc_x, loc_y=loc_y)
    for i in range[1, n_frames]:
        hdu_list.append[sim_image]

    return hdu_list

def generate_observation_scenario(roll_angles, from_cpgs=False):

    # Get the detector, scene and optics used in generate obeservation sequence
    if from_cpgs: 
        hdu_list = []
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(filepath)
        
        for visit in visit_list:
           #optics.roll_angle = visit['roll_angle']
            if visit['isReference']:
                generate_observation_sequence(scene_reference, optics, detector_reference, number_of_frames )
            else:
                generate_observation_sequence(scene_target, optics, detector_target, number_of_frames )


    return hdu_list