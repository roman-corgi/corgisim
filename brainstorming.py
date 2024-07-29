


def generate_L1_data(CPGS_xml, scene_info, proper_arguments, emccd_detect_arguments): 
    '''

    A function that returns a set of L1 data for the given inputs

    The CPGS_xml file should contain: 
    - The instrument configuration used
    - The exposure time
    - The number of exposures
    - To add more

    The scene_info dictionary should contain: 
    - Information about the host star (brightness, spectral type, etc.)
    - A list of point sources (brightness, location, etc.)
    - A 2D background scene that will be convolved with the off-axis PSFs
    - TO ADD MORE? 

    The proper_arguments dictionary should contain:
    - Any important arguments for PROPER that aren't covered by the CPGS_xml file
    - Optional arguments for users to input their own WFE maps and/or DM configuration
    
    The emccd_detect_arguments dictionary should contain:
    - Any important arguments for the EMCCD detector that aren't covered by the CPGS_xml file
    - Optional arguemnts to change default setup (e.g. adding read noise, changing gain, etc.)

    Arguments: 
    CPGS_xml: A string that contains the path to the CPGS xml file
    scene_info: A dictionary that contains the scene information
    proper_arguments: A dictionary that contains the proper arguments
    emccd_detect_arguments: A dictionary that contains the emccd detect arguments

    Returns: 
    A list of astropy HDULists
    
    '''

    convolved_scene = proper_convolve_2d_scene(scene_info, proper_arguments)

    scene_with_sources = place_point_sources(convolved_scene, scene_info)

    scene_with_psf = generate_host_star_psf(scene_with_sources, scene_info, proper_arguments)

    emccd_scene = simulate_detector(scene_with_psf, emccd_detect_arguments)

    #TODO: Look into when to split into multiple scenes. 

    return emccd_scene





