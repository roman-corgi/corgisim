


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
    scene_info: An astropy HDU that contains the scene information
    proper_arguments: A dictionary that contains the proper arguments
    emccd_detect_arguments: A dictionary that contains the emccd detect arguments

    Returns: 
    A list of astropy HDULists
    
    '''

    cpgs_info = open_CPGS_dict(cpgs_xml)

    convolved_scene = convolve_2d_scene(scene_info, proper_arguments, cpgs_info)

    scene_with_sources = place_point_sources(convolved_scene, scene_info, proper_arguments, cpgs_info)

    scene_with_psf = generate_host_star_psf(scene_with_sources, scene_info, proper_arguments)

    emccd_scene = simulate_detector(scene_with_psf, emccd_detect_arguments)

    #TODO: Look into when to split into multiple scenes.

    return emccd_scene


def open_CPGS_dict(cpgs_xml):
    '''

    Function that opens a CGPS file, returns a dictionary with relevant info

    '''
    cpgs_dict = {}

    return cpgs_dict


def convolve_2d_scene(scene_info, proper_arguments, cpgs_info):
    '''
    
    Function that convolves a background scene with off-axis PSFs 

    scene_info: a dictionary entry "scene_hdu" that contains an HDU 
    with a 2D image and a header that contains the pixel scale and center

    proper_arguments and cpgs_info collectively contain rest info we need to generate off-axis
    PSFs (TBD)

    '''
    convolved_hdu = None
    return convolved_hdu


def place_point_sources(background_scene, scene_info, proper_arguments, cpgs_info):
    '''
    
    Function that inserts individual off-axis PSFs 

    Arguments:
    background_scene: a HDU with a 2D image and a header that contains the pixel scale and center
    scene_info: a dictionary with an entry called "point_source_info" that contains a list of point-source 
    brightnesses and locations
    
    proper_arguments and cpgs_info collectively contain rest info we need to generate off-axis
    PSFs (TBD) 

    '''
    point_source_hdu = None
    return point_source_hdu


def generate_host_star_psf(scene_with_sources, scene_info, proper_arguments):
    '''

    Function that generates a PSF of the host star 

    Arguments: (Max, edit these descriptions as necessary)
    scene_with_sources: a HDU with a 2D image and a header that contains the placed point sources 
    scene_info: a dictionary with an entry called "host_star_info" that contains a list of relevant
    info regarding the host star such as brighteness and location
    
    proper_arguments contains rest of info needed to generate off-axis PSFs (TBD)
    '''
    host_star_psf_hdu = None
    return host_star_psf_hdu


def simulate_detector(scene_with_psf, emccd_detect_arguments):
    '''
    
    Function that simulates the detector given EMCCD detect arguments and 2D image with PSF

    Arguments: (Max, edit these descriptions as necessary)
    scene_with_psf: a HDU with a 2D image and a header that contains PSF
    
    emccd_detect_arguments contains rest info we need to simulate the detector (TBD)
    '''
    simulated_detector_hdu = None
    return simulated_detector_hdu