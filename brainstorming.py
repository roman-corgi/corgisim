
### Scene - Make it a structured class. 
### Observation class - generated from CPGS. Contains proper_arguments, emccd_detect_arguments. A Scene can be an attribute of the Observation class. 
### Dataset Object


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

    cpgs_info = open_CPGS_dict(CPGS_xml)

    ## TODO: ADD headers? 
    ## TODO: Make these three steps independent of each other and pass them to simulate_detector (default each one to None)
    ## Each of convolve_2d_scene, place_point_sources, generate_host_star_psf should return a dataset with 2D images in the detector plane
    ## e.g. polarimetry and spectroscopy modes will generate 2D images. 
    ## 


    #We convolve a 2D input scene with off-axis PSFS
    convolved_scene_list = convolve_2d_scene(scene_info, proper_arguments, cpgs_info)

    #We inject point-source companions. We either use this function for spectral mode
    #or we have a separate function for spectral mode enabled by a switch here. 
    scenes_with_sources = place_point_sources(convolved_scene_list, scene_info, proper_arguments, cpgs_info)

    #We generate a PSF for the host star - at this point we will split them off into separate exposures
    scene_list = generate_host_star_psf(scenes_with_sources, scene_info, proper_arguments)

    #Up to now the scene should be in physics units (such as mJy/s or photons/s), now we convert them into detector counts
    emccd_scene = simulate_detector(scene_list, emccd_detect_arguments)

    return emccd_scene

def setup_default_instrument(CPGS_xml): 
    '''

    Given the CPGS setup, generate a default set of proper arguments and emccd_detect arguments

    '''
    proper_dict = {}
    emccd_dict = {}

    return proper_dict, emccd_dict



def open_CPGS_dict(cpgs_xml):
    '''

    Function that opens a CGPS xml file, returns a dictionary with relevant info

    '''
    cpgs_dict = {}

    return cpgs_dict


def convolve_2d_scene(scene_info, proper_arguments, cpgs_info):
    '''
    
    Function that convolves a background scene with spatially varying off-axis PSFs. 
    It returns a list of scenes corresponding to each frame. Might include multiple roll angles. 
    
    MMB Notes: 
    - This function will likely be the most time-consuming if we run it every time. We may want to 
    have the default option be to generate a set of PSFs ahead for each mode 
    of time and store them for easy use. 
    - This should be able to handle polarized inputs. 

    Arguments: 
    scene_info: Includes a dictionary entry "scene_hdu" that contains a (possibly oversampled) HDU
    with a 2D image and a header that contains the pixel scale and center. 

    proper_arguments and cpgs_info collectively contain rest info we need to generate off-axis
    PSFs (TBD)

    Returns:  
    A list of HDUs with the convolved scenes

    '''
    convolved_hdulist = None
    return convolved_hdulist


def place_point_sources(background_scene_list, scene_info, proper_arguments, cpgs_info):
    '''
    
    Function that inserts individual off-axis PSFs to each input scene. The scenes will be different exposures 
    in an L1 dataset, and may include multiple rolls. 

    Arguments:
    background_scene_list: a list of HDUs with a 2D image and a header that contains the pixel scale and center
    scene_info: a dictionary with an entry called "point_source_info" that contains a list of point-source 
    brightnesses and locations
    
    proper_arguments and cpgs_info collectively contain rest info we need to generate off-axis
    PSFs (TBD) 

    '''
    point_source_hdu = None
    return point_source_hdu


def generate_host_star_psf(scene_with_sources, scene_info, proper_arguments):
    '''

    Function that generates a PSF of the host star and places it into each input scece. The input
    scenes will be different exposures in an L1 dataset, and may include multiple rolls.
    
    The function should be able to generate PSFs on the fly, 
    or use a set of pre-generated PSFs, such as the available Observing Scenarios (OS9, OS11, etc.). 

    Arguments: (Max, edit these descriptions as necessary)
    scene_with_sources: a HDU with a 2D image and a header that contains the placed point sources 
    scene_info: a dictionary with an entry called "host_star_info" that contains a list of relevant
    info regarding the host star such as brighteness and location
    
    proper_arguments contains rest of info needed to generate off-axis PSFs (TBD)
    '''
    host_star_psf_hdu = None
    return host_star_psf_hdu


def simulate_detector(scenes_with_psf, emccd_detect_arguments):
    '''
    
    Function that simulates the detector images for each scene
    given EMCCD detect arguments and 2D image with PSF

    Arguments: (Max, edit these descriptions as necessary)
    scenes_with_psf: a HDU with a 2D image and a header that contains PSF
    
    emccd_detect_arguments contains rest info we need to simulate the detector (TBD)
    '''
    simulated_detector_hdu = None
    return simulated_detector_hdu