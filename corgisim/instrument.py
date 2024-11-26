


class CGI_optics():
    '''
    A class that defines the current configuration of the CGI optics, including the telescope
    and the coronagraph.

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    It will need to know the telescope roll angle. 

    '''

    def __init__(self, proper_keywords):
        '''

        Initialize the class a keyword dictionary that defines the setup of cgisim/PROPER 
        and other relevant information (such as the telescope roll angle).


        Initialize the class with two dictionaries: 
        - proper_keywords: A dictionary with the keywords that are used to set up the proper model
        '''

    def get_psf(self, scene, on_the_fly=False, oversample = 1, return_oversample = False):
        '''
        
        Function that provides an on-axis PSF for the current configuration of CGI.

        It should take the host star properties from scene.host_star_properties and return a 
        Simulated_scene object with the host_star_image attribute populated with an astropy HDU 
        that contains a noiseless on-axis PSF, and associated metadata in the header. This on-axis 
        PSF should be either generated on the fly, or picked from a pregenerated library (e.g. OS11 
        or something cached locally). 

        TODO: Figure out the default output units. Current candidate is photoelectrons/s. 

        Arguments: 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSF should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSF
        return_oversample: A boolean that defines whether the function should return the oversampled PSF or not.

        Returns:
        corgisim.scene.Simulated_Scene: A scene object with the host_star_image attribute populated with an astropy
                                        HDU that contains a noiseless on-axis PSF.

        '''


    def simulate_2D_scene(self, scene, on_the_fly=False, oversample = 1, return_oversample = False):
        '''
        Function that simulates a 2D scene with the current configuration of CGI. 

        It should take the image data from the HDU from scene.background_scene and convolve it with a 
        set of off-axis PSFs (also known as PRFs in some circles), and return an updated scene object with the
        background_scene attribute populated with an astropy HDU that contains the simulated scene and associated 
        metadata in the header.

        The off-axis PSFs should be either generated on the fly, or read in from a set of pre-generated PSFs. The 
        convolution should be flux conserving. 

        TODO: Figure out the default output units. Current candidate is photoelectrons/s.

        Arguments: 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSFs
        return_oversample: A boolean that defines whether the function should return the oversampled PSFs or not.

        Returns: 
        corgisim.scene.Simulated_Scene: A scene object with the background_scene attribute populated with an astropy
                                        HDU that contains the simulated scene.
        '''


    def inject_point_sources(self, scene, on_the_fly=False, oversample = 1, return_oversample = False):
        '''
        Function that injects point sources into the scene. 

        It should take the input scene and inject the point sources defined scene.point_source_list, 
        which should give the location and brightness. 

        The off-axis PSFs should be either generated on the fly, or read in from a set of pre-generated PSFs. 

        TODO: Figure out the default output units. Current candidate is photoelectrons/s.
        TODO: We may want this to generate "far away" point sources whose diffraction spikes still 
                end up in our scene. We'll only want to simulate the part of the scene that ends up on the detector.
                How do we do that with corgisim/proper?

        Arguments: 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSFs
        return_oversample: A boolean that defines whether the function should return the oversampled PSFs or not.

        Returns: 
        A 2D numpy array that contains the scene with the injected point sources. 
        '''
    
class CGI_detector(): 
    
    def __init__(self, emccd_keywords):
        '''
        Initialize the class with a dictionary that defines the EMCCD_DETECT input parameters. 

        Arguments: 
        emccd_keywords: A dictionary with the keywords that are used to set up the emccd model
        '''
    
    
    def place_scene_on_detector(self, scene):
        '''
        Function that places the simulated scene on the detector. 

        It should take the input scene from scene.total_scene and return an updated scene object with the
        detector_image attribute populated with an astropy HDU that contains the simulated scene on the detector and associated metadata
        in the header.

        The detector_image should be generated by using the emccd_detect function.
        '''


    def generate_detector_image(self, ):
        '''
        Function that generates a detector image from the input image, using emccd_detect. 

        The input_image probably has to be in electrons. 

        Arguments:
        total_scene: a corgisim.scene.Scene object that contains the scene to be simulated in the total_scene attribute.
        
        Returns:
        A corgisim.scene.Scene object that contains the detector image in the 
        '''
