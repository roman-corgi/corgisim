
import proper
import numpy as np
from astropy.io import fits
import roman_preflight_proper
from corgisim import scene

class CorgiOptics():
    '''
    A class that defines the current configuration of the CGI optics, including the telescope
    and the coronagraph.

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    It will need to know the telescope roll angle. 

    '''

    def __init__(self, lambda_m, proper_keywords, diam = 2.363114):
        '''

        Initialize the class a keyword dictionary that defines the setup of cgisim/PROPER 
        and other relevant information (such as the telescope roll angle).


        Initialize the class with two dictionaries: 
        - lambda_m (float or array): the wavelength for the simulation
        - proper_keywords: A dictionary with the keywords that are used to set up the proper model
        '''
        self.lambda_m = lambda_m ## wavelength in nm
        self.diam = diam  ## diameter of Roman primary meter, default is 2.36114 meter
        self.area = (self.diam/2)**2 * np.pi ### collecting area in unit of m^2
        self.proper_keywords = proper_keywords  # Store the keywords for later use

        print("CorgiOptics initialized with proper keywords.")

    def get_psf(self, input_scene, on_the_fly=False, oversample = 1, return_oversample = False):
        '''
        
        Function that provides an on-axis PSF for the current configuration of CGI.

        It should take the host star properties from scene.host_star_properties and return a 
        Simulated_scene object with the host_star_image attribute populated with an astropy HDU 
        that contains a noiseless on-axis PSF, and associated metadata in the header. This on-axis 
        PSF should be either generated on the fly, or picked from a pregenerated library (e.g. OS11 
        or something cached locally). 

        #Todo Figure out the default output units. Current candidate is photoelectrons/s. 
        #Todo: If the input is a scene.Simulation_Scene instead, then just pull the Scene from the attribute 
                and put the output of this function into the host_star_image attribute.

        Arguments: 
        input_scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSF should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSF
        return_oversample: A boolean that defines whether the function should return the oversampled PSF or not.

        Returns:
        corgisim.scene.Simulated_Scene: A scene object with the host_star_image attribute populated with an astropy
                                        HDU that contains a noiseless on-axis PSF.

        '''
        ### calculate the star flux at the given wavelenghts in unit of W/m^2/nm
        host_star_flux = fstar(lam=self.lambda_m, teff=input_scene.host_star_Teff, rs=input_scene.host_star_Rs, d=input_scene.host_star_Dist)
       
        ### convert star flux W/m^2/nm to photons/m^2/nm
        host_star_photons = power2photon(self.lambda_m, host_star_flux)
        
        ### calculate the photons collected by the telescope in unit of photons/s/nm
        collected_photons = host_star_photons * self.area 
        
        nlam = len(self.lambda_m)
        if  nlam ==1:
            ####simulate monochramtic image, output image is in the unit of photons/s/nm
            (fields, sampling) = proper.prop_run('roman_preflight', self.lambda_m[0]/1e3,self.proper_keywords['npsf'], QUIET=self.proper_keywords['if_quiet'],PRINT_INTENSITY=self.proper_keywords['if_print_intensity'],PASSVALUE=self.proper_keywords)
            image = np.abs(fields)**2  * collected_photons 

        if  nlam > 1:
            ####simulate broadband image, output image is in the unit of photons/s 
            dl = np.append(np.diff(self.lambda_m),np.diff(self.lambda_m)[0])
            collected_photons_l =  collected_photons * dl ###photons/s in each wavelength interval
            
            (fields, sampling) = proper.prop_run_multi('roman_preflight', self.lambda_m/1e3, self.proper_keywords['npsf'], QUIET=self.proper_keywords['if_quiet'],PRINT_INTENSITY=self.proper_keywords['if_print_intensity'],PASSVALUE=self.proper_keywords)
            images = np.abs(fields)**2
            image = np.sum(images *  collected_photons_l[:, np.newaxis, np.newaxis],0)

        sim_scene = scene.SimulatedScene(input_scene)
        
        sim_scene.host_star_image =  image

        return sim_scene


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
        TODO: If the input is a scene.Simulation_Scene instead, then just pull the Scene from the attribute
                and put the output of this function into the twoD_image attribute

        Arguments: 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSFs
        return_oversample: A boolean that defines whether the function should return the oversampled PSFs or not.

        Returns: 
        corgisim.scene.Simulated_Scene: A scene object with the background_scene attribute populated with an astropy
                                        HDU that contains the simulated scene.
        '''
        pass


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
        TODO: If the input is a scene.Simulation_Scene instead, then just pull the Scene from the attribute
                and put the output of this function into the point_source_image attribute

        Arguments: 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.
        on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        oversample: An integer that defines the oversampling factor of the detector when generating the PSFs
        return_oversample: A boolean that defines whether the function should return the oversampled PSFs or not.

        Returns: 
        A 2D numpy array that contains the scene with the injected point sources. 
        '''
        pass
    
class CorgiDetector(): 
    
    def __init__(self, emccd_keywords):
        '''
        Initialize the class with a dictionary that defines the EMCCD_DETECT input parameters. 

        Arguments: 
        emccd_keywords: A dictionary with the keywords that are used to set up the emccd model
        '''
        pass
    

    def generate_detector_image(self, simulated_scene):
        '''
        Function that generates a detector image from the input image, using emccd_detect. 

        The input_image probably has to be in electrons. 

        Arguments:
        total_scene: a corgisim.scene.Scene object that contains the scene to be simulated in the total_scene attribute.
        
        Returns:
        A corgisim.scene.Scene object that contains the detector image in the 
        '''
        pass

    def place_scene_on_detector(self, scene):
        '''
        Function that places the simulated scene on the detector. 

        It should take the input scene from scene.total_scene and return an updated scene object with the
        detector_image attribute populated with an astropy HDU that contains the simulated scene on the detector and associated metadata
        in the header.

        The detector_image should be generated by using the emccd_detect function.

        Likely an intermediate step that typical users won't touch. 
        '''
        pass



def fstar(lam=None, teff=None, rs=None, d=None):
    '''
    Function that calculates the stellar blackbody flux in units of W/m²/nm.

    Arguments: 
        lam (float or array): wavelength (nm)
        teff (float): host star effective temperature (K)
        rs (float): stellar radius (solar radii)
        d (float): distance to star (pc)
       

    Returns:
        flux (float or array): stellar blackbody flux at the given wavelengths (W/m²/nm).
    '''

    rsun = 6.958e8       # solar radius (m)
    ds = 3.08567e16      # parsec (m)
    
    liambda = 1.e-9*lam  # wavelength (m)
    c1 = 3.7417715e-16   # 2*pi*h*c*c (kg m**4 / s**3)
    c2 = 1.4387769e-2    # h*c/k (m K)
    pow =c2/liambda/teff

    Fs = c1/( (liambda**5.)*(np.exp(pow)-1.0) )*1e-9

    return Fs*(rs*rsun/d/ds)**2

def power2photon(wvl, power):
    '''
    Function that coverts spetral power from W/M^2/nm to phonto/M^2/nm

    Arguments: 
        wvl (float or array): wavelength (nm)
        power (float): stellar blackbody flux in W/m²/nm

    Returns:
        photocounts (float or array): in photo/s/m²/nm.
    '''
    
    h = 6.626e-34  ##unit: J*s
    c = 3e8 * 1e9  ## unit: nm/s
    fre = c / wvl
    photonum=power / h / fre

    return photonum
