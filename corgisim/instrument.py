
import proper
import numpy as np
from astropy.io import fits
import roman_preflight_proper
from corgisim import scene
import pkg_resources
import cgisim
from synphot.models import BlackBodyNorm1D, Box1D,Empirical1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
import matplotlib.pyplot as plt

class CorgiOptics():
    '''
    A class that defines the current configuration of the CGI optics, including the telescope
    and the coronagraph.

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    It will need to know the telescope roll angle. 

    '''

    def __init__(self, cgi_mode = None, bandpass= None,  diam = 236.3114, proper_keywords=None, **kwargs):
        '''

        Initialize the class a keyword dictionary that defines the setup of cgisim/PROPER 
        and other relevant information (such as the telescope roll angle).


        Initialize the class with two dictionaries: 
        - cgi_mode (str): define cgi simulation mode, valid values: 'excam', 'spec', ‘lowfs’, ‘excam_efield’
        - bandpass (str): pre-difined bandpass for Roman-CGI
        - diam (float) in meter: diameter of the primaru mirror, the default value is 2.363114 meter
        - proper_keywords: A dictionary with the keywords that are used to set up the proper model

        Raises:
        - ValueError: If `cgi_mode` or `cor_type` is invalid.
        - KeyError: If required `proper_keywords` are missing.
        - KeyError: If forbidden keywords are included.
        """
        '''
         # Initialize proper_keywords safely
        if proper_keywords is None:
            proper_keywords = {}

        #some parameters to the PROPER prescription are required, including 'cor_type', 'polaxis'
        required_keys = {'cor_type', 'polaxis', 'output_dim'}
        missing_keys = required_keys - proper_keywords.keys()
        if missing_keys:
            raise KeyError(f"ERROR: Missing required proper_keywords: {missing_keys}")

        # some parameters to the PROPER prescription are not allowed when calling it from corgisim;
        ## 'final_sampling_m' is directly choosed based on different cgi mode 
        ## 'end_at_fpm_exit_pupil','end_at_fsm' are not allowed because they will give outimage at fsm 
        forbidden_keys = {'final_sampling_lam0', 'final_sampling_m', 'end_at_fpm_exit_pupil','end_at_fsm'}
        forbidden_found = forbidden_keys & proper_keywords.keys()
        if forbidden_found:
            raise KeyError(f"ERROR: Forbidden keywords detected in proper_keywords: {forbidden_found}")


        valid_cgi_modes = ['excam', 'spec', 'lowfs', 'excam_efield']
        valid_cor_types = ['hlc', 'hlc_band1', 'spc-spec', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'spc-wide_band4', 
                        'spc-wide_band1', 'spc-mswc', 'spc-mswc_band4','spc-mswc_band1', 'zwfs',
                        'hlc_band2', 'hlc_band3', 'hlc_band4', 'spc-spec_rotated', 'spc-spec_band2_rotated', 'spc-spec_band3_rotated']

        if cgi_mode not in valid_cgi_modes:
            raise Exception('ERROR: Requested mode does not match any available mode')
     

        if proper_keywords['cor_type'] not in valid_cor_types:
            raise Exception('ERROR: Requested coronagraph does not match any available types')

        self.cgi_mode = cgi_mode
        self.bandpass = bandpass 
        # get mode and bandpass parameters:
        info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
        mode_data, bandpass_data = cgisim.cgisim_read_mode( cgi_mode, proper_keywords['cor_type'], bandpass, info_dir )

        self.lam0_um = bandpass_data["lam0_um"] ##central wavelength of the filter in micron
        self.nlam = bandpass_data["nlam"] 
        self.lam_um = np.linspace( bandpass_data["minlam_um"], bandpass_data["maxlam_um"], self.nlam ) ### wavelength in um
        self.sampling_lamref_div_D = mode_data['sampling_lamref_div_D'] 
        self.lamref_um = mode_data['lamref_um'] ## ref wavelength in micron
        self.owa_lamref = mode_data['owa_lamref'] ## out working angle
        self.sampling_um = mode_data['sampling_um'] ### size of pixel in micron

        self.diam = diam  ## diameter of Roman primary in cm, default is 236.114 cm
        # Effective collecting area in unit of cm^2, 
        # 30.3% central obscuration of the telescope entrance pupil (diameter ratio) from IPAC-Roman website
        self.area = (self.diam/2)**2 * np.pi - (self.diam/2*0.303)**2 * np.pi 
        self.grid_dim_out = proper_keywords['output_dim'] # number of grid in output image in one dimension
        self.proper_keywords = proper_keywords  # Store the keywords for PROPER package
        self.proper_keywords['lam0']=self.lam0_um

        # polarization
        
        if proper_keywords['polaxis'] != 10 and proper_keywords['polaxis'] != -10 and proper_keywords['polaxis'] != 0:
            self.polarizer_transmission = 0.45
        else:
            self.polarizer_transmission = 1.0

        self.integrate_pixels = True ## whether to subsamping the pixels and integrate them
        if "integrate_pixels" in kwargs: self.integrate_pixels = kwargs.get("integrate_pixels")
        self.nd = 0  # integer: 1, 3, or 4 (0 = no ND, the default); this is the ND filter identifier, NOT the amount of ND
        if "nd" in kwargs: self.nd = kwargs.get("nd")

        if 'if_quiet'in kwargs:self.quiet = kwargs.get("if_quiet")



        print("CorgiOptics initialized with proper keywords.")
     

    def get_psf(self, input_scene, on_the_fly=False, oversampling_factor = 7, return_oversample = False):
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
        
        if self.cgi_mode == 'excam':
            # Initialize the bandpass class (from synphot)
            # bp: wavelegth is in unit of angstrom
            # bp: throughput is unitless, including transmission, reflectivity and EMCCD quantum efficiency 
            bp = self.setup_bandpass(self.cgi_mode, self.bandpass, self.nd)
            # Compute the observed stellar spectrum within the defined bandpass
            # obs: wavelegth is in unit of angstrom
            # obs: flux is in unit of photons/s/cm^2/angstrom
            obs = Observation(input_scene.stellar_spectrum, bp)
            

            if self.integrate_pixels:
                oversampling_factor = 7
            else:
                oversampling_factor = 1

            grid_dim_out_tem = self.grid_dim_out * oversampling_factor
            sampling_um_tem = self.sampling_um / oversampling_factor

            self.proper_keywords['output_dim']=grid_dim_out_tem
            self.proper_keywords['final_sampling_m']=sampling_um_tem *1e-6
            
            
            (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=self.proper_keywords,QUIET=self.quiet)
            images_tem = np.abs(fields)**2

            # Initialize the image array based on whether oversampling is returned
            images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
            images = np.zeros(images_shape, dtype=float)

            for i in range(images_tem.shape[0]):
                if return_oversample:
                    ##return the oversampled PSF, default 7 grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:]
                else:
                    ## integrate oversampled PSF back to one grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:].reshape((self.grid_dim_out,oversampling_factor,self.grid_dim_out,oversampling_factor)).mean(3).mean(1) * oversampling_factor**2

                dlam_um = self.lam_um[1]-self.lam_um[0]
                lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                # ares in unit of cm^2
                # counts in unit of photos/s
                counts = self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                images[i,:,:] = images[i,:,:] * counts

        image = np.sum(images, axis=0)

        if self.cgi_mode in ['spec', 'lowfs', 'excam_efield']:
            raise ValueError(f"The mode '{self.cgi_mode}' has not been implemented yet!")
        
        # Initialize SimulatedScene class to restore the output psf
        sim_scene = scene.SimulatedScene(input_scene)
        
        # Prepare header information for the output HDU FITS file
        header_info = {'wvl_c_um':self.lam0_um,
                    's_sptype':input_scene.host_star_sptype,
                    's_Vmag':input_scene.host_star_Vmag,
                    's_magtype':input_scene.host_star_magtype,
                    }
        # Define specific keys from self.proper_keywords to include in the header            
        keys_to_include_in_header = ['cor_type', 'use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                            'use_lyot_stop','use_field_stop','output_dim']  # Specify keys to include
        subset = {key: self.proper_keywords[key] for key in keys_to_include_in_header if key in self.proper_keywords}
        header_info.update(subset)
        # Create the HDU object with the generated header information
        sim_scene.host_star_image = create_hdu(image,header_info =header_info)

        return sim_scene

    def setup_bandpass(self, cgimode, bandpass_name, nd ):
        """
        Sets up the bandpass for the simulation.

        This function initializes a bandpass based on the given parameters, computes 
        wavelength boundaries, and retrieves throughput data.

        Args:
            cgimode (str): The CGI mode to be used.
            bandpass_name (str): Name of the bandpass filter.
            nd (int): Neutral density filter index.

        Returns:
            bp (SpectralElement): A spectral element object containing the bandpass data.

        """
        info_dir = cgisim.lib_dir + '/cgisim_info_dir/'


        dlam_um = self.lam_um[1] - self.lam_um[0]
       
        lam_start_um = np.min(self.lam_um)-  1* dlam_um
        lam_end_um = np.max(self.lam_um)+  1* dlam_um
        bandpass_i = 'lam' + str(lam_start_um*1000) + 'lam' + str(lam_end_um*1000) 

        wave, throughput = cgisim.cgisim_roman_throughput( bandpass_name, bandpass_i, nd, cgimode, info_dir )

        bp = SpectralElement(Empirical1D, points=wave, lookup_table=throughput)

        return bp


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


def create_hdu(data, header_info=None):
        """
        Create an Astropy HDU for the PSF with metadata.

        Parameters:
        - data (numpy.ndarray): 2D array representing the PSF.
        - header_info (dict): Dictionary of metadata to include in the header.

        Returns:
        - hdu (fits.PrimaryHDU): Astropy HDU object containing the data and header_info
        """
        # Create the Primary HDU with the data
        ##primary_hdu = fits.PrimaryHDU()
        # Create an Image HDU with data
        ##image_hdu = fits.ImageHDU(data)
        # Combine them into an HDUList
        ##hdul = fits.HDUList([primary_hdu, image_hdu])

        ####read default header and pass into the hdu
        #ile_path = pkg_resources.resource_filename("corgisim.data", "data/CGI_0000000000000000014_20221004T2359351_L1_.fits")
        #with fits.open(file_path) as hdul_default:
        #    primary_header = hdul_default[0].header
        #    image_header = hdul_default[1].header

        #hdul[0].header = primary_header  # Primary HDU header
        #hdul[1].header = image_header    # Image HDU header
        hdul = fits.PrimaryHDU(data)

        if header_info is not None:
        # Add customerized header info to the header
            #print(header_info)
            hdul.header['COMMENT'] = "This FITS file contains simulated data."
            hdul.header['COMMENT'] = "Header includes stellar properties and other simulation details."

            for key, value in header_info.items():
                comment = key+' : '+str(value)
                hdul.header.add_comment(comment)
                #hdul.header[key] = value
            
        return hdul
            

