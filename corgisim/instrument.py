
import proper
import warnings
import numpy as np
from astropy.io import fits
import astropy.units as u
import roman_preflight_proper
from corgisim import scene
import cgisim
import corgisim
from synphot.models import BlackBodyNorm1D, Box1D,Empirical1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
import matplotlib.pyplot as plt
from emccd_detect.emccd_detect import EMCCDDetectBase, EMCCDDetect
from corgidrp import mocks
from corgisim import outputs
from corgisim import spec
import os
from scipy import interpolate

class CorgiOptics():
    '''
    A class that defines the current configuration of the CGI optics, including the telescope
    and the coronagraph.

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    It will need to know the telescope roll angle. 

    '''

    def __init__(self, cgi_mode = None, bandpass= None,  diam = 236.3114, proper_keywords=None, oversampling_factor = 7, return_oversample = False, **kwargs):
        '''

        Initialize the class a keyword dictionary that defines the setup of cgisim/PROPER 
        and other relevant information (such as the telescope roll angle).


        Initialize the class with two dictionaries: 
        - cgi_mode (str): define cgi simulation mode, valid values: 'excam', 'spec', ‘lowfs’, ‘excam_efield’
        - cor_type (str): define coronagraphic observing modes
        - bandpass (str): pre-difined bandpass for Roman-CGI
        - diam (float) in meter: diameter of the primaru mirror, the default value is 2.363114 meter
        - proper_keywords: A dictionary with the keywords that are used to set up the proper model
        - oversample: An integer that defines the oversampling factor of the detector when generating the image
        - return_oversample: A boolean that defines whether the function should return the oversampled image or not.
    

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
        valid_cor_types = ['hlc', 'hlc_band1', 'spc-wide', 'spc-wide_band4', 
                        'spc-wide_band1', 'hlc_band2', 'hlc_band3', 'hlc_band4', 'spc-spec_rotated', 'spc-spec_band2_rotated', 'spc-spec_band3_rotated']
        
        #these cor_type is availbale in cgisim, but are currently untested in corgisim
        untest_cor_types = ['spc-spec', 'spc-spec_band2', 'spc-spec_band3', 'spc-mswc', 'spc-mswc_band4','spc-mswc_band1', 'zwfs']


        if cgi_mode not in valid_cgi_modes:
            raise Exception('ERROR: Requested mode does not match any available mode')
     

        if proper_keywords['cor_type'] not in valid_cor_types and proper_keywords['cor_type'] not in untest_cor_types:
            raise Exception('ERROR: Requested coronagraph does not match any available types')
        
        if proper_keywords['cor_type'] in untest_cor_types:
            warnings.warn('Warning: Requested coronagraph is currently untested and might not work as expected')

        self.cgi_mode = cgi_mode
        self.cor_type = proper_keywords['cor_type']
        if bandpass  in ['1F','2F','3F','4F']:
            self.bandpass = bandpass.split('F')[0]
        else:
            self.bandpass = bandpass.lower()
        self.bandpass_header = bandpass
        # self.bandpass is used as the keyword for cgisim, while self.bandpass_header is used for setting the FITS header.
        # The distinction arises from differences in naming conventions for filters between cgisim and the latest wiki page.



        #self.bandpass = bandpass 

        # get mode and bandpass parameters:
        info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
        mode_data, bandpass_data = cgisim.cgisim_read_mode( cgi_mode, proper_keywords['cor_type'], self.bandpass, info_dir )

        # Set directory containing reference data for parameters external to CGISim
        ref_data_dir = corgisim.lib_dir + '/data/'
        if not os.path.exists(ref_data_dir):
            raise FileNotFoundError(f"Directory does not exist: {ref_data_dir}")
        else:
            self.ref_data_dir = ref_data_dir
        # Set the spectroscopy parameters
        if self.cgi_mode == 'spec':
            spec_kw_defaults = {
                'slit': 'None', # named FSAM slit
                'slit_x_offset_mas': 0.0, # offset of slit position from star on EXCAM, in mas
                'slit_y_offset_mas': 0.0, # offset of slit position from star on EXCAM, in mas
                'prism': 'None', # named DPAM prism
                'wav_step_um': 1E-3 # wavelength step size of the prism dispersion model, in microns 
            }
            spec_kw_allowed = {
                'slit': ['None', 'R1C2', 'R6C5', 'R3C1'],
                'prism': ['None', 'PRISM3', 'PRISM2']
            }
            for attr_name, default_value in spec_kw_defaults.items():
                if attr_name in kwargs:
                    value = kwargs[attr_name]
                
                    if attr_name in spec_kw_allowed:
                        if value not in spec_kw_allowed[attr_name]:
                            allowed_str = ", ".join(f"'{v}'" for v in spec_kw_allowed[attr_name])
                            raise ValueError(
                                f"Invalid value for '{attr_name}': '{value}'. "
                                f"Must be one of: {allowed_str}"
                            )
                    setattr(self, attr_name, value)
                else:
                    setattr(self, attr_name, default_value)
            if self.prism != 'None':
                prism_param_fname = os.path.join(ref_data_dir, 'TVAC_{:s}_dispersion_profile.npz'.format(self.prism))
                if not os.path.exists(prism_param_fname):
                    raise FileNotFoundError(f"Prism parameter file {prism_param_fname} does not exist")
                else:
                    setattr(self, 'prism_param_fname', prism_param_fname)
            if self.slit != 'None':
                slit_param_fname = os.path.join(ref_data_dir, 'FSAM_slit_params.json')
                if not os.path.exists(slit_param_fname):
                    raise FileNotFoundError(f"Slit aperture parameter file {slit_param_fname} does not exist")
                else:
                    setattr(self, 'slit_param_fname', slit_param_fname)

        self.lam0_um = bandpass_data["lam0_um"] ##central wavelength of the filter in micron
        self.nlam = bandpass_data["nlam"] 
        self.lam_um = np.linspace( bandpass_data["minlam_um"], bandpass_data["maxlam_um"], self.nlam ) ### wavelength in um
        self.sampling_lamref_div_D = mode_data['sampling_lamref_div_D'] 
        self.lamref_um = mode_data['lamref_um'] ## ref wavelength in micron
        self.owa_lamref = mode_data['owa_lamref'] ## out working angle
        if self.cgi_mode == 'spec':
            baseline_mode_data, _ = cgisim.cgisim_read_mode('excam', 'hlc_band1', '1', info_dir=info_dir)
            self.sampling_um = baseline_mode_data['sampling_um']
            # Redefine the wavelength array so that the prism dispersion wavelength bins span the full bandpass
            if self.prism != 'None': 
                dlam_um = self.lam_um[1] - self.lam_um[0]
                self.lam_um = np.linspace( bandpass_data["minlam_um"] - 0.5*dlam_um, bandpass_data["maxlam_um"] + 0.5*dlam_um, self.nlam ) ### wavelength in um
            else:
                self.lam_um = np.linspace( bandpass_data["minlam_um"], bandpass_data["maxlam_um"], self.nlam ) ### wavelength in um
        else:
            self.sampling_um = mode_data['sampling_um'] ### size of pixel in micron

        self.diam = diam  ## diameter of Roman primary in cm, default is 236.114 cm
        # Effective collecting area in unit of cm^2, 
        # 30.3% central obscuration of the telescope entrance pupil (diameter ratio) from IPAC-Roman website
        #self.area = (self.diam/2)**2 * np.pi - (self.diam/2*0.303)**2 * np.pi
        self.area =  35895.212    # primary effective area from cgisim cm^2 
        self.grid_dim_out = proper_keywords['output_dim'] # number of grid in output image in one dimension
        self.proper_keywords = proper_keywords  # Store the keywords for PROPER package
        self.proper_keywords['lam0']=self.lam0_um

        # polarization
        
        if proper_keywords['polaxis'] != 10 and proper_keywords['polaxis'] != -10 and proper_keywords['polaxis'] != 0:
            self.polarizer_transmission = 0.45
        else:
            self.polarizer_transmission = 1.0

        #self.integrate_pixels = True ## whether to subsamping the pixels and integrate them
        #if "integrate_pixels" in kwargs: self.integrate_pixels = kwargs.get("integrate_pixels")
        
        ## setup if to oversampling the image and if return the oversample
        self.oversampling_factor = oversampling_factor
        self.return_oversample = return_oversample


        self.nd = 0  # integer: 1, 3, or 4 (0 = no ND, the default); this is the ND filter identifier, NOT the amount of ND
        if "nd" in kwargs: self.nd = kwargs.get("nd")

        # Initialize the bandpass class (from synphot)
        # bp: wavelegth is in unit of angstrom
        # bp: throughput is unitless, including transmission, reflectivity and EMCCD quantum efficiency 
        self.bp = self.setup_bandpass(self.cgi_mode, self.bandpass, self.nd)



        if 'if_quiet'in kwargs:self.quiet = kwargs.get("if_quiet")



        print("CorgiOptics initialized with proper keywords.")
     

    def get_host_star_psf(self, input_scene, sim_scene=None, on_the_fly=False):
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
        
        Returns:
        corgisim.scene.Simulated_Scene: A scene object with the host_star_image attribute populated with an astropy
                                        HDU that contains a noiseless on-axis PSF.

        '''
        
        if self.cgi_mode == 'excam':
            
            # Compute the observed stellar spectrum within the defined bandpass
            # obs: wavelegth is in unit of angstrom
            # obs: flux is in unit of photons/s/cm^2/angstrom
            obs = Observation(input_scene.stellar_spectrum, self.bp)
            
            #if self.integrate_pixels:
                #oversampling_factor = 7
            #else:
                #oversampling_factor = 1

            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor

            self.proper_keywords['output_dim']=grid_dim_out_tem
            self.proper_keywords['final_sampling_m']=sampling_um_tem *1e-6
            
            
            (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=self.proper_keywords,QUIET=self.quiet)
            images_tem = np.abs(fields)**2

            # Initialize the image array based on whether oversampling is returned
            images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
            images = np.zeros(images_shape, dtype=float)

            for i in range(images_tem.shape[0]):
                if self.return_oversample:
                    ##return the oversampled PSF, default 7 grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:]
                else:
                    ## integrate oversampled PSF back to one grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2
                    ## update the proper_keywords['output_dim'] baclk to non_oversample size
                    self.proper_keywords['output_dim'] = self.grid_dim_out

                dlam_um = self.lam_um[1]-self.lam_um[0]
                lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                # ares in unit of cm^2
                # counts in unit of photos/s
                counts = self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                images[i,:,:] = images[i,:,:] * counts

            image = np.sum(images, axis=0)

        elif self.cgi_mode == 'spec':
            if self.slit != 'None':
                field_stop_array, field_stop_sampling_m = spec.get_slit_mask(self)
                self.proper_keywords['field_stop_array']=field_stop_array
                self.proper_keywords['field_stop_array_sampling_m']=field_stop_sampling_m
            else:
                self.proper_keywords['field_stop_array']=0
                self.proper_keywords['field_stop_array_sampling_m']=0
                
            obs = Observation(input_scene.stellar_spectrum, self.bp)

            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor

            self.proper_keywords['output_dim']=grid_dim_out_tem
            self.proper_keywords['final_sampling_m']=sampling_um_tem *1e-6
            
            (fields, sampling) = proper.prop_run_multi('roman_preflight', self.lam_um, 1024, PASSVALUE=self.proper_keywords, QUIET=self.quiet)
            images_tem = np.abs(fields)**2

            # If a prism was selected, apply the dispersion model and overwrite the image cube and wavelength array.
            if self.prism != 'None': 
                images_tem, dispersed_lam_um = spec.apply_prism(self, images_tem)

                self.nlam = len(dispersed_lam_um)
                self.lam_um = dispersed_lam_um
                dlam_um = dispersed_lam_um[1] - dispersed_lam_um[0]

            # Initialize the image array based on whether oversampling is returned
            images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
            images = np.zeros(images_shape, dtype=float)
            counts = np.zeros(self.nlam) * u.count / u.second

            for i in range(images_tem.shape[0]):
                if self.return_oversample:
                    ##return the oversampled PSF, default 7 grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:]
                else:
                    ## integrate oversampled PSF back to one grid per pixel
                    images[i,:,:] +=  images_tem[i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2

                dlam_um = self.lam_um[1]-self.lam_um[0]
                lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                # ares in unit of cm^2
                # counts in unit of photos/s
                counts[i] = self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

            images *= counts.value[:, np.newaxis, np.newaxis]
            image = np.sum(images, axis=0)

        if self.cgi_mode in ['lowfs', 'excam_efield']:
            raise ValueError(f"The mode '{self.cgi_mode}' has not been implemented yet!")
        
        # Initialize SimulatedImage class to restore the output psf
        if sim_scene == None:
            sim_scene = scene.SimulatedImage(input_scene)
        
        
        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        sim_info = {'host_star_sptype':input_scene.host_star_sptype,
                    'host_star_Vmag':input_scene.host_star_Vmag,
                    'host_star_magtype':input_scene.host_star_magtype,
                    'ref_flag':input_scene.ref_flag,
                    'cgi_mode':self.cgi_mode,
                    'cor_type': self.proper_keywords['cor_type'],
                    'bandpass':self.bandpass_header,
                    'over_sampling_factor':self.oversampling_factor,
                    'return_oversample': self.return_oversample,
                    'output_dim': self.proper_keywords['output_dim'],
                    'nd_filter':self.nd}

        # Define specific keys from self.proper_keywords to include in the header            
        keys_to_include_in_header = ['use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                            'use_lyot_stop','use_field_stop','fsm_x_offset_mas','fsm_y_offset_mas']  # Specify keys to include
        subset = {key: self.proper_keywords[key] for key in keys_to_include_in_header if key in self.proper_keywords}
        sim_info.update(subset)
        sim_info['includ_dectector_noise'] = 'False'
        # Create the HDU object with the generated header information

        sim_scene.host_star_image = outputs.create_hdu(image,sim_info =sim_info)

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
        if cgimode == 'spec': # upsample the wavelength and throughput arrays
            f = interpolate.interp1d(wave, throughput, kind='linear')
            wave = np.linspace(wave[0], wave[-1], 100*len(wave))
            throughput = f(wave)
        bp = SpectralElement(Empirical1D, points=wave, lookup_table=throughput)

        return bp


    def convolve_2D_scene(self, scene, on_the_fly=False):
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
        
        Returns: 
        corgisim.scene.Simulated_Scene: A scene object with the background_scene attribute populated with an astropy
                                        HDU that contains the simulated scene.
        '''
        pass

    def inject_point_sources(self, input_scene, sim_scene=None, on_the_fly=False):
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
        sim_scene: A corgisim.SimulatedImage object to contains the simylated scene.
        on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        

        Returns: 
        A 2D numpy array that contains the scene with the injected point sources. 
        '''
        if self.cgi_mode == 'excam':


            # Extract point source spectra and positions
            point_source_spectra = input_scene.off_axis_source_spectrum
            point_source_x = input_scene.point_source_x
            point_source_y = input_scene.point_source_y

            # Ensure all inputs are lists for uniform processing
            if not isinstance(point_source_spectra, list):
                point_source_spectra = [point_source_spectra]
            if not isinstance(point_source_x, list):
                point_source_x = [point_source_x]
            if not isinstance(point_source_y, list):
                point_source_y = [point_source_y]

            # Ensure all lists have the same length
            if not (len(point_source_spectra) == len(point_source_x) == len(point_source_y)):
                raise ValueError(
                    f"Mismatch in input lengths: {len(point_source_spectra)} spectra, "
                    f"{len(point_source_x)} x-positions, {len(point_source_y)} y-positions. "
                    "Each point source must have a corresponding (x, y) position.")
            
            ##checks to see if point source is within FOV of coronagraph
            #FOV_range is indexed as follows - 0: hlc, 1: spc-spec, 2: spc-wide.
            #FOV_range Values correspond to the inner and outer radius of region of highest contrast and are in units of lambda/d
            FOV_range = [[3, 9.7], [3, 9.1], [5.9, 20.1]]
            if(self.cor_type.find('hlc') != -1):
                FOV_index = 0
            elif (self.cor_type.find('spec') != -1):
                FOV_index = 1
                raise Exception('ERROR: Spectroscopy mode not yet implemented')
                #todo: Add conditions checking if point source is within azimuthal angle range once spectroscopy mode is implemented
            else:
                FOV_index = 2
            #Calculate point source separation from origin in units of lambda/D
            point_source_radius = np.sqrt(np.power(point_source_x, 2) + np.power(point_source_y, 2)) * ((self.diam * 1e-2)/(self.lam0_um * 1e-6 * 206265000))
            for j in range(len(point_source_spectra)):
                if (not FOV_range[FOV_index][0] <= point_source_radius[j] <= FOV_range[FOV_index][1]):
                    warnings.warn(f"Point source #{j} is at separation {point_source_radius[j]} λ/D, "
                                  f"which is outside the coronagraph FOV range of "
                                  f"{FOV_range[FOV_index][0]} to {FOV_range[FOV_index][1]} λ/D for {self.cor_type}")

            # Compute the observed  spectrum for each off-axis source
            obs_point_source = [Observation(spectrum, self.bp) for spectrum in point_source_spectra]
            
            #if self.integrate_pixels:
                #oversampling_factor = 7
            #else:
                #oversampling_factor = 1

            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor
            
            point_source_image = []
            for j in range(len(point_source_spectra )):
            
                proper_keywords_comp = self.proper_keywords.copy()
                proper_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                            'final_sampling_m': sampling_um_tem * 1e-6,
                                            'source_x_offset_mas': point_source_x[j],
                                            'source_y_offset_mas': point_source_y[j]})

                (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE= proper_keywords_comp ,QUIET=True)
                images_tem = np.abs(fields)**2

                # Initialize the image array based on whether oversampling is returned
                images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
                images = np.zeros(images_shape, dtype=float)
                counts = np.zeros(self.nlam) * u.count / u.second

                for i in range(images_tem.shape[0]):
                    if self.return_oversample:
                        ##return the oversampled PSF, default 7 grid per pixel
                        images[i,:,:] +=  images_tem[i,:,:]
                    else:
                        ## integrate oversampled PSF back to one grid per pixel
                        images[i,:,:] +=  images_tem[i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2

                    dlam_um = self.lam_um[1]-self.lam_um[0]
                    lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                    lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                    # ares in unit of cm^2
                    # counts in unit of photos/s
                    counts[i] = self.polarizer_transmission * obs_point_source[j].countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                images *= counts.value[:, np.newaxis, np.newaxis]
                image = np.sum(images, axis=0)
                point_source_image.append(image) 
        elif self.cgi_mode == 'spec':
            # Extract point source spectra and positions
            point_source_spectra = input_scene.off_axis_source_spectrum
            point_source_x = input_scene.point_source_x
            point_source_y = input_scene.point_source_y

            # Ensure all inputs are lists for uniform processing
            if not isinstance(point_source_spectra, list):
                point_source_spectra = [point_source_spectra]
            if not isinstance(point_source_x, list):
                point_source_x = [point_source_x]
            if not isinstance(point_source_y, list):
                point_source_y = [point_source_y]

            # Ensure all lists have the same length
            if not (len(point_source_spectra) == len(point_source_x) == len(point_source_y)):
                raise ValueError(
                    f"Mismatch in input lengths: {len(point_source_spectra)} spectra, "
                    f"{len(point_source_x)} x-positions, {len(point_source_y)} y-positions. "
                    "Each point source must have a corresponding (x, y) position.")

            if self.slit != 'None':
                field_stop_array, field_stop_sampling_m = spec.get_slit_mask(self)
                self.proper_keywords['field_stop_array']=field_stop_array
                self.proper_keywords['field_stop_array_sampling_m']=field_stop_sampling_m
            else:
                self.proper_keywords['field_stop_array']=0
                self.proper_keywords['field_stop_array_sampling_m']=0

            # Compute the observed  spectrum for each off-axis source
            obs_point_source = [Observation(spectrum, self.bp) for spectrum in point_source_spectra]
            
            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor
            
            point_source_image = []
            for j in range(len(point_source_spectra )):
                proper_keywords_comp = self.proper_keywords.copy()
                proper_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                            'final_sampling_m': sampling_um_tem * 1e-6,
                                            'source_x_offset_mas': point_source_x[j],
                                            'source_y_offset_mas': point_source_y[j]})

                (fields, sampling) = proper.prop_run_multi('roman_preflight', self.lam_um, 1024, PASSVALUE=proper_keywords_comp ,QUIET=True)
                images_tem = np.abs(fields)**2

                # If a prism was selected, apply the dispersion model and overwrite the image cube and wavelength array.
                if self.prism != 'None': 
                    images_tem, dispersed_lam_um = spec.apply_prism(self, images_tem)
    
                    self.nlam = len(dispersed_lam_um)
                    self.lam_um = dispersed_lam_um
                    dlam_um = dispersed_lam_um[1] - dispersed_lam_um[0]

                # Initialize the image array based on whether oversampling is returned
                images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
                images = np.zeros(images_shape, dtype=float) 

                for i in range(images_tem.shape[0]):
                    if self.return_oversample:
                        ##return the oversampled PSF, default 7 grid per pixel
                        images[i,:,:] +=  images_tem[i,:,:]
                    else:
                        ## integrate oversampled PSF back to one grid per pixel
                        images[i,:,:] +=  images_tem[i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2
                        ## update the proper_keywords['output_dim'] baclk to non_oversample size
                        self.proper_keywords['output_dim'] = self.grid_dim_out


                    dlam_um = self.lam_um[1]-self.lam_um[0]
                    lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                    lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                    # ares in unit of cm^2
                    # counts in unit of photos/s
                    counts = self.polarizer_transmission * obs_point_source[j].countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                    images[i,:,:] = images[i,:,:] * counts

                image = np.sum(images, axis=0)
                point_source_image.append(image) 

        if self.cgi_mode in ['lowfs', 'excam_efield']:
            raise ValueError(f"The mode '{self.cgi_mode}' has not been implemented yet!")
        
        if sim_scene == None:
            sim_scene = scene.SimulatedImage(input_scene)

        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        npl = len(input_scene.point_source_Vmag)
        sim_info = {'num_off_axis_source': npl}
        ##update the brightness and position for ith companion
        for i in range(npl):
            sim_info[f'pl_Vmag_{i}'] = input_scene.point_source_Vmag[i]
            sim_info[f'pl_magtype_{i}']= input_scene.point_source_magtype[i]
            sim_info[f'position_x_mas_{i}'] = input_scene.point_source_x[i]
            sim_info[f'position_y_mas_{i}'] = input_scene.point_source_y[i]

        # Third: global simulation settings
        sim_info['cgi_mode'] = self.cgi_mode
        sim_info['cor_type'] = self.proper_keywords.get('cor_type')
        sim_info['bandpass'] = self.bandpass_header
        sim_info['over_sampling_factor'] = self.oversampling_factor
        sim_info['return_oversample'] = self.return_oversample
        sim_info['output_dim'] = self.proper_keywords['output_dim'] 
        sim_info['nd_filter'] = self.nd
                            
                # Define specific keys from self.proper_keywords to include in the header            
        keys_to_include_in_header = [ 'use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                            'use_lyot_stop','use_field_stop','fsm_x_offset_mas','fsm_y_offset_mas']  # Specify keys to include
        subset = {key: self.proper_keywords[key] for key in keys_to_include_in_header if key in self.proper_keywords}
        sim_info.update(subset)
        sim_info['includ_dectector_noise'] = 'False'
        # Create the HDU object with the generated header information

        sim_scene.point_source_image = outputs.create_hdu( np.sum(point_source_image,axis=0), sim_info =sim_info)

        return sim_scene



    
class CorgiDetector(): 
    
    def __init__(self ,emccd_keywords, photon_counting = True):
        '''
        Initialize the class with a dictionary that defines the EMCCD_DETECT input parameters. 

        Arguments: 
        emccd_keywords: A dictionary with the keywords that are used to set up the emccd model
        photon_counting: if use photon_counting mode, default is True
        '''
        self.emccd_keywords = emccd_keywords  # Store the keywords for later use
        #self.exptime = exptime ##expsoure time in second
        self.photon_counting = photon_counting


        self.emccd = self.define_EMCCD(emccd_keywords=self.emccd_keywords)
    

    def generate_detector_image(self, simulated_scene, exptime, full_frame= False, loc_x=None, loc_y=None):
        '''
        Function that generates a detector image from the input image, using emccd_detect. 

        The input_image probably has to be in electrons. 

        Arguments:
        simulated_scene: a corgisim.scene.SimulatedScen object that contains the noise-free scene from CorgiOptics
        full_frame: if generated full_frame image in detetor
        loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, 
                     and image from CorgiOptics has size is smaller than 1024×1024
        loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True,
                     and image from CorgiOptics has size is smaller than 1024×1024
        exptime: exptime in second

        Returns:
        A corgisim.scene.SimulatedImage object that contains the detector image in the 
        '''
        # List of possible image components (in order of addition)

        img = None
        sim_info = {}
        components = [simulated_scene.host_star_image,
                      simulated_scene.point_source_image,
                      simulated_scene.twoD_image]
        
        ###check witch components is not None, and combine exsiting simulated scene
        ### read comment header from components is not None to track sim_info
        for component in components:
            if component is not None:
                data = component.data
                if img is None:
                    img = data.copy()  # initialize from first valid image
                else:
                    img += data

                # Collect COMMENT headers if available
                if 'COMMENT' in component.header:
                    comment_lines = component.header['COMMENT']
                    
                for line in comment_lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        sim_info[key.strip()] = val.strip()
                        

        if img is None:
            raise ValueError('No valid simulated scene to put on detector')
      

        if full_frame:
            # If the simulated image is smaller than 1024×1024, place it on the full-frame detector at (loc_x, loc_y)
            # If the image is exactly 1024×1024, assume it's already centered and use it directly
            # If the image exceeds 1024×1024, raise an error
            if (img.shape[0] < 1024) & (img.shape[1] < 1024):
                flux_map = self.place_scene_on_detector( img , loc_x, loc_y)
            if (img.shape[0] == 1024) & (img.shape[1] == 1024):
                flux_map = img
            if (img.shape[0] >1024) or (img.shape[1] >1024):
                raise ValueError("Science image dimensions (excluding pre-scan area) cannot exceed 1024×1024.")
           
            Im_noisy = self.emccd.sim_full_frame(flux_map, exptime).astype(np.uint16)
        else:
            Im_noisy = self.emccd.sim_sub_frame(img, exptime).astype(np.uint16)

        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        sim_info['includ_dectector_noise'] = 'True'
        subset = {key: self.emccd_keywords_default[key] for key in self.emccd_keywords_default}
        sim_info.update(subset)
        
        # Create the HDU object with the generated header information
        if full_frame:
            sim_info['position_on_detector_x'] = loc_x
            sim_info['position_on_detector_y'] = loc_y
            
            if (sim_info['ref_flag'] == 'False') or (sim_info['ref_flag'] == '0'):
                ref_flag = False
            if (sim_info['ref_flag'] == 'True') or (sim_info['ref_flag'] == '1'):
                ref_flag = True
            header_info = {'EXPTIME': exptime,'EMGAIN_C':self.emccd_keywords_default['em_gain'],'PSFREF':ref_flag,
                           'PHTCNT':self.photon_counting,'KGAINPAR':self.emccd_keywords_default['e_per_dn'],'cor_type':sim_info['cor_type'], 'bandpass':sim_info['bandpass'],
                           'cgi_mode': sim_info['cgi_mode'], 'polaxis':sim_info['polaxis']}
            if 'fsm_x_offset_mas' in sim_info:
                header_info['FSMX'] = float(sim_info['fsm_x_offset_mas'])
            if 'fsm_y_offset_mas' in sim_info:
                header_info['FSMY'] = float(sim_info['fsm_y_offset_mas'])
            simulated_scene.image_on_detector = outputs.create_hdu_list(Im_noisy, sim_info=sim_info, header_info = header_info)
        else:
            simulated_scene.image_on_detector = outputs.create_hdu(Im_noisy, sim_info=sim_info)

        return simulated_scene

    def place_scene_on_detector(self, sub_frame, loc_x, loc_y):
        """
        Place the simulated scene onto the detector centered at a specified point.

        This method inserts a sub-frame (the simulated scene) into a larger 1024x1024 detector array,
        where the (loc_x, loc_y) coordinates represent the center of the sub_frame in the detector's
        coordinate system. The rest of the detector frame is padded with zeros. The resulting image is used
        later to generate an astropy HDU with associated metadata via the emccd_detect function.

        Note:
            This function is an intermediate step in the simulation pipeline and is generally not modified by
            end users.

        Args:
            sub_frame (numpy.ndarray): A 2D array representing the simulated scene to be placed on the detector.
            loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted.
            loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted.

        Returns:
            numpy.ndarray: A 1024x1024 2D array (detector frame) with the sub_frame placed at the specified 
                        center location and the remaining areas padded with zeros.

        Raises:
            ValueError: If the sub_frame, when placed at the specified location, exceeds the bounds of the 1024x1024 detector
                        array or if negative indices result.
        """
        # Create the large 1024x1024 array filled with zeros
        full_frame = np.zeros((1024, 1024))

        N_pix_x = sub_frame.shape[0]
        N_pix_y = sub_frame.shape[1]

        # Compute the indices for placing the sub-frame centered at (loc_x, loc_y)
        x_start = loc_x - N_pix_x // 2
        x_end = x_start + N_pix_x
        y_start = loc_y - N_pix_y // 2
        y_end = y_start + N_pix_y

        # Check for out-of-bounds placement
        if x_start < 0 or y_start < 0 or x_end > 1024 or y_end > 1024:
            raise ValueError('The subframe cannot be placed in the given location without exceeding detector bounds.')

        # Insert the small array into the large array
        full_frame[x_start:x_end, y_start:y_end] = sub_frame

        return full_frame



    def define_EMCCD(self, emccd_keywords=None ):

        '''
        Create and configure an EMCCD detector object with optional CTI simulation.

        This method initializes an EMCCD detector object using the provided parameters.
        Optionally, if CTI (Charge Transfer Inefficiency) simulation is required, it updates
        the detector object using a trap model.

        Args:
        # default values match requirements, except QE, which is year 0 curve (already accounted for in counts)
        em_gain (float, optional): EM gain, default 1000.
        full_well_image (float, optional): image full well; 50K is requirement, 60K is CBE
        full_well_serial (float, optional): full well for serial register; 90K is requirement, 100K is CBE
        dark_rate (float, optional): Dark current rate, e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
        cic_noise (float, optional): Clock-induced charge noise, e-/pix/frame; Defaults to 0.01.
        read_noise (float, optional): Read noise, e-/pix/frame; 125 is requirement, 100 is CBE
        bias (int, optional): Bias level (in digital numbers). Defaults to 0.
        qe (float): Quantum efficiency, set to 1 here, because already counted in counts
        cr_rate (int, optional): Cosmic ray event rate, hits/cm^2/s (0 for none, 5 for L2) 
        pixel_pitch (float, optional): Pixel pitch (in meters). Defaults to 13e-6.
        e_per_dn (float, optional): post-multiplied electrons per data unit
        numel_gain_register (int, optional): Number of elements in the gain register. Defaults to 604.
        nbits (int, optional): Number of bits in the analog-to-digital converter. Defaults to 14.
        use_traps (bool, optional): Flag indicating whether to simulate CTI effects using trap models. Defaults to False.
        date4traps (float, optional): Decimal year of observation; only applicable if `use_traps` is True. Defaults to 2028.0.

        Returns:
        emccd (EMCCDDetectBase): A configured EMCCD detector object. If `use_traps` is True, the detector's CTI is updated
                         using the corresponding trap model.
        
        '''
        # Initialize emccd_keywords safely
        #if emccd_keywords is None:
        # default parameters for RMCCD on Roman-CGI, except QE, which set to 1 (already accounted for in counts)
        self.emccd_keywords_default = {'full_well_serial': 105000.0,         # full well for serial register; 90K is requirement, 100K is CBE
                                  'full_well_image': 90000.0,                 # image full well; 50K is requirement, 60K is CBE
                                  'dark_rate': 0.001,                  # e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
                                  'cic_noise': 0.0088,                    # e-/pix/frame; 0.1 is requirement, 0.01 is CBE
                                  'read_noise': 165.0,                  # e-/pix/frame; 125 is requirement, 100 is CBE
                                  'cr_rate': 5,                         # hits/cm^2/s (0 for none, 5 for L2) 
                                  'em_gain': 1000.0 ,                      # EM gain
                                  'bias': 0,
                                  'pixel_pitch': 13e-6 ,                # detector pixel size in meters
                                  'apply_smear': True ,                 # (LOWFS only) Apply fast readout smear?  
                                  'e_per_dn':8.7  ,                    # post-multiplied electrons per data unit
                                  'nbits': 14  ,                        # ADC bits
                                  'numel_gain_register': 604,           # Number of gain register elements 
                                  'use_traps': False,                    # include CTI impact of traps
                                  'date4traps': 2028.0}                        # decimal year of observation}

        if emccd_keywords is not None:                    
            if 'qe' in emccd_keywords.keys():
                raise Warning("Quantum efficiency has been added in the bandpass throughput; it must be enforced as 1 here.")
            # Override default parameters with user-specified ones
            for key, value in emccd_keywords.items():
                if key in self.emccd_keywords_default:
                    self.emccd_keywords_default[key] = value

    
        emccd = EMCCDDetect( em_gain=self.emccd_keywords_default['em_gain'], full_well_image=self.emccd_keywords_default['full_well_image'], full_well_serial=self.emccd_keywords_default['full_well_serial'],
                             dark_current=self.emccd_keywords_default['dark_rate'], cic=self.emccd_keywords_default['cic_noise'], read_noise=self.emccd_keywords_default['read_noise'], bias=self.emccd_keywords_default['bias'],
                             qe=1.0, cr_rate=self.emccd_keywords_default['cr_rate'], pixel_pitch=self.emccd_keywords_default['pixel_pitch'], eperdn=self.emccd_keywords_default['e_per_dn'],
                             numel_gain_register=self.emccd_keywords_default['numel_gain_register'], nbits=self.emccd_keywords_default['nbits'] )
        
        if self.emccd_keywords_default['use_traps']: 
            raise ValueError(f"The part to simulate CTI effects using trap models has not been implemented yet!")
        
            #from cgisim.rcgisim import model_for_Roman
            #from arcticpy.roe import ROE
            #from arcticpy.ccd import CCD
            #traps = model_for_Roman( date4traps )  
            #ccd = CCD(well_fill_power=0.58, full_well_depth=full_well_image)
            #roe = ROE()
            #emccd.update_cti( ccd=ccd, roe=roe, traps=traps, express=1 )
        
        return emccd


