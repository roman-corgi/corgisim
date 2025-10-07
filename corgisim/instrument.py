import proper
import warnings
import numpy as np
from astropy.io import fits
import roman_preflight_proper
from corgisim import scene
from corgisim.sat_spots import add_cos_pattern_dm
import cgisim
import corgisim
from synphot.models import BlackBodyNorm1D, Box1D,Empirical1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
import matplotlib.pyplot as plt
from emccd_detect.emccd_detect import EMCCDDetectBase, EMCCDDetect
from corgidrp import mocks
from corgisim import outputs, spec, pol
import copy
import os
from scipy import interpolate
import time
import sys
import astropy.units as u

from corgisim.convolution import (
    create_wavelength_grid_and_weights,
    build_radial_grid,
    build_azimuth_grid,
    convolve_with_prfs, 
    get_valid_polar_positions,
    ARCSEC_PER_RAD,
    PIXEL_SCALE_ARCSEC
)

from corgisim.scene import Scene, SimulatedImage

class CorgiOptics():
    '''
    A class that defines the current configuration of the CGI optics, including the telescope
    and the coronagraph.

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    It will need to know the telescope roll angle. 

    '''

    def __init__(self, cgi_mode = None, bandpass= None,  diam = 236.3114, optics_keywords=None, roll_angle= 0,satspot_keywords=None, oversampling_factor = 7, return_oversample = False, **kwargs):
        '''

        Initialize the class a keyword dictionary that defines the setup of cgisim/PROPER 
        and other relevant information (such as the telescope roll angle).


        Initialize the class with two dictionaries: 
        Args:
            - cgi_mode (str): define cgi simulation mode, valid values: 'excam', 'spec', ‘lowfs’, ‘excam_efield’
            - cor_type (str): define coronagraphic observing modes
            - bandpass (str): pre-difined bandpass for Roman-CGI
            - diam (float) in meter: diameter of the primaru mirror, the default value is 2.363114 meter
            - optics_keywords: A dictionary with the keywords that are used to set up the proper model
	        - satspot_keywords: A dictionary with the keywords that are used to add satellite spots. See add_satspot for the keywords.
            - oversampling_factor: An integer that defines the oversampling factor of the detector when generating the image
            - return_oversample: A boolean that defines whether the function should return the oversampled image or not.
            - roll_angle : float, optional, Telescope roll angle in degrees (0 to 360). 
                           Defined as the rotation angle of the excam coordinates (X, Y) relative to the sky coordinates(RA,DEC), positive is counter-clockwise.
                           Default is 0 degrees, corresponding to North up, East left in the sky coordinates.
        Raises:
            - ValueError: If `cgi_mode` or `cor_type` is invalid.
            - KeyError: If required `optics_keywords` are missing.
            - KeyError: If forbidden keywords are included.
        
        '''
        
         # Initialize optics_keywords safely
        if optics_keywords is None:
            raise KeyError(f"ERROR: optics_keywords are required to create an Optics object")
        
        optics_keywords_internal = optics_keywords.copy()
        #some parameters to the PROPER prescription are required, including 'cor_type', 'polaxis'
        required_keys = {'cor_type', 'polaxis', 'output_dim'}
        missing_keys = required_keys - optics_keywords_internal.keys()

        if missing_keys:
            raise KeyError(f"ERROR: Missing required optics_keywords: {missing_keys}")

        # some parameters to the PROPER prescription are not allowed when calling it from corgisim;
        ## 'final_sampling_m' is directly choosed based on different cgi mode 
        ## 'end_at_fpm_exit_pupil','end_at_fsm' are not allowed because they will give outimage at fsm 
        forbidden_keys = {'final_sampling_lam0', 'final_sampling_m', 'end_at_fpm_exit_pupil','end_at_fsm'}
        forbidden_found = forbidden_keys & optics_keywords_internal.keys()

        if forbidden_found:
            raise KeyError(f"ERROR: Forbidden keywords detected in optics_keywords: {forbidden_found}")


        valid_cgi_modes = ['excam', 'spec', 'lowfs', 'excam_efield']
        valid_cor_types = ['hlc', 'hlc_band1', 'spc-wide', 'spc-wide_band4', 
                        'spc-wide_band1', 'hlc_band2', 'hlc_band3', 'hlc_band4','spc-spec', 'spc-spec_band2', 'spc-spec_band3' ]
        
        #these cor_type is availbale in cgisim, but are currently untested in corgisim
        untest_cor_types = ['spc-spec_rotated', 'spc-spec_band2_rotated', 'spc-spec_band3_rotated','spc-mswc', 'spc-mswc_band4','spc-mswc_band1', 'zwfs']

        if cgi_mode not in valid_cgi_modes:
            raise Exception('ERROR: Requested mode does not match any available mode')
     

        if optics_keywords_internal['cor_type'] not in valid_cor_types and optics_keywords_internal['cor_type'] not in untest_cor_types:
            raise Exception('ERROR: Requested coronagraph does not match any available types')
        
        if optics_keywords_internal['cor_type'] in untest_cor_types:
            warnings.warn('Warning: Requested coronagraph is currently untested and might not work as expected')

        self.cgi_mode = cgi_mode
        self.cor_type = optics_keywords_internal['cor_type']
        self.roll_angle = roll_angle % 360  # Ensure roll angle is within 0-360 degrees

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

        mode_data, bandpass_data = cgisim.cgisim_read_mode( cgi_mode, optics_keywords_internal['cor_type'], self.bandpass, info_dir )

        # Set directory containing reference data for parameters external to CGISim
        ref_data_dir = os.path.join(corgisim.lib_dir, 'data')
        if not os.path.exists(ref_data_dir):
            raise FileNotFoundError(f"Directory does not exist: {ref_data_dir}")
        else:
            self.ref_data_dir = ref_data_dir
        #set polarimetry wollaston prism parameter
        if self.cgi_mode == 'excam':
            #DPAM prisms allowed for polarimetry
            valid_prisms = ['None', 'POL0', 'POL45']
            if 'prism' in optics_keywords:
                if optics_keywords['prism'] not in valid_prisms:
                    raise ValueError(f'Invalid value for prism: {optics_keywords['prism']}.'
                                     f'Must be one of: {valid_prisms}')
                setattr(self, 'prism', optics_keywords['prism'])
            else:
                setattr(self, 'prism', 'None')
        # Set the spectroscopy parameters
        if self.cgi_mode == 'spec':
            spec_kw_defaults = {
                'slit': 'None', # named FSAM slit
                'slit_ra_offset_mas': 0.0, # offset of slit position from star on sky coordinate, in mas
                'slit_dec_offset_mas': 0.0, # offset of slit position from star on sky coordinate, in mas
                'slit_x_offset_mas': None, # offset of slit position from star on excam coordinate, in mas
                'slit_y_offset_mas': None, # offset of slit position from star on excam coordinate, in mas
                'prism': 'None', # named DPAM prism
                'wav_step_um': 1E-3, # wavelength step size of the prism dispersion model, in microns 
                'fsm_x_offset_mas': 0.0, # FSM x offset in mas 
                'fsm_y_offset_mas': 0.0  # FSM y offset in mas
            }
            #### allowed slit for band2 
            if '2' in self.bandpass:     
                spec_kw_allowed = {
                    'slit': ['None', 'R6C5', 'R3C1'],
                    'prism': ['None', 'PRISM3', 'PRISM2']}
            #### allowed slit for band3
            elif '3' in self.bandpass:
                spec_kw_allowed = {
                    'slit': ['None', 'R1C2', 'R3C1'],
                    'prism': ['None', 'PRISM3', 'PRISM2']}
            for attr_name, default_value in spec_kw_defaults.items():
                if attr_name in optics_keywords:
                    value = optics_keywords[attr_name]
                
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
            
            # If excam coordinates are not provided, compute them from sky coordinates
            if (self.slit_x_offset_mas is None) and (self.slit_y_offset_mas is None):
                # Convert slit location from sky coordinates to EXCAM coordinates (mas)
                self.slit_x_offset_mas, self.slit_y_offset_mas = skycoord_to_excamcoord(self.slit_ra_offset_mas, self.slit_dec_offset_mas, self.roll_angle)

            # If excam coordinates are provided, they take precedence over sky coordinates
            elif (self.slit_x_offset_mas is not None) and (self.slit_y_offset_mas is not None):
                warnings.warn(
                    "Slit location provided in EXCAM coordinates (x, y). "
                    "These values will override the sky-coordinate inputs (dRA, dDec). "
                    "Please ensure the EXCAM coordinates correspond to the intended slit position on the sky.",
                    UserWarning,
                )

                
            if self.prism != 'None':
                prism_param_fname = os.path.join(ref_data_dir, 'TVAC_{:s}_dispersion_profile.npz'.format(self.prism))
                if not os.path.exists(prism_param_fname):
                    raise FileNotFoundError(f"Prism parameter file {prism_param_fname} does not exist")
                else:
                    setattr(self, 'prism_param_fname', prism_param_fname)

            else:
                warnings.warn("No prism selected in spec mode, the dispersion model will not be applied to the image cube.")
            
            ### give a warning if the prism is not the default one for the bandpass
            if (self.prism == 'PRISM2')&('3' in self.bandpass):
                warnings.warn("PRISM2 is selected for Band 3, which is not the default setting for the Roman CGI, but it can still be simulated with CorgiSim.")
            if (self.prism == 'PRISM3')&('2' in self.bandpass):
                warnings.warn("PRISM3 is selected for Band 2, which is not the default setting for the Roman CGI, but it can still be simulated with CorgiSim.")
            
            if self.slit != 'None':
                slit_param_fname = os.path.join(ref_data_dir, 'FSAM_slit_params.json')
                if not os.path.exists(slit_param_fname):
                    raise FileNotFoundError(f"Slit aperture parameter file {slit_param_fname} does not exist")
                else:
                    setattr(self, 'slit_param_fname', slit_param_fname)
            else:
                warnings.warn("No slit selected in spec mode, the slit mask will not be applied to the image cube.")
            

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

        self.diam = diam ## diameter of Roman primary in cm, default is 236.114 cm
        # Effective collecting area in unit of cm^2, 
        # 30.3% central obscuration of the telescope entrance pupil (diameter ratio) from IPAC-Roman website
        #self.area = (self.diam/2)**2 * np.pi - (self.diam/2*0.303)**2 * np.pi
        self.area =  35895.212    # primary effective area from cgisim cm^2 
        self.grid_dim_out = optics_keywords_internal['output_dim'] # number of grid in output image in one dimension
        
        optics_keywords_internal['lam0']=self.lam0_um
        if 'use_fpm' not in optics_keywords_internal:
            optics_keywords_internal['use_fpm'] = 1  # use fpm by default
        if 'use_lyot_stop' not in optics_keywords_internal:
            optics_keywords_internal['use_lyot_stop'] = 1  # use lyot stop by default
        if 'use_field_stop' not in optics_keywords_internal:
            optics_keywords_internal['use_field_stop'] = 1  # use field stop by default
        if 'use_pupil_lens' not in optics_keywords_internal:
            optics_keywords_internal['use_pupil_lens'] = 0  # not use pupil lens by default

        if optics_keywords_internal['use_pupil_lens']==1 :
            if (optics_keywords_internal['use_fpm']==1) or (optics_keywords_internal['use_lyot_stop']==1) or (optics_keywords_internal['use_field_stop']==1):
                raise ValueError("When simulating a pupil image (use_pupil_lens=1), disable use_fpm, use_lyot_stop, and use_field_stop.")


        # polarization
        
        if optics_keywords_internal['polaxis'] != 10 and optics_keywords_internal['polaxis'] != -10 and optics_keywords_internal['polaxis'] != 0:
            # wollaston transmission is around 0.96%, divide by two to split between polarization
            self.polarizer_transmission = 0.48
        else:
            self.polarizer_transmission = 1.0

        #self.integrate_pixels = True ## whether to subsamping the pixels and integrate them
        #if "integrate_pixels" in kwargs: self.integrate_pixels = kwargs.get("integrate_pixels")
        
        ## setup if to oversampling the image and if return the oversample
        self.oversampling_factor = oversampling_factor
        self.return_oversample = return_oversample


        self.nd = 0  # integer: 1, 2, or 3 (0 = no ND, the default); this is the ND filter identifier, NOT the amount of ND
        #if "nd" in kwargs: self.nd = kwargs.get("nd")
        if "nd" in optics_keywords.keys():
            if optics_keywords["nd"] ==1:
                self.nd = '2.25'
            elif optics_keywords["nd"] ==2:
                self.nd = '4.75fpam'
            elif optics_keywords["nd"] ==3:
                self.nd = '4.75fsam'
            elif optics_keywords["nd"] ==0:
                self.nd = 0
            else:
                raise ValueError(f"Invalid ND filter value: {optics_keywords['nd']}. Must be 0, 1, 2, or 3.")
            
        # Initialize the bandpass class (from synphot)
        # bp: wavelegth is in unit of angstrom
        # bp: throughput is unitless, including transmission, reflectivity and EMCCD quantum efficiency 
        self.bp = self.setup_bandpass(self.cgi_mode, self.bandpass, self.nd)
        self.optics_keywords = optics_keywords_internal  # Store the keywords for PROPER package



        if 'if_quiet'in kwargs:self.quiet = kwargs.get("if_quiet")

        ##self.SATSPOTS is the value to be populated to L1 header prihdr[SATSPOTS]
        # prihdr[SATSPOTS]= 0: No satellite spots present 
        # prihdr[SATSPOTS]= 1: Satellite spots present
        if satspot_keywords == None:
            self.SATSPOTS = int(0)
        else:
            # check keywords
            if optics_keywords['use_dm1'] != 1:
                raise KeyError('ERROR: use_dm1 in optics_keywords is not set 1')
            required_keys_satspot = {'num_pairs','sep_lamD', 'angle_deg', 'contrast', 'wavelength_m'}
            missing_keys = required_keys_satspot - satspot_keywords.keys()
            if missing_keys:
                raise KeyError(f"ERROR: Missing required satspot_keywords: {missing_keys}")

            #### call self.add_satspot() to satellite spots in DM files, update the dm1 info in self.optics_keywords
            self.optics_keywords['dm1_v'] = self.add_satspot(satspot_keywords=satspot_keywords)

            self.SATSPOTS = int(1)
            print("satellite spots are added to DM1.")

        print("CorgiOptics initialized with proper keywords.")
     
    def get_e_field(self):
        '''
        Function that only returns the e fields 
        Returns: 
            - fields: an array that contains the field

        '''
        grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
        sampling_um_tem = self.sampling_um / self.oversampling_factor

        self.optics_keywords['output_dim']=grid_dim_out_tem
        self.optics_keywords['final_sampling_m']=sampling_um_tem *1e-6
        
        
        (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=self.optics_keywords,QUIET=self.quiet)

        # Initialize the image array based on whether oversampling is returned
        images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
        images = np.zeros(images_shape, dtype=complex)
    
        for i in range(fields.shape[0]):
            ## integrate oversampled PSF back to one grid per pixel
            images[i,:,:] +=  fields[i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2
            ## update the optics_keywords['output_dim'] baclk to non_oversample size
        self.optics_keywords['output_dim'] = self.grid_dim_out

        return images
        
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

            self.optics_keywords['output_dim']=grid_dim_out_tem
            self.optics_keywords['final_sampling_m']=sampling_um_tem *1e-6

            #if polarimetry mode is enabled
            if self.prism == 'POL0':
                #0/90 case
                # models the polarization aberration of the speckle field
                # polaxis=-1 and 1 gives -45->X and 45->X aberrations, incoherently
                # averaging the two gives the x polarized intensity data.
                # polaxis=-2 and 2 gives -45->Y and 45->Y aberrations, incoherently
                # averaging the two gives the y polarized intensity data. 
                polaxis_params = [-1, 1, -2, 2]
                fields = []
                optics_keywords_pol_xy = self.optics_keywords.copy()
                for polaxis in polaxis_params:
                    optics_keywords_pol_xy['polaxis'] = polaxis
                    (field, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=optics_keywords_pol_xy,QUIET=self.quiet)
                    fields.append(field)
                #obtain 0/90 degree polarization intensities
                intensity_x = ((np.abs(fields[0]) ** 2) + (np.abs(fields[1]) ** 2)) / 2
                intensity_y = ((np.abs(fields[2]) ** 2) + (np.abs(fields[3]) ** 2)) / 2
                images_tem = [intensity_x, intensity_y]
            elif self.prism == 'POL45':
                #45/135 case
                # models the polarization aberration of the speckle field
                # polaxis=-3 and 3 gives -45->45 and 45->45 aberrations, incoherently
                # averaging the two gives the 45 degree polarized intensity data.
                # polaxis=-2 and 2 gives -45->-45 and 45->-45 aberrations, incoherently
                # averaging the two gives the -45 degree polarized intensity data. 
                polaxis_params = [-3, 3, -4, 4]
                fields = []
                optics_keywords_pol_45 = self.optics_keywords.copy()
                for polaxis in polaxis_params:
                    optics_keywords_pol_45['polaxis'] = polaxis
                    (field, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=optics_keywords_pol_45,QUIET=self.quiet)
                    fields.append(field)
                #obtain 45/135 degree polarization intensities
                intensity_45 = ((np.abs(fields[0]) ** 2) + (np.abs(fields[1]) ** 2)) / 2
                intensity_135 = ((np.abs(fields[2]) ** 2) + (np.abs(fields[3]) ** 2)) / 2
                images_tem = [intensity_45, intensity_135]
            elif self.optics_keywords['polaxis'] == -10:
                # if polaxis is set to -10, obtain full aberration model by individually summing intensities obtained from polaxis=-2, -1, 1, 2
                optics_keywords_m10 = self.optics_keywords.copy()
                polaxis_params = [-2, -1, 1, 2]
                images_pol = []
                for polaxis in polaxis_params:
                    optics_keywords_m10['polaxis'] = polaxis
                    (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=optics_keywords_m10,QUIET=self.quiet)
                    images_pol.append(np.abs(fields) ** 2)
                images_tem = np.array(sum(images_pol)) / 4
            else: 
                # use built in polaxis settings to obtain specific/averaged aberration 
                (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=self.optics_keywords,QUIET=self.quiet)
                images_tem = np.abs(fields)**2
            # construct the correct image array size depending on if wollaston is used or not
            if self.prism in ['POL0', 'POL45']:
                # Initialize the image array based on whether oversampling is returned
                images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
                images_1 = np.zeros(images_shape, dtype=float)
                images_2 = np.zeros(images_shape, dtype=float)

                for i in range(images_tem[0].shape[0]):
                    if self.return_oversample:
                        ##return the oversampled PSF, default 7 grid per pixel
                        images_1[i,:,:] +=  images_tem[0][i,:,:]
                        images_2[i,:,:] += images_tem[1][i,:,:]
                    else:
                    ## integrate oversampled PSF back to one grid per pixel
                        images_1[i,:,:] +=  images_tem[0][i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2
                        images_2[i,:,:] +=  images_tem[1][i,:,:].reshape((self.grid_dim_out,self.oversampling_factor,self.grid_dim_out,self.oversampling_factor)).mean(3).mean(1) * self.oversampling_factor**2

                    dlam_um = self.lam_um[1]-self.lam_um[0]
                    lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                    lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                    # ares in unit of cm^2
                    # counts in unit of photos/s
                    # wollaston transmission is around 0.96%, divide by two to split between polarization
                    counts = 0.48 * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                    images_1[i,:,:] = images_1[i,:,:] * counts
                    images_2[i,:,:] = images_2[i,:,:] * counts

                # 3D datacube with the 3rd axis being polarization 
                image = np.array([np.sum(images_1, axis=0), np.sum(images_2, axis=0)])
            else:
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
                        ## update the optics_keywords['output_dim'] baclk to non_oversample size
                        self.optics_keywords['output_dim'] = self.grid_dim_out

                    dlam_um = self.lam_um[1]-self.lam_um[0]
                    lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
                    lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
                    # ares in unit of cm^2
                    # counts in unit of photos/s
                    counts = self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])

                    images[i,:,:] = images[i,:,:] * counts
                
                # 2D array if no wollaston is used
                image = np.sum(images, axis=0)

        elif self.cgi_mode == 'spec':
            if self.slit != 'None':
                field_stop_array, field_stop_sampling_m = spec.get_slit_mask(self)
                self.optics_keywords['field_stop_array']=field_stop_array
                self.optics_keywords['field_stop_array_sampling_m']=field_stop_sampling_m
            else:
                self.optics_keywords['field_stop_array']=0
                self.optics_keywords['field_stop_array_sampling_m']=0
                
            obs = Observation(input_scene.stellar_spectrum, self.bp)

            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor

            self.optics_keywords['output_dim']=grid_dim_out_tem
            self.optics_keywords['final_sampling_m']=sampling_um_tem *1e-6
            
            if self.optics_keywords['polaxis'] == -10:
                optics_keywords_m10 = self.optics_keywords.copy()
                images_tem = self.generate_full_aberration_psf(optics_keywords_m10)
            else: 
                (fields, sampling) = proper.prop_run_multi('roman_preflight', self.lam_um, 1024, PASSVALUE=self.optics_keywords, QUIET=self.quiet)
                images_tem = np.abs(fields)**2

            # If a prism was selected, apply the dispersion model and overwrite the image cube and wavelength array.
            if self.prism != 'None': 
                images_tem, dispersed_lam_um, disp_shift_lam0_x, disp_shift_lam0_y = spec.apply_prism(self, images_tem)

                self.nlam = len(dispersed_lam_um)
                self.lam_um = dispersed_lam_um
                dlam_um = dispersed_lam_um[1] - dispersed_lam_um[0]

                mas_pix = 500E-9 * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000 / 2
                (frame_loc_x, frame_loc_y) = (512, 512)
                image_centx = self.grid_dim_out // 2 + self.fsm_x_offset_mas / mas_pix
                image_centy = self.grid_dim_out // 2 + self.fsm_y_offset_mas / mas_pix
                print("source location (x, y) without prism = {:.3f}, {:.3f}".format(image_centx, image_centy))
                self.optics_keywords['dispersed_image_centx'] = image_centx + disp_shift_lam0_x / self.oversampling_factor 
                self.optics_keywords['dispersed_image_centy'] = image_centy + disp_shift_lam0_y / self.oversampling_factor 
                print("source location (x, y) with prism = {:.3f}, {:.3f}".format(self.optics_keywords['dispersed_image_centx'], 
                                                                                  self.optics_keywords['dispersed_image_centy']))
                self.optics_keywords['dispersed_fullframe_centx'] = frame_loc_x + 1088 + (self.optics_keywords['dispersed_image_centx'] - self.grid_dim_out // 2)
                self.optics_keywords['dispersed_fullframe_centy'] = frame_loc_y + 13 + (self.optics_keywords['dispersed_image_centy'] - self.grid_dim_out // 2)
                print("full frame source location (x, y) with prism = {:.3f}, {:.3f}".format(self.optics_keywords['dispersed_fullframe_centx'],
                                                                                             self.optics_keywords['dispersed_fullframe_centy']))

            # Initialize the image array based on whether oversampling is returned
            images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
            images = np.zeros(images_shape, dtype=float)
            counts = np.zeros(self.nlam)

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
                counts[i] = self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u]).value

            images *= counts[:, np.newaxis, np.newaxis]
            image = np.sum(images, axis=0)

        if self.cgi_mode in ['lowfs', 'excam_efield']:
            raise ValueError(f"The mode '{self.cgi_mode}' has not been implemented yet!")
        
        # Initialize SimulatedImage class to restore the output psf
        if sim_scene == None:
            sim_scene = scene.SimulatedImage(input_scene)
        
        
        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        ## determine header info based on prism used
        if self.prism == 'POL0':
            polarization_basis = '0/90 degrees'
        elif self.prism == 'POL45':
            polarization_basis = '45/135 degrees'
        else:
            polarization_basis = 'None'
        sim_info = {'host_star_sptype':input_scene.host_star_sptype,
                    'host_star_Vmag':input_scene.host_star_Vmag,
                    'host_star_magtype':input_scene.host_star_magtype,
                    'ref_flag':input_scene.ref_flag,
                    'cgi_mode':self.cgi_mode,
                    'cor_type': self.optics_keywords['cor_type'],
                    'bandpass':self.bandpass_header,
                    'polarization_basis': polarization_basis,
                    'over_sampling_factor':self.oversampling_factor,
                    'return_oversample': self.return_oversample,
                    'output_dim': self.optics_keywords['output_dim'],
                    'nd_filter':self.nd}

        # Define specific keys from self.optics_keywords to include in the header            
        keys_to_include_in_header = ['use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                            'slit_x_offset_mas','slit_y_offset_mas','use_pupil_lens', 'use_lyot_stop', 'use_field_stop',
                            'fsm_x_offset_mas','fsm_y_offset_mas','slit','prism',
                            'dispersed_image_centx','dispersed_image_centy',
                            'dispersed_fullframe_centx','dispersed_fullframe_centy']  # Specify keys to include
        subset = {key: self.optics_keywords[key] for key in keys_to_include_in_header if key in self.optics_keywords}
        sim_info.update(subset)
        ## add sattelite spots info 
        sim_info['SATSPOTS'] = self.SATSPOTS
        sim_info['includ_dectector_noise'] = 'False'
        sim_info['roll_angle'] = self.roll_angle
        # Create the HDU object with the generated header information

        sim_scene.host_star_image = outputs.create_hdu(image,sim_info =sim_info)

        return sim_scene

    def setup_bandpass(self, cgimode, bandpass_name, nd ):
        """
        Sets up the bandpass for the simulation.

        This function initializes a bandpass based on the given parameters, computes 
        wavelength boundaries, and retrieves throughput data.

        Args:
            - cgimode (str): The CGI mode to be used.
            - bandpass_name (str): Name of the bandpass filter.
            - nd (int): Neutral density filter index.

        Returns:
            - bp (SpectralElement): A spectral element object containing the bandpass data.

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

    def _compute_single_off_axis_psf(self, radius_lamD, azimuth_angle, 
                                    wavelength_grid, wavelength_weights):
        """
        Compute a weighted off-axis PSF at a given polar position (r, θ).

        Parameters
        ----------
        radius_lamD : float
            Off-axis distance in λ/D.
        azimuth_deg : float
            Off-axis angle in degrees.
        lam_grid : ndarray
            Wavelengths to simulate (in microns).
        lam_weights : ndarray
            Normalised weights corresponding to each wavelength.

        Returns
        -------
        weighted_psf : ndarray
            2D PSF image, weighted across the spectral band.
        """

        # Convert polar to Cartesian coordinates
        dx = self.res_mas * radius_lamD * np.cos(azimuth_angle.to_value(u.rad))
        dy = self.res_mas * radius_lamD * np.sin(azimuth_angle.to_value(u.rad))

        grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
        sampling_um_tem = self.sampling_um / self.oversampling_factor

        print(f"Computing off-axis PSF at r={radius_lamD:.2f} λ/D, θ={azimuth_angle:.1f} deg -> (dx, dy)=({dx:.1f}, {dy:.1f}) mas")

        optics_keywords_comp = self.optics_keywords.copy()
        optics_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                       'final_sampling_m': sampling_um_tem * 1e-6,
                                       'source_x_offset_mas': dx,
                                       'source_y_offset_mas': dy})

        # # Configure PROPER simulation options
        # simulation_options = dict(self.optics_keywords,
        #                         source_x_offset=dx,
        #                         source_y_offset=dy,
        #                         output_dim=self.grid_dim_out * self.oversampling_factor,
        #                         final_sampling_m=)
        #                         # final_sampling_m=self.sampling_um / self.oversampling_factor * 1e-6)
        
        # Run PROPER simulation
        (fields, sampling) = proper.prop_run_multi('roman_preflight', wavelength_grid, 1024,PASSVALUE= optics_keywords_comp, QUIET=self.quiet)

        # Apply spectral weighting and bin down
        intensity = np.abs(fields)**2
        weighted_img = np.tensordot(wavelength_weights, intensity, axes=(0, 0))

        # TODO - Currently we skipped counts when calculating off-axis PSF for convolution
        # input_scene.twoD_scene_spectrum
 

        # Bin down to detector resolution
        binned = weighted_img.reshape(
            (self.grid_dim_out, self.oversampling_factor,
             self.grid_dim_out, self.oversampling_factor)
        ).mean(3).mean(1) * self.oversampling_factor**2

        return binned

    def make_prf_cube(self, radii_lamD, azimuths_deg, source_sed=None):
        """
        Build a psf cube by evaluating the off-axis PSF at specified polar positions.

        Parameters
        ----------
        radii_lamD : array-like
            Radial offsets in λ/D.
        azimuths_deg : array-like
            Azimuthal angles in degrees.
        source_sed : array-like or None
            Spectral energy distribution weights, optional.

        Returns
        -------
        prf_cube : ndarray, shape (N_positions, Ny, Nx)
            Cube of PSFs at all requested (r, θ) positions.
        """
        # TODO - source_sed is currently not used, just assumed to be flat. For scene with extended source, here should be disk spectrum?
        wavelength_grid, wavelength_weights = create_wavelength_grid_and_weights(self.lam_um, source_sed)
        valid_positions = get_valid_polar_positions(radii_lamD, azimuths_deg)   

        num_positions = len(valid_positions)
        prf_cube = np.empty((num_positions, self.grid_dim_out, self.grid_dim_out), dtype=np.float32)

        show_progress = num_positions > 10  # Show progress bar for larger jobs

        for i, (radius_lamD, azimuth_angle) in enumerate(valid_positions):
            prf_cube[i] = self._compute_single_off_axis_psf(radius_lamD, azimuth_angle, wavelength_grid, wavelength_weights)

            if show_progress:  # Only show progress for larger jobs
                progress = (i + 1) / num_positions
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r[{bar}] {i+1}/{num_positions} ({progress:.1%})', end='', flush=True)

        if show_progress:
            print()  # New line after progress bar
        
        return prf_cube

    def convolve_2d_scene(self, input_scene, sim_scene=None, **kwargs):
        """
        Simulate a two-dimensional scene through the current CGI configuration.

        This method retrieves the image data from `scene.background_scene` (or from
        `scene.twoD_image` when `scene` is a `SimulatedImage`), then convolves it
        with a grid of off-axis PSFs (also termed PRFs) either generated on the fly
        or supplied via `prf_cube`. The convolution is flux-conserving and the result
        is stored in the appropriate scene attribute (`twoD_image` or
        `background_scene`). The output units are currently assumed to be
        photoelectrons per second.

        Two modes of operation:
        1. **Pre-computed PRF mode**: Provide `prf_cube`, `prf_grid_radii`, `prf_grid_azimuths`
        2. **Generate-on-fly mode**: Provide grid parameters (iwa, owa, steps, etc.)

        Parameters
        ----------
        input_scene : Scene or SimulatedImage
            Scene object containing 2D image data to be convolved.
        sim_scene : SimulatedImage, optional
            Pre-initialized SimulatedImage object. If None, will be created.
            
        kwargs : keyword arguments
            
            Mode 1 parameters (from scene.twoD_scene_info):
            prf_cube_path : str
                Path to pre-computed PRF cube file.
            radii_lamD : array-like
                Radial grid values in λ/D corresponding to PRF cube.
            azimuths_deg : array-like
                Azimuthal grid values in degrees corresponding to PRF cube.
            disk_model_path : str
                Path to disk model file.

            Mode 2 parameters (from kwargs):
            iwa : float, optional
                Inner working angle in λ/D. Default is 0.
            owa : float, required
                Outer working angle in λ/D.
            inner_step : float, required
                Radial step size in λ/D for inner region (r ≤ iwa).
            mid_step : float, required
                Radial step size in λ/D for middle region (iwa < r < owa).
            outer_step : float, required
                Radial step size in λ/D for outer region (r ≥ owa).
            step_deg : float, required
                Azimuthal step size in degrees.
            max_radius : float, optional
                Maximum radius in λ/D for PRF generation. Default is max(15, 1.5 * owa).

        Returns
        -------
        Scene or SimulatedImage
            The input scene updated with its 2D image replaced by the convolved result.

        Raises
        ------
        ValueError
            If neither mode has complete parameters, or if both modes are specified.

        Examples
        --------
        Mode 1 - Using pre-computed PRF cube from scene:
        >>> # Scene must have twoD_scene_info with prf_cube_path, radii_lamD, etc.
        >>> result = optics.convolve_2d_scene(scene_with_prf_info)
        
        Mode 2 - Auto-generate PRF cube:
        >>> result = optics.convolve_2d_scene(scene,
        ...                                   iwa=3.0, owa=9.0,
        ...                                   inner_step=1.0, mid_step=1.5, 
        ...                                   outer_step=2.0, step_deg=90.0)
        """
        # Determine which mode to use based on provided kwargs
        if input_scene.twoD_scene_info['prf_cube_path'] is not None:
            has_prf_cube = True 
            has_grid_params = False
        else:
            has_prf_cube = False
            has_grid_params = 'iwa' in kwargs or 'owa' in kwargs

        # Extract optional parameter that applies to both modes
        use_bilinear_interpolation = kwargs.get('use_bilinear_interpolation', False)

        if has_prf_cube == has_grid_params:
            raise ValueError("Provide either prf_cube or grid parameters")
        
        # input disk model
        disk_model_data = fits.getdata(input_scene.twoD_scene_info['disk_model_path'])
        disk_model_norm = disk_model_data/np.nansum(disk_model_data, axis=(0,1)) # normalisation of the disk

        prf_cube_path = input_scene.twoD_scene_info['prf_cube_path'] 
        radii_lamD = input_scene.twoD_scene_info['radii_lamD'] # arrays 
        azimuths_deg = input_scene.twoD_scene_info['azimuths_deg'] # arrays 

        if has_prf_cube:
            # Mode 1: Pre-computed PRF cube: Apply convolution 
            conv2d = convolve_with_prfs(obj=disk_model_norm, prfs_array=fits.getdata(prf_cube_path), radii_lamD=radii_lamD , azimuths_deg=azimuths_deg, pix_scale_mas=PIXEL_SCALE_ARCSEC * 1e3, res_mas=self.res_mas, use_bilinear_interpolation=use_bilinear_interpolation)
        else:
            # Mode 2: Generate cube
            required = ['owa', 'inner_step', 'mid_step', 'outer_step', 'step_deg']
            missing = [k for k in required if k not in kwargs]
            if missing:
                raise ValueError(f"Missing parameters: {missing}")

            # Extract required parameters
            owa = kwargs['owa'] 
            inner_step = kwargs['inner_step']
            mid_step = kwargs['mid_step']
            outer_step = kwargs['outer_step']
            step_deg = kwargs['step_deg']
        
            # Extract optional parameters with defaults
            iwa = kwargs.get('iwa', 0) # default to 0 if not provided
            max_radius = kwargs.get('max_radius')
        
            radii_lamD = build_radial_grid(iwa, owa, inner_step, mid_step, outer_step, max_radius)
            azimuths_deg = build_azimuth_grid(step_deg)
            prf_cube = self.make_prf_cube(radii_lamD, azimuths_deg)
                
            # Apply convolution
            conv2d = convolve_with_prfs(disk_model_norm, prf_cube, radii_lamD, azimuths_deg, PIXEL_SCALE_ARCSEC * 1e3, self.res_mas, use_bilinear_interpolation)
        
        # TODO - simplify the steps below for the count and flux calculation
        # normalize to the given contrast
        # obs: flux is in unit of photons/s/cm^2/angstrom
        obs = Observation(input_scene.twoD_scene_spectrum, self.bp)
        counts = np.zeros((self.lam_um.shape[0]))
        for i in range(self.lam_um.shape[0]):
            dlam_um = self.lam_um[1]-self.lam_um[0]
            lam_um_l = (self.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
            lam_um_u = (self.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
            counts[i] = (self.polarizer_transmission * obs.countrate(area=self.area, waverange=[lam_um_l, lam_um_u])).value

        psf_area = np.pi*(self.res_mas/(PIXEL_SCALE_ARCSEC*1e3)/2)**2 # area of the PSF FWHM in the unit of pixel
        disk_region = (conv2d > 0.5*np.max(conv2d)).sum()  # number of pixs in the disk region (>=50% maximum disk flux) 
        conv2d *= np.sum(counts, axis=0) * disk_region/psf_area   # per resolution element

        # Update the scene with the convolved image
        if isinstance(input_scene, SimulatedImage):
            input_scene.twoD_image = conv2d
        elif isinstance(input_scene, Scene):
            input_scene.background_scene = conv2d
        else:
            raise ValueError(f"Unsupported scene type: {type(input_scene)}")

        if self.cgi_mode in ['spec', 'lowfs', 'excam_efield']:
            raise ValueError(f"The mode '{self.cgi_mode}' has not been implemented yet!")
        
        # Initialize SimulatedImage class to restore the output psf
        if sim_scene == None:
            sim_scene = scene.SimulatedImage(input_scene)
        
        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        sim_info = {'ref_flag': False,
                    'cgi_mode':self.cgi_mode,
                    'cor_type': self.optics_keywords['cor_type'],
                    'bandpass':self.bandpass_header,
                    'over_sampling_factor':self.oversampling_factor,
                    'return_oversample': self.return_oversample,
                    'output_dim': self.optics_keywords['output_dim'],
                    'nd_filter':self.nd}

        # Define specific keys from self.optics_keywords to include in the header
        keys_to_include_in_header = ['use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm','use_lyot_stop','use_field_stop','fsm_x_offset_mas','fsm_y_offset_mas']  # Specify keys to include
        subset = {key: self.optics_keywords[key] for key in keys_to_include_in_header if key in self.optics_keywords}
        sim_info.update(subset)
        sim_info['include_detector_noise'] = 'False'

        # Create the HDU object with the generated header information
        sim_scene.twoD_image = outputs.create_hdu(conv2d, sim_info=sim_info)

        return sim_scene

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
            - scene: A corgisim.scene.Scene object that contains the scene to be simulated.
            - sim_scene: A corgisim.SimulatedImage object to contains the simylated scene.
            - on_the_fly: A boolean that defines whether the PSFs should be generated on the fly.
        

        Returns: 
            - A 2D numpy array that contains the scene with the injected point sources. 
        '''

        # Extract point source spectra, positions, and polarization
        point_source_spectra = input_scene.off_axis_source_spectrum
        point_source_dra = input_scene.point_source_dra
        point_source_ddec = input_scene.point_source_ddec
        point_source_pol = input_scene.point_source_pol_state

        # Ensure all inputs are lists for uniform processing
        if not isinstance(point_source_spectra, list):
            point_source_spectra = [point_source_spectra]
        if not isinstance(point_source_dra, list):
            point_source_dra = [point_source_dra]
        if not isinstance(point_source_ddec, list):
            point_source_ddec = [point_source_ddec]

        # Ensure all lists have the same length
        if not (len(point_source_spectra) == len(point_source_dra) == len(point_source_ddec)):
            raise ValueError(
                f"Mismatch in input lengths: {len(point_source_spectra)} spectra, "
                f"{len(point_source_dra)} dRA positions, {len(point_source_ddec)} dDEC positions. "
                "Each point source must have a corresponding (dRA, dDEC) position.")

        if self.cgi_mode == 'excam':
            
            ##checks to see if point source is within FOV of coronagraph
            #FOV_range is indexed as follows - 0: hlc, 1: spc-spec, 2: spc-wide.
            #FOV_range Values correspond to the inner and outer radius of region of highest contrast and are in units of lambda/d
            FOV_range = [[3, 9.7], [3, 9.1], [5.9, 20.1]]
            if(self.cor_type.find('hlc') != -1):
                FOV_index = 0
            elif (self.cor_type.find('spec') != -1):
                FOV_index = 1
                #todo: Add conditions checking if point source is within azimuthal angle range once spectroscopy mode is implemented
            else:
                FOV_index = 2
            #Calculate point source separation from origin in units of lambda/D
            point_source_radius = np.sqrt(np.power(point_source_dra, 2) + np.power(point_source_ddec, 2)) * ((self.diam * 1e-2)/(self.lam0_um * 1e-6 * 206265000))
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
            
                self.optics_keywords_comp = self.optics_keywords.copy()
                ## convert companion sky coord to exacam coord, using roll angle
                point_source_dx, point_source_dy = skycoord_to_excamcoord(point_source_dra[j], point_source_ddec[j], self.roll_angle)

                self.optics_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                            'final_sampling_m': sampling_um_tem * 1e-6,
                                            'source_x_offset_mas': point_source_dx,
                                            'source_y_offset_mas': point_source_dy})
                
                if self.optics_keywords['polaxis'] == -10:
                    optics_keywords_comp_m10 = self.optics_keywords_comp.copy()
                    images_tem = self.generate_full_aberration_psf(optics_keywords_comp_m10)
                else: 
                    (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE= self.optics_keywords_comp ,QUIET=True)
                    images_tem = np.abs(fields)**2

                # Initialize the image array based on whether oversampling is returned
                images_shape = (self.nlam, grid_dim_out_tem, grid_dim_out_tem) if self.return_oversample else (self.nlam, self.grid_dim_out, self.grid_dim_out)
                images = np.zeros(images_shape, dtype=float)
                counts = np.zeros(self.nlam)

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
                    counts[i] = self.polarizer_transmission * obs_point_source[j].countrate(area=self.area, waverange=[lam_um_l, lam_um_u]).value

                # if wollaston is used, compute point source stokes vector and scale intensity accordingly
                # multiply input point source stokes vector by instrument mueller matrix, renormalize, then
                # multiply that by the 0/90/45/135 degree polarizer matrix of the CGI wollaston
                if self.prism in ['POL0', 'POL45']:
                    images_1 = np.zeros(images_shape, dtype=float)
                    images_2 = np.zeros(images_shape, dtype=float)
                    # transform point source stokes vector by instrument Mueller matrix
                    # renormalize since instrument Mueller matrix decreases total intensity, and that should already
                    # be accounted for in the proper model
                    source_pol_after_instrument = np.matmul(pol.get_instrument_mueller_matrix(self.lam_um), point_source_pol[j])
                    source_pol_after_instrument = source_pol_after_instrument / source_pol_after_instrument[0]
                    if (self.prism == 'POL0'):
                        source_pol_path_1 = np.matmul(pol.get_wollaston_mueller_matrix(0), source_pol_after_instrument)
                        source_pol_path_2 = np.matmul(pol.get_wollaston_mueller_matrix(90), source_pol_after_instrument)
                    else:
                        source_pol_path_1 = np.matmul(pol.get_wollaston_mueller_matrix(45), source_pol_after_instrument)
                        source_pol_path_2 = np.matmul(pol.get_wollaston_mueller_matrix(135), source_pol_after_instrument)
                    counts_after_wollaston = [counts * source_pol_path_1[0], counts * source_pol_path_2[0]]
                    images_1 = images * counts_after_wollaston[0][:, np.newaxis, np.newaxis]
                    images_2 = images * counts_after_wollaston[1][:, np.newaxis, np.newaxis]
                    # 3D datacube of point source by polarization state if wollaston is used
                    point_source_image.append(np.array([np.sum(images_1, axis=0), np.sum(images_2, axis=0)]))
                else:
                    images *= counts[:, np.newaxis, np.newaxis]
                    image = np.sum(images, axis=0)
                    # singular unpolarized image if no wollaston
                    point_source_image.append(image) 
        elif self.cgi_mode == 'spec':
            

            if self.slit != 'None':
                field_stop_array, field_stop_sampling_m = spec.get_slit_mask(self)
                self.optics_keywords['field_stop_array']=field_stop_array
                self.optics_keywords['field_stop_array_sampling_m']=field_stop_sampling_m
            else:
                self.optics_keywords['field_stop_array']=0
                self.optics_keywords['field_stop_array_sampling_m']=0

            # Compute the observed  spectrum for each off-axis source
            obs_point_source = [Observation(spectrum, self.bp) for spectrum in point_source_spectra]
            
            grid_dim_out_tem = self.grid_dim_out * self.oversampling_factor
            sampling_um_tem = self.sampling_um / self.oversampling_factor
            
            point_source_image = []
            for j in range(len(point_source_spectra )):
                self.optics_keywords_comp = self.optics_keywords.copy()
                ## convert companion sky coord to exacam coord, using roll angle
                point_source_dx, point_source_dy = skycoord_to_excamcoord(point_source_dra[j], point_source_ddec[j], self.roll_angle)
               
                self.optics_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                            'final_sampling_m': sampling_um_tem * 1e-6,
                                            'source_x_offset_mas': point_source_dx,
                                            'source_y_offset_mas': point_source_dy})
                if self.optics_keywords['polaxis'] == -10:
                    optics_keywords_comp_m10 = self.optics_keywords_comp.copy()
                    polaxis_params = [-2, -1, 1, 2]
                    images_pol = []
                    for polaxis in polaxis_params:
                        optics_keywords_comp_m10['polaxis'] = polaxis
                        (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=optics_keywords_comp_m10,QUIET=True)
                        images_pol.append(np.abs(fields) ** 2)
                    images_tem = np.array(sum(images_pol)) / 4
                else: 
                    (fields, sampling) = proper.prop_run_multi('roman_preflight', self.lam_um, 1024, PASSVALUE=self.optics_keywords_comp ,QUIET=True)
                    images_tem = np.abs(fields)**2

                # If a prism was selected, apply the dispersion model and overwrite the image cube and wavelength array.
                if self.prism != 'None': 
                    images_tem, dispersed_lam_um, disp_shift_lam0_x, disp_shift_lam0_y = spec.apply_prism(self, images_tem)
    
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
                        ## update the optics_keywords['output_dim'] baclk to non_oversample size
                        self.optics_keywords['output_dim'] = self.grid_dim_out


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
            sim_info[f'position_dra_mas_{i}'] = input_scene.point_source_dra[i]
            sim_info[f'position_ddec_mas_{i}'] = input_scene.point_source_ddec[i]

        # Third: global simulation settings
        if self.prism == 'POL0':
            polarization_basis = '0/90 degrees'
        elif self.prism == 'POL45':
            polarization_basis = '45/135 degrees'
        else:
            polarization_basis = 'None'
        sim_info['cgi_mode'] = self.cgi_mode
        sim_info['cor_type'] = self.optics_keywords.get('cor_type')
        sim_info['bandpass'] = self.bandpass_header
        sim_info['polarization_basis'] = polarization_basis
        sim_info['over_sampling_factor'] = self.oversampling_factor
        sim_info['return_oversample'] = self.return_oversample
        sim_info['output_dim'] = self.optics_keywords['output_dim'] 
        sim_info['nd_filter'] = self.nd
        sim_info['roll_angle'] = self.roll_angle
                            
                # Define specific keys from self.optics_keywords to include in the header            
        keys_to_include_in_header = [ 'use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                            'use_lyot_stop','use_field_stop','fsm_x_offset_mas','fsm_y_offset_mas','slit','prism',
                            'slit_x_offset_mas','slit_y_offset_mas','use_pupil_lens', 'use_lyot_stop', 'use_field_stop']  # Specify keys to include
        subset = {key: self.optics_keywords[key] for key in keys_to_include_in_header if key in self.optics_keywords}
        sim_info.update(subset)

        ## add sattelite spots info
        sim_info['SATSPOTS'] = self.SATSPOTS
        sim_info['includ_dectector_noise'] = 'False'
        # Create the HDU object with the generated header information

        sim_scene.point_source_image = outputs.create_hdu( np.sum(point_source_image,axis=0), sim_info =sim_info)

        return sim_scene

    def add_satspot(self,satspot_keywords):
        """
        Add satellite spots to deformable mirror (DM) settings.
        This function modifies the deformable mirror settings stored in `self.optics_keywords['dm1_v']` 
        by injecting satellite spots according to the provided `satspot_keywords`.

        Parameters:
        ----------
        satspot_keywords : dict
            Dictionary specifying the parameters needed to define and inject 
            satellite spots (sep_lamD, angle_deg, contrast, wavelength_m).

        Returns:
        -------
        dm1_cos_added : 2D ndarray
            Updated DM1 voltage map with satellite spots added.
       
        """

        # extract DM1
        proper_keywords = self.optics_keywords.copy()
        dm1_input = proper_keywords['dm1_v']

        # extract satspot_keywords
        num_pairs = satspot_keywords['num_pairs']
        sep_lamD = satspot_keywords['sep_lamD']
        angle_deg = satspot_keywords['angle_deg']
        contrast = satspot_keywords['contrast']
        wavelength_m = satspot_keywords['wavelength_m']

        dm1_cos_added = add_cos_pattern_dm(dm1_input,num_pairs,sep_lamD,angle_deg,contrast,wavelength_m)

        return dm1_cos_added

    def generate_full_aberration_psf(self, optics_keywords):
        '''
        Calls proper.prop_run_multi() with polaxis set to -2, 2, -1, and 1 in order to obtain fields with
        polarization aberrations describing -45->Y, 45->Y, -45->X, and 45->X, incoherently average those
        four fields to obtain an intensity image containing the full polarization aberration information

        Arguments:
            optics_keywords: A dictionary with the keywords that are used to set up the proper model
        
        Returns:
            images: 3D datacube containing the PSFs sampled at various wavelengths in the pass band
        '''    
        # polaxis values to be passed into the proper model
        polaxis_params = [-2, -1, 1, 2]
        images_pol = []

        # compute the field for each polaxis value
        for polaxis in polaxis_params:
            optics_keywords['polaxis'] = polaxis
            (fields, sampling) = proper.prop_run_multi('roman_preflight',  self.lam_um, 1024,PASSVALUE=optics_keywords,QUIET=True)
            images_pol.append(np.abs(fields) ** 2)

        # compute the average intensity
        images = np.array(sum(images_pol)) / 4
        return images
    
class CorgiDetector(): 
    
    def __init__(self ,emccd_keywords, photon_counting = True):
        '''
        Initialize the class with a dictionary that defines the EMCCD_DETECT input parameters. 

        Arguments: 
            - emccd_keywords: A dictionary with the keywords that are used to set up the emccd model
            - photon_counting: if use photon_counting mode, default is True
        '''
        if emccd_keywords is None:
            self.emccd_keywords = None
        else:
            self.emccd_keywords = emccd_keywords.copy()  # Store the keywords for later use
        #self.exptime = exptime ##expsoure time in second
        self.photon_counting = photon_counting


        self.emccd = self.define_EMCCD(emccd_keywords=self.emccd_keywords)
    

    def generate_detector_image(self, simulated_scene, exptime, full_frame= False, loc_x=512, loc_y=512):
        '''
        Function that generates a detector image from the input image, using emccd_detect. 

        The input_image probably has to be in electrons. 

        Arguments:
            - simulated_scene: a corgisim.scene.SimulatedScen object that contains the noise-free scene from CorgiOptics 
            - full_frame: if generated full_frame image in detetor
            - loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
            - loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
            - exptime: exptime in second

        Returns:
            - A corgisim.scene.SimulatedImage object that contains the detector image in the 
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
      
        if sim_info['polarization_basis'] == 'None':
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
        else:
            if full_frame:
                #images separated 7.5" or 344 pix on the detector (1 pix=0.0218")
                #0/90 degree images are placed on x-axis symmetric about the user defined location
                #45/135 degree images are placed on -45 degree axis symmetric about the user defined location
                if sim_info['polarization_basis'] == '0/90 degrees':
                    loc_x_from_center = 172
                    loc_y_from_center = 0
                else:
                    loc_x_from_center = 122
                    loc_y_from_center = 122
                if (img[0].shape[0] < 512) & (img[0].shape[1] < 512):
                    flux_map = self.place_scene_on_detector(img[0] , loc_x-loc_x_from_center, loc_y+loc_y_from_center) + self.place_scene_on_detector(img[1] , loc_x+loc_x_from_center, loc_y-loc_y_from_center)
                else:
                    raise ValueError("Polarimetry mode image dimensions cannot exceed 512x512 to ensure images do not go off detector.")
                Im_noisy = self.emccd.sim_full_frame(flux_map, exptime).astype(np.uint16)
            else:
                #currently runs sim_sub_frame twice for each image
                #add warning about subframes having different noises
                warnings.warn('Detector noise will be different for each sub frame in polarimetry mode. For accurate detector image with noise, please generate full frame image.')
                Im_noisy = np.array([self.emccd.sim_sub_frame(img[0], exptime).astype(np.uint16), self.emccd.sim_sub_frame(img[1], exptime).astype(np.uint16)])
            
        # Prepare additional information to be added as COMMENT headers in the primary HDU.
        # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
        sim_info['includ_dectector_noise'] = 'True'
        subset = {key: self.emccd_keywords_default[key] for key in self.emccd_keywords_default}
        sim_info.update(subset)
        
        # Create the HDU object with the generated header information
        if full_frame:
            sim_info['position_on_detector_x'] = loc_x
            sim_info['position_on_detector_y'] = loc_y
            
           
            ref_flag = outputs.str2bool(sim_info['ref_flag'])
            use_fpm = outputs.str2bool(sim_info['use_fpm'])
            use_pupil_lens = outputs.str2bool(sim_info['use_pupil_lens'])
            use_lyot_stop = outputs.str2bool(sim_info['use_lyot_stop'])
            use_field_stop = outputs.str2bool(sim_info['use_field_stop'])   
            
            header_info = {'EXPTIME': exptime,'EMGAIN_C':self.emccd_keywords_default['em_gain'],'PSFREF':ref_flag,
                           'PHTCNT':self.photon_counting,'KGAINPAR':self.emccd_keywords_default['e_per_dn'],'cor_type':sim_info['cor_type'], 'bandpass':sim_info['bandpass'],
                           'cgi_mode': sim_info['cgi_mode'], 'polaxis':sim_info['polaxis'],'use_fpm':use_fpm,'nd_filter':sim_info['nd_filter'], 'polarization_basis': sim_info['polarization_basis'],'SATSPOTS':sim_info['SATSPOTS'],
                           'use_pupil_lens':use_pupil_lens,'use_lyot_stop':use_lyot_stop, 'use_field_stop':use_field_stop, 'ROLL': float(sim_info['roll_angle']),
                           'EACQ_ROW': loc_x, 'EACQ_COL': loc_y}
            if 'fsm_x_offset_mas' in sim_info:
                header_info['FSMX'] = float(sim_info['fsm_x_offset_mas'])
            if 'fsm_y_offset_mas' in sim_info:
                header_info['FSMY'] = float(sim_info['fsm_y_offset_mas'])

            header_info['slit'] = sim_info.get('slit', 'None')
            header_info['prism'] = sim_info.get('prism', 'None')
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
            - sub_frame (numpy.ndarray): A 2D array representing the simulated scene to be placed on the detector.
            - loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted.
            - loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted.

        Returns:
            - numpy.ndarray: A 1024x1024 2D array (detector frame) with the sub_frame placed at the specified center location and the remaining areas padded with zeros.

        Raises:
            - ValueError: If the sub_frame, when placed at the specified location, exceeds the bounds of the 1024x1024 detector array or if negative indices result.
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
        #full_frame[x_start:x_end, y_start:y_end] = sub_frame
        full_frame[y_start:y_end, x_start:x_end] = sub_frame

        return full_frame



    def define_EMCCD(self, emccd_keywords=None ):

        '''
        Create and configure an EMCCD detector object with optional CTI simulation.

        This method initializes an EMCCD detector object using the provided parameters.
        Optionally, if CTI (Charge Transfer Inefficiency) simulation is required, it updates
        the detector object using a trap model.

        Args:
            - # default values match requirements, except QE, which is year 0 curve (already accounted for in counts)
            - em_gain (float, optional): EM gain, default 1000.
            - full_well_image (float, optional): image full well; 50K is requirement, 60K is CBE
            - full_well_serial (float, optional): full well for serial register; 90K is requirement, 100K is CBE
            - dark_rate (float, optional): Dark current rate, e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
            - cic_noise (float, optional): Clock-induced charge noise, e-/pix/frame; Defaults to 0.01.
            - read_noise (float, optional): Read noise, e-/pix/frame; 125 is requirement, 100 is CBE
            - bias (int, optional): Bias level (in digital numbers). Defaults to 0.
            - qe (float): Quantum efficiency, set to 1 here, because already counted in counts
            - cr_rate (int, optional): Cosmic ray event rate, hits/cm^2/s (0 for none, 5 for L2) 
            - pixel_pitch (float, optional): Pixel pitch (in meters). Defaults to 13e-6.
            - e_per_dn (float, optional): post-multiplied electrons per data unit
            - numel_gain_register (int, optional): Number of elements in the gain register. Defaults to 604.
            - nbits (int, optional): Number of bits in the analog-to-digital converter. Defaults to 14.
            - use_traps (bool, optional): Flag indicating whether to simulate CTI effects using trap models. Defaults to False.
            - date4traps (float, optional): Decimal year of observation; only applicable if `use_traps` is True. Defaults to 2028.0.

        Returns:
            - emccd (EMCCDDetectBase): A configured EMCCD detector object. If `use_traps` is True, the detector's CTI is updated using the corresponding trap model.
        
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


def skycoord_to_excamcoord(dra, ddec, roll_angle):
    """Convert sky coordinates to EXCAM coordinates. These are both relative astrometry of a companion relative to host star (or central of the frame)

    Args:
        dra (float): The right ascension offset in mas.
        ddec (float): The declination offset in mas.
        roll_angle (float): The roll angle in degrees.

    Returns:
        dx (float): The converted EXCAM x-coordinate offset in mas.
        dy (float): The converted EXCAM y-coordinate offset in mas.
    """
    # Apply roll angle rotation
    # Because we rotate the *companion coords*, use the opposite sense: θ_comp = -roll_angle.
    theta_comp = np.deg2rad(-1 * roll_angle)

    dx = dra * np.cos(theta_comp) - ddec * np.sin(theta_comp)
    dy = dra * np.sin(theta_comp) + ddec * np.cos(theta_comp)
    return dx, dy
