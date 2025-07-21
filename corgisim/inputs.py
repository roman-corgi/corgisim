from corgisim import scene, instrument
import xml.etree.ElementTree as ET
from  xml.etree.ElementTree import ParseError
from astropy.io import fits
from corgidrp import mocks
import numpy as np
import roman_preflight_proper as rp
import types
class Input():
    """
    A class that holds all the information necessary for a simulation.

    Input can be created without arguments (default values will be used), with individual arguments or with dictionnaries for host star, 
    proper keywords and emccd keywords 

    Attribute Access:
        - All attributes are made read-only after initialization.
        - Accessing an attribute (e.g., `input.source_x_offset`) will retrieve the corresponding 
          internal value (e.g., `_source_x_offset`).

    Raises:
        AttributeError: If an attempt is made to modify any attribute after initialization.
    
    """

    def __init__(self, **kwargs ):
        #Defaults dictionnaries

        # All possible proper keywords, as defined in roman_proper_preflight
        proper_keywords_default = { 'use_cvs' : 0,                 # use CVS instead of telescope? (1=yes, 0=no)
                                    'cvs_stop_x_shift_m' : 0,      # shift of CVS entrance pupil mask in meters
                                    'cvs_stop_y_shift_m' : 0,
                                    'cvs_stop_z_shift_m' : 0,     # shift of CVS entrance pupil mask along optical axis (+ is downstream)
                                    'cvs_stop_rotation_deg' : 0,   # rotation of CVS entrance pupil mask in degrees
                                    'small_spc_grid' : 0,          # set to 1 to use 500 pix across pupil, else 1000 (baseline SPCs only)
                                    'pupil_array' : 0,             # 2D array containing pupil pattern (overrides default)
                                    'pupil_mask_array' : 0,        # 2D array containing SPC pupil mask pattern (overrides default)
                                    'fpm_array' : 0,               # 2D array containing FPM mask pattern (overrides default)
                                    'fpm_mask' : 0,                # 2D array where 1=FPM pattern defined, 0=substrate
                                    'lyot_stop_array' : 0,         # 2D array containing Lyot stop mask pattern (overrides default)
                                    'field_stop_array' : 0,        # 2D array containing field stop mask pattern (overrides default)

                                    'cor_type' : 'hlc',            # coronagraph type ('hlc', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'none')
                                    'source_x_offset_mas' : 0,     # source offset in mas (tilt applied at primary)
                                    'source_y_offset_mas' : 0,                
                                    'source_x_offset' : 0,   # source offset in lambda0_m/D radians (tilt applied at primary)
                                    'source_y_offset' : 0,          
                                    'cvs_source_z_offset_m' : 0,             # additional distance between CVS source and next optic, in meters
                              
      'cvs_jitter_mirror_x_offset_mas' : 0,      # source offset in milliarcsec (tilt applied at CVS jitter mirror)
                                    'cvs_jitter_mirror_y_offset_mas' : 0,      # 
                                    'cvs_jitter_mirror_x_offset' : 0,          # source offset in lambda0_m/D radians (tilt applied at CVS jitter mirror) 
                                    'cvs_jitter_mirror_y_offset' : 0,
                                    'polaxis' : 0,                 # polarization axis aberrations: 
                                                                #    -2 : -45d in, Y out 
                                                                #    -1 : -45d in, X out 
                                                                #     1 : +45d in, X out 
                                                                #     2 : +45d in, Y out 
                                                                #     5 : mean of modes -1 & +1 (X channel polarizer)
                                                                #     6 : mean of modes -2 & +2 (Y channel polarizer)
                                                                #    10 : mean of all modes (no polarization filtering)
                                    'use_errors' : 1,              # use optical surface phase errors? 1 or 0 
                                    'zindex' : np.array([0,0]),    # array of Zernike polynomial indices
                                    'zval_m' : np.array([0,0]),    # array of Zernike coefficients (meters RMS WFE)
                                    'sm_despace_m' : 0,            # secondary mirror despace (meters) 
                                    'use_pupil_defocus' : 1,       # include pupil defocus
                                    'use_aperture' : 0,            # use apertures on all optics? 1 or 0
                                    'cgi_x_shift_pupdiam' : 0,     # X,Y shear of wavefront at FSM (bulk displacement of CGI); normalized relative to pupil diameter
                                    'cgi_y_shift_pupdiam' : 0,          
                                    'cgi_x_shift_m' : 0,           # X,Y shear of wavefront at FSM (bulk displacement of CGI) in meters
                                    'cgi_y_shift_m' : 0,          
                                #    'end_at_fsm' : 0,              # end propagation after propagating to FSM (no FSM errors)
                                    'fsm_x_offset_mas' : 0,       # offset in focal plane caused by tilt of FSM in mas
                                    'fsm_y_offset_mas' : 0,         
                                    'fsm_x_offset' : 0,            # offset in focal plane caused by tilt of FSM in lambda0/D
                                    'fsm_y_offset' : 0,            
                                    'fcm_z_shift_m' : 0,          # offset (meters) of focus correction mirror (+ increases path length)
                                    'use_dm1' : 0,                 # use DM1? 1 or 0
                                    'use_dm2' : 0,                 # use DM2? 1 or 0
                                #   'dm_version' : rp.dm_version  # string, DM version
                                    'dm_v_quant' : 110.0 / 2.**16, # DM DAC voltage quantization resolution 
                                    'dm_sampling_m' : 0.9906e-3,   # actuator spacing in meters
                                    'dm_temp_c' : 26.0,
                                    'dm1_v' : np.zeros((48,48)),
                                    'dm1_xc_act' : 23.5,           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) : 1st actuator center
                                    'dm1_yc_act' : 23.5,              
                                    'dm1_xtilt_deg' : 9.65,        # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
                                    'dm1_ytilt_deg' : 0, 
                                    'dm1_ztilt_deg' : 0, 
                                    'dm2_v' : np.zeros((48,48)),
                                    'dm2_xc_act' : 23.5 - 0.1,     # for 48x48 DM, wavefront centered at actuator intersections: (0,0) : 1st actuator center
                                    'dm2_yc_act' : 23.5,               
                                    'dm2_xtilt_deg' : 9.65, 
                                    'dm2_ytilt_deg' : 0,
                                    'dm2_ztilt_deg' : 0,
                                    'spam_x_shift_pupdiam' : 0,    # X,Y shift of wavefront at SPAM; normalized relative to pupil diameter
                                    'spam_y_shift_pupdiam' : 0,
                                    'spam_x_shift_m' : 0,          # X,Y shift of wavefront at SPAM in meters
                                    'spam_y_shift_m' : 0,
                                    'use_pupil_mask' : 1,          # SPC only: use SPC pupil mask (0 or 1)
                                    'mask_x_shift_pupdiam' : 0,    # X,Y shear of shaped pupil mask; normalized relative to pupil diameter
                                    'mask_y_shift_pupdiam' : 0,          
                                    'mask_x_shift_m' : 0,          # X,Y shear of shaped pupil mask in meters
                                    'mask_y_shift_m' : 0,          
                                    'mask_rotation_deg' : 0,
                                    'use_fpm' : 1,                 # use occulter? 1 or 0
                                    'fpm_x_offset' : 0,            # FPM x,y offset in lambda0/D
                                    'fpm_y_offset' : 0,
                                    'fpm_x_offset_m' : 0,          # FPM x,y offset in meters
                                    'fpm_y_offset_m' : 0,
                                    'fpm_z_shift_m' : 0,           # occulter offset in meters along optical axis (+ : away from prior optics)
                                    'pinhole_diam_m' : 0,          # FPM pinhole diameter in meters
                                #    'end_at_fpm_exit_pupil' : 0,   # return field at FPM exit pupil?
                                    'use_lyot_stop' : 1,           # use Lyot stop? 1 or 0
                                    'lyot_x_shift_pupdiam' : 0,    # X,Y shear of Lyot stop mask; normalized relative to pupil diameter
                                    'lyot_y_shift_pupdiam' : 0,  
                                    'lyot_x_shift_m' : 0,          # X,Y shear of Lyot stop mask in meters
                                    'lyot_y_shift_m' : 0,  
                                    'lyot_rotation_deg' : 0,
                                    'use_field_stop' : 1,          # use field stop (HLC)? 1 or 0
                                    'field_stop_radius_lam0' : 0,  # field stop radius in lambda0/D
                                    'field_stop_x_offset' : 0,     # field stop offset in lambda0/D
                                    'field_stop_y_offset' : 0,
                                    'field_stop_x_offset_m' : 0,   # field stop offset in meters
                                    'field_stop_y_offset_m' : 0,
                                    'use_pupil_lens' : 0,          # use pupil imaging lens? 0 or 1
                                    'use_defocus_lens' : 0,        # use defocusing lens? Options are 1, 2, 3, 4
                                    'end_at_exit_pupil' : 0,       # return exit pupil corresponding to final image plane
                                    'excam_despace_m' : 0,         # increase in spacing between final optic and detector
                                #    'final_sampling_m' : 0,        # final sampling in meters (overrides final_sampling_lam0)
                                #    'final_sampling_lam0' : 0,     # final sampling in lambda0/D
                                    'output_dim' : 201,    # dimension of output in pixels (overrides output_dim0)
                                    'image_x_offset_m' : 0,        # shift of image at detector plane in meters
                                    'image_y_offset_m' : 0}

        emccd_keywords_default = {'full_well_serial': 100000.0,         # full well for serial register; 90K is requirement, 100K is CBE
                                  'full_well_image': 60000.0,                 # image full well; 50K is requirement, 60K is CBE
                                  'dark_rate': 0.00056,                  # e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
                                  'cic_noise': 0.01,                    # e-/pix/frame; 0.1 is requirement, 0.01 is CBE
                                  'read_noise': 100.0,                  # e-/pix/frame; 125 is requirement, 100 is CBE
                                  'cr_rate': 0,                         # hits/cm^2/s (0 for none, 5 for L2) 
                                  'em_gain': 1000.0 ,                      # EM gain
                                  'bias': 0,
                                  'pixel_pitch': 13e-6 ,                # detector pixel size in meters
                                  'apply_smear': True ,                 # (LOWFS only) Apply fast readout smear?  
                                  'e_per_dn':1.0  ,                    # post-multiplied electrons per data unit
                                  'nbits': 14  ,                        # ADC bits
                                  'numel_gain_register': 604,           # Number of gain register elements 
                                  'use_traps': False,                    # include CTI impact of traps
                                  'date4traps': 2028.0}                        # decimal year of observation}      

        host_star_properties_default = {'spectral_type' : 'G0V',
                                        'Vmag' :  5  , 
                                        'magtype' : 'vegamag'}


        # To make attributes 'read-only'
        for key in list(proper_keywords_default):
            proper_keywords_default['_'+key] = proper_keywords_default.pop(key)       
        vars(self).update(proper_keywords_default)

        for key in list(emccd_keywords_default):
            emccd_keywords_default['_'+key] = emccd_keywords_default.pop(key) 
        vars(self).update(emccd_keywords_default)

        for key in list(host_star_properties_default):
            host_star_properties_default['_'+key] = host_star_properties_default.pop(key) 
        vars(self).update(host_star_properties_default)

        for key in list(proper_keywords_default):
            proper_keywords_default[key[1:]] = proper_keywords_default.pop(key)    

        for key in list(emccd_keywords_default):
            emccd_keywords_default[key[1:]] = emccd_keywords_default.pop(key) 

        for key in list(host_star_properties_default):
            host_star_properties_default[key[1:]] = host_star_properties_default.pop(key)


        # Create mutable copies for internal use during initialization
        self.__mutable_proper_keywords = proper_keywords_default.copy()
        self.__mutable_emccd_keywords = emccd_keywords_default.copy()
        self.__mutable_host_star_properties = host_star_properties_default.copy()

        # Remaining Inputs for scene 
        self._point_source_info = None
        self._twoD_scene_hdu = None

        # Remaining inputs for Optics
        self._cgi_mode = 'excam'
        self._bandpass = '1b'
        self._diam =  236.3114
        
        self._oversampling_factor = 7
        self._return_oversample = False
        self._nd = 0

        # If Input if made based on cpgs_file, file path to the file
        self._cpgs_file = None


        # Update the default input with kwargs

        # Update dictionaries
        if 'proper_keywords' in kwargs : 
            new_proper_keywords = kwargs['proper_keywords'].copy()
            self.__mutable_proper_keywords.update(new_proper_keywords)
            # Put dictionary value into attributes
            for key in list(kwargs['proper_keywords']):
                new_proper_keywords['_'+key] = new_proper_keywords.pop(key)    
            vars(self).update(new_proper_keywords)
                    
            del kwargs['proper_keywords']

        if 'emccd_keywords' in kwargs : 
            new_emccd_keywords = kwargs['emccd_keywords'].copy()
            self.__mutable_emccd_keywords.update(kwargs['emccd_keywords'])
            # Put dictionary value into attributes
            for key in list(kwargs['emccd_keywords']):
                new_emccd_keywords['_'+key] = new_emccd_keywords.pop(key)    
            vars(self).update(new_emccd_keywords)

            del kwargs['emccd_keywords']

        if 'host_star_properties' in kwargs : 
            new_host_star_properties = kwargs['host_star_properties'].copy()
            self.__mutable_host_star_properties.update(kwargs['host_star_properties'])
            # Put dictionary value into attributes
            for key in list(kwargs['host_star_properties']):
                new_host_star_properties['_'+key] = new_host_star_properties.pop(key)    
            vars(self).update(new_host_star_properties)
            del kwargs['host_star_properties']

        # Update the rest of the values
        for key in list(kwargs):
            kwargs['_'+key] = kwargs.pop(key) 
        vars(self).update(kwargs)
        
        # Make sure there are no discrepancies between dictionnaries and individual values
        for key, val in kwargs.items():
            if key[1:] in self.__mutable_proper_keywords:
                self.__mutable_proper_keywords[key[1:]] = val
            elif key[1:] in self.__mutable_emccd_keywords:
                self.__mutable_emccd_keywords[key[1:]] = val
            elif key[1:] in self.__mutable_host_star_properties:
                self.__mutable_host_star_properties[key[1:]] = val

        # Now, create the immutable views of the dictionaries
        self._proper_keywords = types.MappingProxyType(self.__mutable_proper_keywords)
        self._emccd_keywords = types.MappingProxyType(self.__mutable_emccd_keywords)
        self._host_star_properties = types.MappingProxyType(self.__mutable_host_star_properties)

        self._initialized = True
        
    #Make all attributes read-only
    def __setattr__(self, attr, value):
        if '_initialized' not in self.__dict__: #If not initialized, 
            super().__setattr__(attr, value)
        else :
            raise AttributeError(f'Cannot set attribute {attr}')

    def __getattr__(self, attr):
        name ='_'+attr
        if name in  self.__dict__ :
            return self.__dict__[name]



def load_cpgs_data(filepath, return_input=False):
    """
    Creates a scene and optics based on the content of a cpgs file

    :param filepath: path to the input file
    :type filepath: string
    :param return_input: if True, returns an Input object
    .
    :return: a scene list and optics if return_input is False; an Input if return_input is True

    """
    # Parse the file 
    try: 
        tree = ET.parse(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath} does not exists.")
    except ParseError: 
        raise TypeError(f"{filepath} is not an xml file.") 


    root = tree.getroot()
    
    # Create a host star and scene for each target
    target_list = root.find('target_list')
    host_star_properties_list = []
    scene_list = []
    for target in target_list.iter('target'):
        Vmag = float(target.find('v_mag').text)
        # Luminosity currently not an attribute of the target in cpgs file
        sptype = target.find('spec_type').text + target.find('sub_type').text 
        # MAG Type not currently not an attribute of the target in cpgs file, using vegamag by default
        magtype = "vegamag"
        host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':magtype}
        base_scene = scene.Scene(host_star_properties)

        host_star_properties_list.append(host_star_properties)
        scene_list.append(base_scene)
        if (target.find('target_id').text == '1'):
            host_star_properties_target =host_star_properties
    # Create optics and detector for every visit

    cpgs_input = root.find('cpgs_input')


    # For now, filter can only take two values in cpgs:
    #   1 <-> Band 1 (575 nm)
    #   2 <-> Band 4 (825 nm)
    filter_dict = {'1':'1', '2':'4'}
    filt = cpgs_input.find('filter').text
    bandpass = filter_dict[filt]
    # For now, coronagraph_mask can only take one value in cpgs:
    #   1 <-> hlc
    
    coronograph_mask = cpgs_input.find('coronagraph_mask').text

    match bandpass:
        case '1':
            if coronograph_mask == '1':
                cor_type = 'hlc_band1'
            else:
                raise NotImplementedError("HLC is the only implemented mode")
        case '4':
            if coronograph_mask == '1':
                cor_type = 'hlc_band4'
            else:
                raise NotImplementedError("HLC is the only implemented mode")                

        case _:
            raise NotImplementedError("Only Band 1 and Band 4 have been implemented.")                

    # Polarization
    # Polarimetry is not yet implemented, but the structure is left as to simplify future implementation
    if cpgs_input.find('with_polarization').text == '1' : 
        match cpgs_input.find('wollaston').text :
         # 0/90 deg
            case '1' :
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")       

            # 45/135 deg
            case '2' :
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")       

            case _: 
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")
    else :
        polaxis = 0         

    # Only mode implemented for now
    cgi_mode = 'excam'

    proper_keywords ={'cor_type':cor_type, 'polaxis':polaxis, 'output_dim':201}

    if return_input == True :
        input = Input(cgi_mode=cgi_mode, bandpass=bandpass, proper_keywords=proper_keywords,host_star_properties=host_star_properties_target, cpgs_file = filepath) 
        return input
    else:
        optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True)
        return scene_list, optics



