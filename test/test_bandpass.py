from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
from synphot import units, SourceSpectrum, SpectralElement, Observation
import re


#@pytest.mark.parametrize("interp_method", ['linear', 'cubic'])
def test_bandpass():
    print('testrun')

    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    host_star_properties = {'Vmag': 0.6, 'spectral_type': 'A0V','magtype': 'vegamag'}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    sp_rn =  base_scene.stellar_spectrum

    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1A'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type': cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':101,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    
### test the disable use of ND filter with FPM/FSM teh same time
def test_nd_filter_FPM_conflicts():
    host_star_properties = {'Vmag': 0.6, 'spectral_type': 'A0V','magtype': 'vegamag'}
    
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1A'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type': cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':51,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,\
                        'use_field_stop':1, 'nd':1 }
    msg1 = "FPM cannot be used with ND filter 1 (225@FPAM) or ND filter 2 (475@FPAM), because they occupy the same position in the optical path."
    with pytest.raises(ValueError, match=re.escape(msg1)):
        instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    
    msg2 = "Lyot stop cannot be used with ND filter 3 (475@FSAM), because they occupy the same position in the optical path."
    optics_keywords2 ={'cor_type': cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':51,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,\
                        'use_field_stop':1, 'nd':3, }
    with pytest.raises(ValueError, match=re.escape(msg2)):
        instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords2, if_quiet=True)
  
  
    
if __name__ == '__main__':
    test_bandpass()
    test_nd_filter_FPM_conflicts()