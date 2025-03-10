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



#@pytest.mark.parametrize("interp_method", ['linear', 'cubic'])
def run_sim():
    print('testrun')

    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    host_star_properties = {'Vmag': 0.6, 'spectral_type': 'A0V','magtype': 'vegamag'}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    sp_rn =  base_scene.stellar_spectrum

    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    proper_keywords ={'cor_type': cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':101,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True, integrate_pixels=True)
    
    print(optics.lam_um[1])
    bp = optics.setup_bandpass(cgi_mode, bandpass, 0)
  
    #print(bp.avgwave(), bp.tlambda(), bp.efficiency(), bp.tpeak(),bp.wpeak())
    #print(bp.equivwidth())
    
    #sp_rn.plot()
    bp.plot()
 
    plt.show()
  
  
    
if __name__ == '__main__':
    run_sim()