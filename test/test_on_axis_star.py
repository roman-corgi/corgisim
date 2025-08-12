from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import copy 

#######################
### Set up a scene. ###
#######################

def test_on_axis_star():
    print('testrun')
    
    #Define the host star properties
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    bandpass_cgisim = '1'
    cor_type = 'hlc_band1'
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_keywords_copy = copy.deepcopy(optics_keywords)
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True, integrate_pixels=True)

    sim_scene = optics.get_host_star_psf(base_scene)
    
    for key, value in optics_keywords.items():
        if key not in optics_keywords_copy.keys():
            pytest.fail(f'optics keywords have been changed')
        else:
            if isinstance(optics_keywords[key], np.ndarray) :
                assert (optics_keywords[key] == optics_keywords_copy[key]).all
            else:
                assert optics_keywords[key] == optics_keywords_copy[key]

    image = sim_scene.host_star_image.data
    print('Final_intensity_get:', np.sum(image, dtype = np.float64))
 
 

    ########################  simulate using cgisim
    polaxis_cgisim = -10
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    a0_sim_allpol, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )

    print(a0_counts, np.sum(a0_sim_allpol, dtype = np.float64))

if __name__ == '__main__':
    #run_sim()
    test_on_axis_star()





