from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim

'''
Based on test_on_axis_star, checks to see if the spc coronagraph types are implemented correctly
'''

def test_spc_mode():
    print('testrun')
    
    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '2F'
    bandpass_cgisim = '2'
    cor_type = 'spc-spec_band2'
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['1e-9']       
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    proper_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
   
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, proper_keywords=proper_keywords, if_quiet=True, integrate_pixels=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    image_star_corgi = sim_scene.host_star_image.data
    print('Final_intensity_get:', np.sum(image_star_corgi, dtype = np.float64))
    #print(sim_scene.host_star_image[1].header)
    #print(sim_scene.host_star_image[0].header)

    ########################  simulate using cgisim
    polaxis_cgisim = -10
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    image_star_cgi, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )

    print(a0_counts, np.sum(image_star_cgi, dtype = np.float64))

    #check to see that the corgisim output matches the cgisim output within a 0.5% tolerance
    assert image_star_corgi == pytest.approx(image_star_cgi, rel=0.5)

if __name__ == '__main__':
    #run_sim()
    test_spc_mode()





