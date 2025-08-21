from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim

#######################
### Set up a scene. ###
#######################

def test_on_axis_star_on_detector_subframe():
    print('testrun')
    
    # Test scene creation with None host star 
    base_scene = scene.Scene()
    assert isinstance(base_scene, scene.Scene)

    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    Vmag = 5
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
   
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True, integrate_pixels=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    image = sim_scene.host_star_image.data
    
    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 30
    detector = instrument.CorgiDetector( emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene,exptime)
    image2 = sim_scene.image_on_detector.data


    print('Final_intensity_get:', np.sum(image, dtype = np.float64))
    

    #print(sim_scene.host_star_image[1].header)

    ########################  simulate using cgisim
    polaxis_cgisim = -10
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    a0_sim_allpol, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )
    print(a0_counts, np.sum(a0_sim_allpol, dtype = np.float64))


    ################################
    a0_sim_allpol_ccd, a0_counts_ccd = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, ccd={'gain':gain,'exptime':exptime},
        star_spectrum=sptype.lower(), star_vmag=Vmag )

    # Use pytest.approx to check similarity within a tolerance
    assert  image  == pytest.approx(a0_sim_allpol, rel=0.5)
    print("noise free image Pass")


    

    

if __name__ == '__main__':
    #run_sim()
    test_on_axis_star_on_detector_subframe()





