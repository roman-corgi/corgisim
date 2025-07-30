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
    bandpass_corgisim = '4F'
    bandpass_cgisim = '4'
    cor_type = 'spc-wide'

    #Define companion properties
    #Add two companions, one in FOV, one out of FOV to test warning message
    mag_companion = [25, 25]
    companion_x_pos = [740, 1250]
    companion_y_pos = [740, 1250]
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x': companion_x_pos[0], 'position_y': companion_y_pos[0]},
                         {'Vmag': mag_companion[1], 'magtype': 'vegamag','position_x': companion_x_pos[1], 'position_y': companion_y_pos[1]}]
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)
    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['2e-8']       
    rootname = 'spc-wide_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
   
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True, integrate_pixels=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    image_star_corgi = sim_scene.host_star_image.data
    #check warning is there for out of FOV source
    with pytest.warns(UserWarning):
        sim_scene = optics.inject_point_sources(base_scene, sim_scene)
    image_comp_corgi = sim_scene.point_source_image.data 

    #### simulate using cgisim
    polaxis_cgisim = -10
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    image_star_cgi, a0_counts = cgisim.rcgisim(cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum = sptype.lower(), star_vmag = Vmag)
    
    image_comp = []
    for i in range(len(mag_companion)):
        params['source_x_offset_mas'] = companion_x_pos[i]
        params['source_y_offset_mas'] = companion_y_pos[i]
        comp_sim_allpol, comp_counts = cgisim.rcgisim(cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum = sptype.lower(), star_vmag = mag_companion[i])
        image_comp.append(comp_sim_allpol)
        a0_counts = a0_counts + comp_counts
    image_comp_cgi = np.sum(image_comp,axis=0) 

    #check to see that the corgisim output matches the cgisim output within a 0.5% tolerance
    assert image_star_corgi == pytest.approx(image_star_cgi, rel=0.5)
    assert image_comp_corgi == pytest.approx(image_comp_cgi, rel=0.5)

if __name__ == '__main__':
    #run_sim()
    test_spc_mode()





