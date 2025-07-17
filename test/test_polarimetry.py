from corgisim import scene
from corgisim import instrument
import numpy as np
import proper
import roman_preflight_proper
import cgisim
import pytest

'''
Test file for polarimetry mode, currently just tests 0°/90° polarization, will be updated
once 45°/135° polarized speckle field is implemented correctly
'''

def test_polarimetry():
    print('This test checks that the images from polarimetry mode is generated correctly')

    #define host star properties
    Vmag = 8
    sptype = 'G0V'
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}

    #define companion properties, including polarization
    mag_companion = 25
    companion_x_pos = 148
    companion_y_pos = 148
    companion_pol = np.array([1, 0.3, 0.1, 0])
    point_source_info = [{'Vmag': mag_companion, 'magtype': 'vegamag','position_x':companion_x_pos , 'position_y':companion_y_pos, 'pol_state': companion_pol}]

    base_scene = scene.Scene(host_star_properties, point_source_info)

    #define instrument properties, set instrument to use 0° wollaston to image 0° and 90° linear polarization intensities
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    bandpass_cgisim = '1'
    cor_type = 'hlc'
    output_dim = 201
    wollaston_prism = 1
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_ni_3e-8_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_ni_3e-8_dm2_v.fits' )

    proper_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':output_dim,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, wollaston_prism=wollaston_prism, proper_keywords=proper_keywords, if_quiet=True, integrate_pixels=True)

    #simulate using corgisim
    sim_scene = optics.get_host_star_psf_polarized(base_scene)
    image_star_corgi_x = sim_scene.host_star_image[0].data
    image_star_corgi_y = sim_scene.host_star_image[1].data
    sim_scene = optics.inject_point_sources_polarized(base_scene, sim_scene)
    image_comp_corgi_x = sim_scene.point_source_image[0].data
    image_comp_corgi_y = sim_scene.point_source_image[1].data

    #simulate using cgisim
    polaxis_cgisim_x = -5
    polaxis_cgisim_y = -6
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    image_star_cgi_x, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim_x, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )
    image_star_cgi_y, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim_y, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )
    
    #check similarity of host star and speckle field image
    assert image_star_corgi_x  == pytest.approx(image_star_cgi_x, rel=0.5)
    assert image_star_corgi_y  == pytest.approx(image_star_cgi_y, rel=0.5)

    #since cgisim does not have the option to generate polarized point sources, this will instead check
    #to see if the sum of orthogonal polarized intensities add up to the total unpolarized intensity for the point source
    sim_scene_unpol = optics.inject_point_sources(base_scene)
    image_comp_unpol = sim_scene_unpol.point_source_image.data
    image_comp_pol_combined = image_comp_corgi_x + image_comp_corgi_y
    assert image_comp_unpol == pytest.approx(image_comp_pol_combined, rel=0.1)

if __name__ == '__main__':
    test_polarimetry()