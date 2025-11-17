from corgisim import scene, instrument, pol
import numpy as np
import proper
import roman_preflight_proper
import pytest

'''
Test file to check that polarized images are generated correctly
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

    #define instrument properties
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc'
    output_dim = 51
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_ni_3e-8_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_ni_3e-8_dm2_v.fits' )
    

    #Generate 0/90 image pair
    optics_keywords_0_90 = {'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':output_dim, 'prism':'POL0',\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_0_90 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_0_90, if_quiet=True)
    sim_scene_0_90 = optics_0_90.get_host_star_psf(base_scene)
    image_star_corgi_x = sim_scene_0_90.host_star_image.data[0]
    image_star_corgi_y = sim_scene_0_90.host_star_image.data[1]
    sim_scene_0_90 = optics_0_90.inject_point_sources(base_scene, sim_scene_0_90)
    image_comp_corgi_x = sim_scene_0_90.point_source_image.data[0]
    image_comp_corgi_y = sim_scene_0_90.point_source_image.data[1]

    #Generate 45/135 image pair
    optics_keywords_45_135 = {'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':output_dim, 'prism':'POL45',\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_45_135 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_45_135, if_quiet=True)
    sim_scene_45_135 = optics_45_135.get_host_star_psf(base_scene)
    image_star_corgi_45 = sim_scene_45_135.host_star_image.data[0]
    image_star_corgi_135 = sim_scene_45_135.host_star_image.data[1]
    sim_scene_45_135 = optics_45_135.inject_point_sources(base_scene, sim_scene_0_90)
    image_comp_corgi_45 = sim_scene_45_135.point_source_image.data[0]
    image_comp_corgi_135 = sim_scene_45_135.point_source_image.data[1]

    #Generate unpolarized image
    #leave prism keyword blank to test that it autofills to None
    optics_keywords_unpol = {'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':output_dim,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_unpol = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_unpol, if_quiet=True)
    sim_scene_unpol = optics_unpol.get_host_star_psf(base_scene)
    image_star_corgi_unpol = sim_scene_unpol.host_star_image.data
    sim_scene_unpol = optics_unpol.inject_point_sources(base_scene, sim_scene_unpol)
    image_comp_corgi_unpol = sim_scene_unpol.point_source_image.data
    
    #check polarized intensities add up to 0.96 * unpolarized intensity (wollaston have 96% transmission)
    #check sum of 0 and 90 image is the same as the sum of 45 and 135 image
    assert (image_star_corgi_x + image_star_corgi_y)  == pytest.approx(image_star_corgi_unpol * 0.96, rel=0.05)
    assert (image_star_corgi_x + image_star_corgi_y) == pytest.approx(image_star_corgi_45 + image_star_corgi_135, rel=0.05)
    assert (image_comp_corgi_x + image_comp_corgi_y)  == pytest.approx(image_comp_corgi_unpol * 0.96, rel=0.05)
    assert (image_comp_corgi_x + image_comp_corgi_y) == pytest.approx(image_comp_corgi_45 + image_comp_corgi_135, rel=0.05)

    ## double check the output polarized intensities of the point sources is what's expected
    # instrument mueller matrix at 575nm, the band 1 center
    instrument_mm = pol.get_instrument_mueller_matrix([0.575])
    # the four wollaston mueller matrices
    wollaston_mm_0 = pol.get_wollaston_mueller_matrix(0)
    wollaston_mm_45 = pol.get_wollaston_mueller_matrix(45)
    wollaston_mm_90 = pol.get_wollaston_mueller_matrix(90)
    wollaston_mm_135 = pol.get_wollaston_mueller_matrix(135)
    # apply instrument polarization effects to source stokes vector
    companion_pol = instrument_mm @ companion_pol
    companion_pol = companion_pol / companion_pol[0]
    # apply polarizer
    i_0 = (wollaston_mm_0 @ companion_pol)[0]
    i_45 = (wollaston_mm_45 @ companion_pol)[0]
    i_90 = (wollaston_mm_90 @ companion_pol)[0]
    i_135 = (wollaston_mm_135 @ companion_pol)[0]
    # check that the calculated polarized intensities here matches up with the simulation
    assert image_comp_corgi_x == pytest.approx(i_0 * image_comp_corgi_unpol, rel=0.05)
    assert image_comp_corgi_45 == pytest.approx(i_45 * image_comp_corgi_unpol, rel=0.05)
    assert image_comp_corgi_y == pytest.approx(i_90 * image_comp_corgi_unpol, rel=0.05)
    assert image_comp_corgi_135 == pytest.approx(i_135 * image_comp_corgi_unpol, rel=0.05)



if __name__ == '__main__':
    test_polarimetry()