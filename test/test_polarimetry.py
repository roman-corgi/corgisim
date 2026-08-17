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
    host_star_pol = np.array([1, 0.05, 0.02, 0])
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag', 'pol_state': host_star_pol}

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
    roll_angle = 26
    

    #Generate 0/90 image pair
    optics_keywords_0_90 = {'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':output_dim, 'prism':'POL0',\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_0_90 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_0_90, if_quiet=True, roll_angle=roll_angle)
    sim_scene_0_90 = optics_0_90.get_host_star_psf(base_scene)
    image_star_corgi_x = sim_scene_0_90.host_star_image.data[0]
    image_star_corgi_y = sim_scene_0_90.host_star_image.data[1]
    sim_scene_0_90 = optics_0_90.inject_point_sources(base_scene, sim_scene_0_90)
    image_comp_corgi_x = sim_scene_0_90.point_source_image.data[0]
    image_comp_corgi_y = sim_scene_0_90.point_source_image.data[1]

    #Put on detector 
    gain =1000
    emccd_keywords ={'em_gain':gain}
    detector = instrument.CorgiDetector(emccd_keywords)
    exptime = 100
    sim_scene_0_90 = detector.generate_detector_image(sim_scene_0_90,exptime,  cut_sub_frame = True)
    image_tot_corgi_sub_0= sim_scene_0_90.image_on_detector.data[0]
    image_tot_corgi_sub_90= sim_scene_0_90.image_on_detector.data[1]
    #Due to the randomness of cosmic rays, this test could fail incorrectly, but not consistently

    for image in [image_tot_corgi_sub_0,image_tot_corgi_sub_90]:
        #Test that there are no wrap in the tail of the cosmic rays
        last_col = image[:, -1] 
        first_col = image[:, 0]
        #Identify rays that could wrap
        index_last = [i for i in range(len(last_col)) if last_col[i] > gain]
        index_first = [i for i in range(len(first_col)) if first_col[i] > gain]
        wrap =False
        #Due to the randomness of cosmic rays, this test could fail incorrectly, but not consistently
        for index in index_last:
            if index+1 in index_first:
                wrap = True
        assert wrap==False
    #Generate 45/135 image pair
    optics_keywords_45_135 = {'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':output_dim, 'prism':'POL45',\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_45_135 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_45_135, if_quiet=True, roll_angle=roll_angle)
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
    optics_unpol = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_unpol, if_quiet=True, roll_angle=roll_angle)
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

    # check that the output polarization of the companion matches what is expected
    # since the companion input is a point source, we sum the flux over the entire focal plane for both the difference and sum images
    # to obtain normalized scalar values for the Q and U output of corgisim
    comp_q_flux_normalized = np.sum(image_comp_corgi_x - image_comp_corgi_y) / np.sum(image_comp_corgi_x + image_comp_corgi_y)
    comp_u_flux_normalized = np.sum(image_comp_corgi_45 - image_comp_corgi_135) / np.sum(image_comp_corgi_45 + image_comp_corgi_135)
    # next check that the corgisim Q and U output is the same as the companion stokes vector transformed by the predefined IP
    telescope_roll_mm = pol.get_rotation_mueller_matrix(roll_angle)
    instrument_pol_mm = pol.get_instrument_mueller_matrix(optics_unpol.lam_um)
    # normalize
    instrument_pol_mm = instrument_pol_mm / instrument_pol_mm[0, 0]
    # transform to obtain expected output
    comp_stokes_output = instrument_pol_mm @ telescope_roll_mm @ companion_pol
    # check the two match
    assert comp_q_flux_normalized == pytest.approx(comp_stokes_output[1], rel=0.05)
    assert comp_u_flux_normalized == pytest.approx(comp_stokes_output[2], rel=0.05)

if __name__ == '__main__':
    test_polarimetry()