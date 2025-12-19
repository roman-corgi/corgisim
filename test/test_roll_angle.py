from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim



def test_roll_imaging():
    #### this testing function test if the roll angle function have the results we expect
    #### Spcifically, I generate two simulation, one with 0 deg roll angle and one with 15 deg roll angle
    #### The function test of the PA for the companion out of two siulation offset by 15 deg and in the direction we want

    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc'
    roll_angle=15 ##degree

    #Define companion properties
    #Add one companions, one in FOV
    mag_companion = [20]
    companion_x_pos = [3*49.3]
    companion_y_pos = [-3*49.3]
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x': companion_x_pos[0], 'position_y': companion_y_pos[0]}]
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)
    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['2e-8']       
    rootname = 'spc-wide_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':51,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1}

    ####first roll,roll=0deg
    optics_roll1 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True,oversampling_factor = 1)
    sim_scene_roll1 = optics_roll1.inject_point_sources(base_scene)
    x1, y1 = optics_roll1.optics_keywords_comp['source_x_offset_mas'],optics_roll1.optics_keywords_comp['source_y_offset_mas']
    sep1 = np.sqrt(x1**2+y1**2)
    PA1 = np.rad2deg(np.arctan2(y1, x1))

    ####second roll,roll=roll_angle
    optics_roll2 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, roll_angle= roll_angle,if_quiet=True,oversampling_factor = 1)
    sim_scene_roll2 = optics_roll2.inject_point_sources(base_scene)
    x2, y2 = optics_roll2.optics_keywords_comp['source_x_offset_mas'],optics_roll2.optics_keywords_comp['source_y_offset_mas']
    sep2 = np.sqrt(x2**2+y2**2)
    PA2 = np.rad2deg(np.arctan2(y2, x2))

    dPA = PA1-PA2
    assert dPA == pytest.approx(roll_angle,abs=0.1), (f"Roll-angle check failed: expected  {roll_angle:.3f} degree, got roll={dPA:.3f}degree).")
    assert sep1 == pytest.approx(sep2,abs=0.1), (f"Separation check failed: expected  {sep1:.3f} mas, got {sep2:.3f} mas).")

def test_roll_spec():
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'spec'
    bandpass_corgisim = '3F'
    cor_type = 'spc-spec_band3'
    roll_angle=15 ##degree

    #Define companion properties
    #Add one companions, one in FOV
    mag_companion = [20]
    companion_x_pos = [740]
    companion_y_pos = [740]
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x': companion_x_pos[0], 'position_y': companion_y_pos[0]}]
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)
    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['2e-8']      
    # cases = ['1e-9']      
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    mas_per_lamD = 63.72 # Band 3
    source_x_offset, source_y_offset = 6.0, 6.0 #lam/D
    source_x_offset_mas, source_y_offset_mas = source_x_offset * mas_per_lamD, source_y_offset * mas_per_lamD

    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1,\
                        'slit':'R1C2','prism':'PRISM3','slit_ra_offset_mas':source_x_offset_mas,'slit_dec_offset_mas':source_y_offset_mas,}

    ####first roll,roll=0deg
    optics_roll1 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True,oversampling_factor = 1)
    sim_scene_roll1 = optics_roll1.inject_point_sources(base_scene)
    x1, y1 = optics_roll1.optics_keywords_comp['source_x_offset_mas'],optics_roll1.optics_keywords_comp['source_y_offset_mas']
    sep1 = np.sqrt(x1**2+y1**2)
    PA1 = np.rad2deg(np.arctan2(y1, x1))
  

    x1_slit, y1_slit =  optics_roll1.slit_x_offset_mas,optics_roll1.slit_y_offset_mas
    sep1_slit = np.sqrt(x1_slit**2+y1_slit**2)
    PA1_slit =  np.rad2deg(np.arctan2(y1_slit, x1_slit))

    ####second roll,roll=roll_angle
    optics_roll2 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, roll_angle= roll_angle,if_quiet=True,oversampling_factor = 1)
    sim_scene_roll2 = optics_roll2.inject_point_sources(base_scene)
    x2, y2 = optics_roll2.optics_keywords_comp['source_x_offset_mas'],optics_roll2.optics_keywords_comp['source_y_offset_mas']
    sep2 = np.sqrt(x2**2+y2**2)
    PA2 = np.rad2deg(np.arctan2(y2, x2))


    x2_slit, y2_slit =  optics_roll2.slit_x_offset_mas,optics_roll2.slit_y_offset_mas
    sep2_slit = np.sqrt(x2_slit**2+y2_slit**2)
    PA2_slit =  np.rad2deg(np.arctan2(y2_slit, x2_slit))

    dPA = PA1-PA2
    assert sep1 == pytest.approx(sep2,abs=0.1), (f"Separation check failed: expected  {sep1:.3f} mas, got {sep2:.3f} mas).")
    assert dPA == pytest.approx(roll_angle,abs=0.1), (f"Roll-angle check failed: expected  {roll_angle:.3f} degree, got roll={dPA:.3f}degree).")

    dPA_slit = PA1_slit-PA2_slit
    assert sep1_slit == pytest.approx(sep2_slit,abs=0.1), (f"Separation for slit check failed: expected  {sep1_slit:.3f} mas, got {sep2_slit:.3f} mas).")
    assert dPA_slit == pytest.approx(roll_angle,abs=0.1), (f"Roll-angle for slit check failed: expected  {roll_angle:.3f} degree, got roll={dPA_slit:.3f}degree).")


 

if __name__ == '__main__':
    #test_roll_imaging()
    test_roll_spec()