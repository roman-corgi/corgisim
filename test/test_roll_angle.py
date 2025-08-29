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
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '4F'
    cor_type = 'spc-wide'
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
    rootname = 'spc-wide_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1}

    ####first roll,roll=0deg
    optics_roll1 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene_roll1 = optics_roll1.inject_point_sources(base_scene)
    x1, y1 = optics_roll1.optics_keywords_comp['source_x_offset_mas'],optics_roll1.optics_keywords_comp['source_y_offset_mas']
    PA1 = np.rad2deg(instrument.calculate_PA(x1, y1 ))

    ####second roll,roll=roll_angle
    optics_roll2 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, roll_angle= roll_angle,if_quiet=True)
    sim_scene_roll2 = optics_roll2.inject_point_sources(base_scene)
    x2, y2 = optics_roll2.optics_keywords_comp['source_x_offset_mas'],optics_roll2.optics_keywords_comp['source_y_offset_mas']
    PA2 = np.rad2deg(instrument.calculate_PA(x2, y2 ))
 
    dPA = PA2-PA1
    assert dPA == pytest.approx(roll_angle,abs=0.1), (f"Roll-angle check failed: expected  {roll_angle:.3f} degree, got roll={dPA:.3f}degree).")

def test_roll_spec():
    pass

if __name__ == '__main__':
    #run_sim()
    test_roll_imaging()