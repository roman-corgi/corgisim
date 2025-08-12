from corgisim import scene, instrument, inputs, observation
from corgisim.scene import SimulatedImage
from astropy.io import fits
import proper
import roman_preflight_proper
import pytest
import numpy as np
import os

def test_generate_observation_sequence():

    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    gain =1000
    emccd_keywords ={'em_gain':gain}

    base_scene = scene.Scene(host_star_properties)
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True, integrate_pixels=True)
    detector = instrument.CorgiDetector( emccd_keywords)
    
    exp_time = 2000
    n_frames = 1
    
    # Test a single frame 
    simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == n_frames
    assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

    # Test a single full frame 
    simulatedImage_list_fullframe = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames,full_frame=True, loc_x=300, loc_y=300)

    assert isinstance(simulatedImage_list_fullframe, list)
    assert len(simulatedImage_list_fullframe) == n_frames
    assert isinstance(simulatedImage_list_fullframe[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector, fits.hdu.hdulist.HDUList)

    assert len(simulatedImage_list_fullframe[n_frames-1].image_on_detector) == 2 # Primary and Image HDU
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].data, np.ndarray)

    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EXPTIME'] == exp_time
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EMGAIN_C'] == gain

    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['PHTCNT'] == detector.photon_counting
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['OPGAIN'] == gain
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['FRAMET'] == exp_time
    # Test several frames

    n_frames = 1000
    simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == n_frames
    assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

def test_generate_observation_scenario_from_cpgs():
    script_dir = os.getcwd()

    #Test with target and reference
    filepath = 'test/test_data/cpgs_short_sequence.xml'
    abs_path =  os.path.join(script_dir, filepath)
   
    scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(abs_path)
    len_list = 0 
    for visit in visit_list:
        len_list += visit['number_of_frames']

    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == len_list
    assert isinstance(simulatedImage_list[0], SimulatedImage)

    #Test with only target
    filepath = 'test/test_data/cpgs_without_reference.xml'
    abs_path =  os.path.join(script_dir, filepath)
   
    scene_target, optics, detector_target, visit_list = inputs.load_cpgs_data(abs_path)
    len_list = 0 
    for visit in visit_list:
        len_list += visit['number_of_frames']

    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == len_list
    assert isinstance(simulatedImage_list[0], SimulatedImage)

if __name__ == '__main__':
    test_generate_observation_sequence()
    test_generate_observation_scenario_from_cpgs()