import corgisim
from corgisim import scene, instrument, outputs, inputs, observation
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os, shutil
import glob


def test_fastgain_params():
    #### testing the defalut value pass to header
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
    cor_type = 'hlc_band1'
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':51,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1, }
                

    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    #############################################################################################################
    emccd_keywords ={'em_gain':1000}
    exptime = 3000
 
    detector = instrument.CorgiDetector( emccd_keywords)
    
    ## defult value of fast_gain_mode is auto
    ## for em_gain>200, fast_gain_mode is True and gain_CIC_roman is 'Roman'
    assert detector.emccd.fast_gain_mode == True, f"Expected FAST_GAIN_MODE='True', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == pytest.approx(detector.emccd.avg_gain_P/40, rel=0.05)
    print(detector.emccd.fast_gain_mode, detector.emccd.gain_CIC_Q)

    #############################################################################################################
    emccd_keywords ={'em_gain':100}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    
    ## defult value of fast_gain_mode is auto
    ## for em_gain<200, fast_gain_mode is False and gain_CIC_roman is '0'
    assert detector.emccd.fast_gain_mode == False, f"Expected FAST_GAIN_MODE='False', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == 0, f"Expected gain_CIC_Q='0', but got {detector.emccd.gain_CIC_Q}"


    #############################################################################################################

    emccd_keywords ={'em_gain':1000, 'gain_CIC_Q':0}
    exptime = 3000
    
    with pytest.warns(UserWarning, match='gain_CIC_Q is set by default'):
        detector = instrument.CorgiDetector(emccd_keywords)
    
    ## defult value of fast_gain_mode is auto
    ## for em_gain>200, fast_gain_mode is True and gain_CIC_roman is 'Roman'
    ## the input 'gain_CIC_Q':0 will be overwrite if fast_gain_mode is Auto
    assert detector.emccd.fast_gain_mode == True, f"Expected FAST_GAIN_MODE='True', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == pytest.approx(detector.emccd.avg_gain_P/40, rel=0.05)
    
    
    
    #############################################################################################################
    emccd_keywords ={'em_gain':100, 'gain_CIC_Q':0}
    exptime = 3000

    with pytest.warns(UserWarning, match='gain_CIC_Q is set by default'):
        detector = instrument.CorgiDetector(emccd_keywords)

    ## defult value of fast_gain_mode is auto
    ## for em_gain<200, fast_gain_mode is False and gain_CIC_roman is '0'
    ## the input 'gain_CIC_Q':0 will be overwrite if fast_gain_mode is Auto
    assert detector.emccd.fast_gain_mode == False, f"Expected FAST_GAIN_MODE='False', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == 0, f"Expected gain_CIC_Q='0', but got {detector.emccd.gain_CIC_Q}"
    

    #############################################################################################################

    emccd_keywords ={'em_gain':1000, 'fast_gain_mode':True, 'gain_CIC_Q':0}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    
    ##test the emccd_keywords passed to emccd_detect

    ## when fast_gain_mode is not auto, gain_CIC_Q got from  emccd_keywords 
    assert detector.emccd.fast_gain_mode == True, f"Expected FAST_GAIN_MODE='True', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == 0, f"Expected gain_CIC_Q='0', but got {detector.emccd.gain_CIC_Q}"
 
    
    
    #############################################################################################################
    emccd_keywords ={'em_gain':100, 'fast_gain_mode':True,'gain_CIC_Q':'roman'}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)

    ## when fast_gain_mode is not auto, gain_CIC_Q got from  emccd_keywords 
    assert detector.emccd.fast_gain_mode == True, f"Expected FAST_GAIN_MODE='True', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == pytest.approx(detector.emccd.avg_gain_P/40, rel=0.05)

    #############################################################################################################

    emccd_keywords ={'em_gain':1000, 'fast_gain_mode':False, 'gain_CIC_Q':0}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    
    ##test the emccd_keywords passed to emccd_detect

    ## when fast_gain_mode is not auto, gain_CIC_Q got from  emccd_keywords 
    assert detector.emccd.fast_gain_mode == False, f"Expected FAST_GAIN_MODE='False', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == 0, f"Expected gain_CIC_Q='0', but got {detector.emccd.gain_CIC_Q}"
 
    
    
    #############################################################################################################
    emccd_keywords ={'em_gain':100, 'fast_gain_mode':False,'gain_CIC_Q':'roman'}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)

    ## when fast_gain_mode is not auto, gain_CIC_Q got from  emccd_keywords 
    assert detector.emccd.fast_gain_mode == False, f"Expected FAST_GAIN_MODE='False', but got {detector.emccd.fast_gain_mode}"
    assert detector.emccd.gain_CIC_Q == pytest.approx(detector.emccd.avg_gain_P/40, rel=0.05)
    
    

    

if __name__ == '__main__':
    test_fastgain_params()
