from corgisim import scene, instrument, observation
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os

#######################
### Set up a scene. ###
#######################

def test_smear():
    print('testrun')

    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    # ── Instrument configuration ──────────────────────────────────────────────────
    CGI_MODE     = 'excam'
    BANDPASS     = '1F'           # CFAMNAME 1F (broadband filter)
    COR_TYPE     = 'hlc_band1'    

    VMAG         = 5              
    SPTYPE       = 'G4IV'      

    # Observation parameters
    EM_GAIN      = 10
    EXP_TIME_S   = 5
    KGAIN = 8.3 # simulate number different from corgisim default, which is 8.7
    OUTPUT_DIM   = 1024
    OVERSAMPLING = 1 # don't need oversampling for these pupil images
    CR_RATE      = 0     # hits/cm^2/s
    READ_NOISE = 100. # e-; simulate value different from default in DRP
    # EMCCD QE included in bandpass throughput in corgisim, so no need to simulate it here

    base_scene = scene.Scene(
        {'Vmag': VMAG, 'spectral_type': SPTYPE, 'magtype': 'vegamag'})
    optics_keywords = {
            'cor_type':       COR_TYPE,
            'use_errors':     1,
            'polaxis':        10,
            'output_dim':     OUTPUT_DIM,
            'use_pupil_lens' : 1,
            'use_fpm': 0,
            'use_lyot_stop': 0,
            'use_field_stop': 0
        }
    optics = instrument.CorgiOptics(
            CGI_MODE, BANDPASS,
            optics_keywords=optics_keywords,
            oversampling_factor=OVERSAMPLING,
            if_quiet=True,
        )
    # also test that nonlinearity and flat field inputs are functional
    script_dir = os.getcwd()
    nonlin_path = os.path.join(script_dir, 'test', 'test_data', 'nonlin_table_TVAC.txt')
    flat_path = os.path.join(script_dir, 'test', 'test_data', 'flat_sample.fits')
    det = instrument.CorgiDetector(
        {'em_gain': EM_GAIN, 'cr_rate': CR_RATE, 'bias': 2000., 'e_per_dn':KGAIN, 'nonlin_path': nonlin_path, 'flat_path': flat_path,
        'read_noise': READ_NOISE, 'row_read_time': 1}, # large row read time to make smearing obvious
        photon_counting=False,
    )
    sim_images = observation.generate_observation_sequence(
                base_scene, optics, det, EXP_TIME_S, 1,
                save_as_fits=False,
                full_frame=True, loc_x=512, loc_y=512,
            )
    image = sim_images[0].image_on_detector[1].data
    # check that region above pupil image (further from readout register) 
    # has more counts that region below (rows that are closer to readout 
    # and don't clock through pupil region) 
    assert np.nanmean(image[700:1000, 1500:1700]) > np.nanmean(image[40:300, 1500:1700])
    print("Smear test passed")


    

    

if __name__ == '__main__':
    #run_sim()
    test_smear()





