from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
from synphot.models import BlackBodyNorm1D, Box1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
from astropy import units as u


#@pytest.mark.parametrize("interp_method", ['linear', 'cubic'])
def test_empty_scene():
    print('This is a test file to test empty_scene for bia and dark frame simulation')
    
    cgi_mode = 'excam'
    cor_type = 'hlc_band1'
    bandpass = '1b'
    proper_keywords ={'cor_type':cor_type,  'output_dim':1024,'polaxis':10 }
    
     #Create a empty Scene object 
    base_scene = scene.Scene()

    optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True)
    sim_scene = optics.generate_empty_scene(base_scene)
  

    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 100
    detector = instrument.CorgiDetector( emccd_keywords)

    sim_scene = detector.generate_detector_image(sim_scene,exptime,full_frame=True,loc_x=512, loc_y=512)
    
   


if __name__ == '__main__':
    test_empty_scene()