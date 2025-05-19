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

def test_sptype_teff_mapping():

    sptype='G0V'
    Vmag = 8
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties=host_star_properties)


    ###### testing different spectral type inputs
    ############################# O type stars
    #sptype='O0V'
    #sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    #assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"
    
    sptype='O9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  30000 <= v0 <= 50000, f"Temperature {v0} out of expected range for {sptype}"


    sptype='O0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  30000 <= v0 <= 50000, f"Temperature {v0} out of expected range for {sptype}"

   
    sptype='O0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  30000 <= v0 <= 50000, f"Temperature {v0} out of expected range for {sptype}"


    sptype='O0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  30000 <= v0 <= 50000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='O0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  30000 <= v0 <= 50000, f"Temperature {v0} out of expected range for {sptype}"
    ############################# B type stars
    sptype='B0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"
    
    sptype='B9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"

    sptype='B0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0<= 31500, f"Temperature {v0} out of expected range for {sptype}"
   
    sptype='B0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"

    sptype='B0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"

    sptype='B0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  10000 <= v0 <= 31500, f"Temperature {v0} out of expected range for {sptype}"

    ############################# A type stars
    sptype='A0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0 <= 10000, f"Temperature {v0} out of expected range for {sptype}"
    
    
    sptype='A9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0 <=10000, f"Temperature {v0} out of expected range for {sptype}"
    

    sptype='A0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0<= 10000, f"Temperature {v0} out of expected range for {sptype}"
 
   
    sptype='A0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0 <= 10000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='A0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0 <= 10000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='A0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  7500 <= v0 <= 10000, f"Temperature {v0} out of expected range for {sptype}"

    ############################# F type stars
    sptype='F0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0 <= 7800, f"Temperature {v0} out of expected range for {sptype}"
   
  
    
    sptype='F9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0 <=7800, f"Temperature {v0} out of expected range for {sptype}"

    sptype='F0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0<= 7800, f"Temperature {v0} out of expected range for {sptype}"
 
    sptype='F0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0 <= 7800, f"Temperature {v0} out of expected range for {sptype}"

    sptype='F0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0 <= 7800, f"Temperature {v0} out of expected range for {sptype}"

    sptype='F0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  5700 <= v0 <= 7800, f"Temperature {v0} out of expected range for {sptype}"

    ############################# G type stars
    sptype='G0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0 <= 6000, f"Temperature {v0} out of expected range for {sptype}"
    
    sptype='G9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0 <=6000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='G0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0<= 6000, f"Temperature {v0} out of expected range for {sptype}"
 
    sptype='G0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0 <= 6000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='G0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0 <= 6000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='G0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  4500 <= v0 <= 6000, f"Temperature {v0} out of expected range for {sptype}"
   

    ############################# K type stars
    sptype='K0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700  <= v0 <= 5400, f"Temperature {v0} out of expected range for {sptype}"
    
    sptype='K9I'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700 <= v0 <=5400, f"Temperature {v0} out of expected range for {sptype}"

    sptype='K0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700 <= v0<= 5400, f"Temperature {v0} out of expected range for {sptype}"
 
    sptype='K0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700 <= v0 <= 5400, f"Temperature {v0} out of expected range for {sptype}"

    sptype='K0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700 <= v0 <= 5400, f"Temperature {v0} out of expected range for {sptype}"

    sptype='K0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  3700 <= v0 <= 5400, f"Temperature {v0} out of expected range for {sptype}"

    ############################# M type stars
    sptype='M0V'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400  <= v0 <= 4000, f"Temperature {v0} out of expected range for {sptype}"
    
    sptype='M7II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <=4000, f"Temperature {v0} out of expected range for {sptype}"
    

    sptype='M8II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <=4000, f"Temperature {v0} out of expected range for {sptype}"
 
    sptype='M6II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <=4000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='M0II'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0<= 4000, f"Temperature {v0} out of expected range for {sptype}"
 
    sptype='M0III'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <= 4000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='M0IV'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <= 4000, f"Temperature {v0} out of expected range for {sptype}"

    sptype='M0'
    sp, v0= base_scene.get_stellar_spectrum(sptype, Vmag, 'vegamag', return_teff=True  )
    assert  2400 <= v0 <= 4000, f"Temperature {v0} out of expected range for {sptype}"



if __name__ == '__main__':
   
    test_sptype_teff_mapping()
    
