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
import os

def test_off_axis_source():
    print('This is a test file to check if the simulated off-axis sources from corgisim agree with those from cgisim')
    
    #### simulate using corgisim
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'

    mag_companion = [25,25]
    ###the position of companions in unit of mas
    ####550nm/2.3m = 29.4 mas
    ###we used sep = 3 lambda/D here 
    dx= [3*49.3,-3*49.3]
    dy= [3*49.3,-3*49.3]
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]},
                         {'Vmag': mag_companion[1], 'magtype': 'vegamag','position_x':dx[1] , 'position_y':dy[1]}]


    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':0, 'output_dim':1024,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    
    print('mid')
    sim_scene = optics.inject_point_sources(base_scene,sim_scene)
    
    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 5000
    detector = instrument.CorgiDetector( emccd_keywords)
    #sim_scene = detector.generate_detector_image(sim_scene,exptime)
   
    sim_scene = detector.generate_detector_image(sim_scene,exptime,full_frame=True)
    #image_tot_corgi_full = sim_scene.image_on_detector[1].data

        

if __name__ == '__main__':
    test_off_axis_source()
  