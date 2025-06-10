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
def test_off_axis_source():
    print('This is a test file to check if the simulated off-axis sources from corgisim agree with those from cgisim')
    
    #### simulate using corgisim
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    bandpass_cgisim = '1'
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

    proper_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
   
    optics = instrument.CorgiOptics(cgi_mode,  bandpass_corgisim, proper_keywords=proper_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    image_star_corgi = sim_scene.host_star_image.data

    sim_scene = optics.inject_point_sources(base_scene,sim_scene)
    image_comp_corgi = sim_scene.point_source_image.data 
    tot_counts_corgi = np.sum(image_comp_corgi+image_star_corgi, dtype = np.float64)

    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 5000
    detector = instrument.CorgiDetector( emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene,exptime)
    image_tot_corgi = sim_scene.image_on_detector.data

    sim_scene = detector.generate_detector_image(sim_scene,exptime,full_frame=True,loc_x=300, loc_y=300)
    image_tot_corgi_full = sim_scene.image_on_detector[1].data
    

    #### simulate using cgisim
    polaxis_cgisim = -10
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    image_star_cgi, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum=sptype.lower(), star_vmag=Vmag )
    
    image_comp = []
    for i in range(len(mag_companion )):

        params['source_x_offset_mas']=dx[i]
        params['source_y_offset_mas']=dy[i]
        comp_sim_allpol, comp_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass_cgisim,  polaxis_cgisim, params, 
        star_spectrum=sptype.lower(), star_vmag=mag_companion[i] )

        image_comp.append(comp_sim_allpol)
        a0_counts= a0_counts+comp_counts 
    image_comp_cgi = np.sum(image_comp,axis=0)
  
    ####################################Pytest
    # Use pytest.approx to check similarity within a tolerance between corgisim and cgisim output
    ## the test is for noise-free simulation
    assert  image_star_corgi  == pytest.approx(image_star_cgi, rel=0.5)
    assert  image_comp_corgi  == pytest.approx(image_comp_cgi, rel=0.5)

if __name__ == '__main__':
    test_off_axis_source()
  