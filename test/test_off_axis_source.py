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
def run_sim():
    #print('Test if the off axis flat spectrum is correct by comparing with analitical model')

    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1b'
    cor_type = 'hlc_band1'

    mag_companion = [25,25]
    dx= [3,-3]
    dy= [3,-3]
    

    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    
    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    point_source_info = {'Vmag': mag_companion, 'magtype': 'vegamag','position_x':dx , 'position_y':dy}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    proper_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True, integrate_pixels=True)
    sim_scene = optics.get_psf(base_scene)
    image = sim_scene.host_star_image.data

    image2 = optics.inject_point_sources(base_scene)
    
    #print('Final_intensity_get:', np.sum(image, dtype = np.float64))
    #print(sim_scene.host_star_image[1].header)
    
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Host star Vmag=8')

    co = plt.colorbar(shrink=0.7)
    plt.subplot(132)
    plt.imshow(image2)
    plt.title('Companion Vmag=25')

    co = plt.colorbar(shrink=0.7)

    plt.subplot(133)
    plt.imshow(image2+image)
    plt.title('Combined Image')

    co = plt.colorbar(shrink=0.7)
    plt.show()
    

if __name__ == '__main__':
    run_sim()
  