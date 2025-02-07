from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest

#######################
### Set up a scene. ###
#######################

def run_sim():
    print('testrun')
    
    
    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    host_star_properties = {'Teff': 5770, 'Dist':10, 'Rs':1}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    #sim_scene = scene.SimulatedScene(base_scene)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image
    lam0 = 575  # unit nm
    #nlam = 7
    nlam = 1
    
    if nlam > 1:
        ####simulate broadband image
        bandwidth = 0.1 
        minlam = lam0 * (1 - bandwidth/2)
        maxlam = lam0 * (1 + bandwidth/2)
        lam_array = np.linspace( minlam, maxlam, nlam )  ### array of wavelength in nm

    if nlam ==1:
        ####simulate monochramtic image
        lam_array = np.array([lam0])
    
    print('Begin the simulation at wavelength:'+ str(lam_array))

    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    npsf = 256
    final_sampling = 0.1
    proper_keywords ={'cor_type':'hlc', 'use_errors':2, 'polaxis':10, 'final_sampling_lam0':final_sampling,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1,\
                       'npsf':npsf, 'if_quiet':0, 'if_print_intensity':1 }

    optics = instrument.CorgiOptics(lam_array, proper_keywords=proper_keywords)
    sim_scene = optics.get_psf(base_scene)

    image = sim_scene.host_star_image[1].data
    emccd_keywords ={'em_gain':10.0, 'full_well_image':60000.0, 'full_well_serial':100000.0,\
                     'dark_rate':0.000014, 'cic_noise':0.016, 'read_noise':120, 'bias':0,'qe':1.0, 'cr_rate':0,\
                     'pixel_pitch':13e-6, 'e_per_dn':1.0, 'numel_gain_register':604, 'nbits':14,\
                     'use_traps' : False,'date4traps':2028.0, 'exptime':3600}

    detector = instrument.CorgiDetector(emccd_keywords=emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene)
    image2 = sim_scene.host_star_image_on_detector[1].data

    print('Final_intensity_get:', np.sum(image, dtype = np.float64))
    #print(sim_scene.host_star_image[1].header)

    fig1 = plt.figure()
    plt.imshow(sim_scene.host_star_image[1].data)
    co = plt.colorbar()
    co.set_label(r'$\rm Counts\ [photons\ nm^{-1}\ s^{-1}]$')
    plt.title('On-axis sun-like star at 10 pc')

    fig2 = plt.figure()
    plt.imshow(image2)
    co = plt.colorbar()
    co.set_label(r'$\rm Counts\ [photons\ s^{-1}]$')
    plt.title('On-axis sun-like star at 10 pc on detector')
    plt.show()


    exit()

def run_sim_multi():
    print('testrun')
    
    
    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    host_star_properties = {'Teff': 5770, 'Dist':10, 'Rs':1}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    #sim_scene = scene.SimulatedScene(base_scene)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image
    lam0 = 575  # unit nm
    #nlam = 7
    nlam = 7
    
    if nlam > 1:
        ####simulate broadband image
        bandwidth = 0.1 
        minlam = lam0 * (1 - bandwidth/2)
        maxlam = lam0 * (1 + bandwidth/2)
        lam_array = np.linspace( minlam, maxlam, nlam )  ### array of wavelength in nm

    if nlam ==1:
        ####simulate monochramtic image
        lam_array = np.array([lam0])


    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    npsf = 256
    final_sampling = 0.1
    proper_keywords ={'cor_type':'hlc', 'use_errors':2, 'polaxis':10, 'final_sampling_lam0':final_sampling,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1,\
                       'npsf':npsf, 'if_quiet':0, 'if_print_intensity':1 }

    optics = instrument.CorgiOptics(lam_array, proper_keywords=proper_keywords)
    sim_scene = optics.get_psf(base_scene)
    image = sim_scene.host_star_image[1].data

    emccd_keywords ={'em_gain':1000.0, 'full_well_image':60000.0, 'full_well_serial':100000.0,\
                     'dark_rate':0.00056, 'cic_noise':0.01, 'read_noise':100.0, 'bias':0,'qe':1.0, 'cr_rate':0,\
                     'pixel_pitch':13e-6, 'e_per_dn':1.0, 'numel_gain_register':604, 'nbits':14,\
                     'use_traps' : False,'date4traps':2028.0, 'exptime':3600}
    detector = instrument.CorgiDetector(emccd_keywords=emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene)
    image2 = sim_scene.host_star_image_on_detector[1].data


    print('Final_intensity_get:', np.sum(image, dtype = np.float64))
    #print(sim_scene.host_star_image[1].header)

    fig1 = plt.figure()
    plt.imshow(image)
    co = plt.colorbar()
    co.set_label(r'$\rm Counts\ [photons\ s^{-1}]$')
    plt.title('On-axis sun-like star at 10 pc')


    fig2 = plt.figure()
    plt.imshow(image2)
    co = plt.colorbar()
    co.set_label(r'$\rm Counts\ [photons\ s^{-1}]$')
    plt.title('On-axis sun-like star at 10 pc on detector')

    plt.show()


    exit()

    

if __name__ == '__main__':
    #run_sim()
    run_sim_multi()





