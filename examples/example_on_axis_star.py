from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper

#######################
### Set up a scene. ###
#######################

def run_sim():
    print('testrun')
    
    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    host_star_properties = {'Teff': 5770, 'Dist':4.8481e-6, 'Rs':1}

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    sim_scene = scene.SimulatedScene(base_scene)

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
    optics.get_psf(base_scene, sim_scene)

    print('Final_intensity_get:', np.sum(sim_scene.host_star_image, dtype = np.float64))

    fig = plt.figure()
    plt.imshow(sim_scene.host_star_image)
    plt.colorbar()
    plt.show()


    exit()

    ##########################################
    ### Define the state of the instrument ###
    ##########################################

    #Set up the configuration of the instrument and the detector
    proper_keywords = {"filter":"Band 1","roll_angle": 36.5, "dm1_filename": "dm1.fits", "dm2_filename": "dm2.fits"}
    emccd_keywords = {"read_noise": 3}

    # Set up the optics
    optics = instrument.CorgiOptics(proper_keywords)

    # Set up the detector
    detector = instrument.CorgiDetector(emccd_keywords)

    ##########################
    ### Simulate the scene ###
    ##########################

    # Populate scene.host_star_image with an astropy HDU that contains a noiseless on-axis PSF
    simulated_scene_with_psf = optics.get_psf(base_scene, on_the_fly=True, oversample = 1, return_oversample = False)
    total_scene = scene.combine_simulated_scence_list(simulated_scene_with_psf)

    ############################
    ### Simulate the readout ###
    ############################
    #Final Image should be an HDU with the image in the first extension and the 0th extension contains the L1 header information. 
    final_image = detector.generate_detector_image(total_scene)


if __name__ == '__main__':
    run_sim()





