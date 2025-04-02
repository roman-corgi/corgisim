from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim

#######################
### Set up a scene. ###
#######################

def run_sim_multi():
    print('testrun')
    
    #Define the host star properties
    #host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
    Vmag = 5
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1'
    cor_type = 'hlc_band1'
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
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
    
    gain =500
    emccd_keywords ={'em_gain':gain}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    #print('aaa',detector.emccd.meta.geom )
    sim_scene = detector.generate_detector_image(sim_scene, exptime,full_frame=True,loc_x=300, loc_y=300)
    image2 = sim_scene.image_on_detector.data


    print('Final_intensity_get:', np.sum(image, dtype = np.float64))
    

    #print(sim_scene.host_star_image[1].header)
    ####################################################################
    fig = plt.figure(figsize=(8,4))
    plt.subplot(111)
    
    plt.imshow(image2, origin='lower')
    #plt.scatter(1088,13,c='red',s=10)
    #plt.text(1088,13,'[1088,13]',c='red')

    #plt.scatter(2111,13,c='red',s=10)
    #plt.text(2111-300,13,'[2111,13]',c='red')

    #plt.scatter(1088,1036,c='red',s=10)
    #plt.text(1088,1036-60,'[1088,1036]',c='red')

    #plt.scatter(2111,1036,c='red',s=10)
    #plt.text(2111-400,1036-60,'[2111,1036]',c='red')
    
    co = plt.colorbar(shrink=0.7)
    co.set_label(r'$\rm Counts\ [photons\ s^{-1}]$')
    plt.xlabel('X (Pixel)')
    plt.ylabel('X (Pixel)')
    plt.title(f"On-axis star on EMCCD: {sptype} and {Vmag} mag (corgisim)")

    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.show()


    exit()

    

if __name__ == '__main__':
    #run_sim()
    run_sim_multi()





