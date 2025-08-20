from corgisim import scene,instrument
import numpy as np
import proper
import roman_preflight_proper
from pyklip.fakes import gaussfit2d

def test_add_satellite_spots():
    
    #### simulation parameters
    # --- Host Star Properties ---
    Vmag = 5                            # V-band magnitude of the host star
    sptype = 'G0V'                      # Spectral type of the host star
    host_star_properties = {'Vmag': Vmag,
                        'spectral_type': sptype,
                        'magtype': 'vegamag'}
    # contrast of satellite spots
    contrast = 1e-5
    ####

    # --- Create the Astrophysical Scene ---
    base_scene = scene.Scene(host_star_properties)

    # Simulation mode (currently only 'excam' is implemented)
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    ##  Define the polaxis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10
    # output_dim define the size of the output image
    output_dim = 51


    ## 1) simulate an offset stellar PSF, as a reference ##
    # calculate offset
    wavelength = 0.575e-6 # meter, assuming band 1
    lam_D = np.degrees(wavelength/2.3)*3600*1000 # in mas
    shift = [0, 6*lam_D] # shift in [x,y]

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1,\
                 'source_x_offset_mas': shift[0], 'source_y_offset_mas': shift[1]}

    ##define the corgi.optics class that hold all information about the instrument paramters
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)

    # generate the offset PSF
    sim_scene = optics.get_host_star_psf(base_scene)
    image_star = sim_scene.host_star_image.data

    ## 2) simulate satellite spots by modifying the DM1 solutions ##
    optics_keywords_ss ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    satspot_keywords = {'sep_lamD': 6, 'angle_deg': [0,90], 'contrast': contrast, 'wavelength_m': wavelength}

    ##define the corgi.optics class that hold all information about the instrument paramters                    
    optics_with_spots = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords_ss, satspot_keywords=satspot_keywords, if_quiet=True)

    sim_scene_with_spots = optics_with_spots.get_host_star_psf(base_scene)
    image_star_with_spots = sim_scene_with_spots.host_star_image.data


    #### PSF fitting of the reference PSF and the satellite spots
    ## 1) reference
    fov_shape = image_star.shape
    xcen = (fov_shape[1]-1)/2
    ycen = (fov_shape[0]-1)/2
    pix_scale = 0.0218*1000 #mas/pix
    guess_x = xcen+shift[0]/pix_scale
    guess_y = ycen+shift[1]/pix_scale

    peak_ref,fwhm_ref,x_ref,y_ref=gaussfit2d(image_star,guess_x,guess_y,guesspeak= np.nanmax(image_star))

    ## 2) satellite spots
    fitresult=[]
    for angle in satspot_keywords['angle_deg']:
        sep = satspot_keywords['sep_lamD']*lam_D # mas

        guess_x1 = xcen+sep*np.cos(np.radians(angle))/pix_scale
        guess_x2 = xcen-sep*np.cos(np.radians(angle))/pix_scale

        guess_y1 = ycen+sep*np.sin(np.radians(angle))/pix_scale
        guess_y2 = ycen-sep*np.sin(np.radians(angle))/pix_scale

        for guess_x,guess_y in zip([guess_x1,guess_x2],[guess_y1,guess_y2]):
            peak,fwhm,x,y=gaussfit2d(image_star_with_spots,guess_x,guess_y)
            fitresult.append([peak,fwhm,x,y])

    #### check location, averaged coordinates the four satellite spots are located at the FoV center
    guessx_star = np.mean(np.array(fitresult)[:,2])
    guessy_star = np.mean(np.array(fitresult)[:,3])
    assert np.sqrt((guessx_star-xcen)**2+(guessy_star-ycen)**2) < 0.5

    #### check PSF amplitude, averaged peak value of the satellite spots are consistent with the scaled stellar PSF at the given contrast
    peak_satspots_ave = np.mean(np.array(fitresult)[:,0])
    assert peak_satspots_ave/(peak_ref*contrast) < 1.5, peak_satspots_ave/(peak_ref*contrast) > 0.5 
    #threshold is set to 50%

if __name__ == '__main__':
    test_add_satellite_spots()
