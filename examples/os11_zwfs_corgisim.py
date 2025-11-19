from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
from scipy.interpolate import interp1d
from corgisim import outputs
from corgisim.wavefront_estimation import *
import os

def get_optimal_emgain(noiseless_image_hdu):
    """

    Args:
        noiseless_image, numpy.array: c(org)isim psf simulation data

    Returns:
        emgain, float: optimal emgain for emccd tuning

    """
    noiseless_image = noiseless_image_hdu.data
    saturation = 12000
    threshold = 0.8 * saturation
    max = np.nanmax(noiseless_image)
    emgain = threshold/max

    return emgain


def get_clear_pupil(star_properties, frame_exp, bandpass = '1F', dm_case='flat', add_drift=False, outdir=None):
    """

    Args:
        star_properties (dict): A dictionary of the star properties
        frame_exp (float): single frame exposure in seconds
        bandpass (str): The bandpass to use
        dm_case, string: can be set to 'flat', '3e-8' ...
        add_drift, bool: whether to add a drift to the simulation

    Returns:

    """
    if outdir is None:
        outdir = "/Users/abidot/Desktop/corgi_outputs/zwfs/"
        outdir_noiseless = "/Users/abidot/Desktop/corgi_outputs/zwfs/noiseless/"
    else:
        outdir_noiseless = os.path.join(outdir, 'noiseless')

    output_save_file = f'clear_pupil_dm_{dm_case}.fits'
    output_ccd_save_file = f'clear_pupil_dm_{dm_case}_ccd.fits'


    base_scene = scene.Scene(star_properties) #define the astrophysical scene

    if dm_case == 'flat':
        dm_case_name = 'hlc_flat_wfe'
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    else:
        dm_case_name = 'hlc_ni_' + dm_case
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')



    optics_keywords = {'cor_type': 'zwfs', 'use_errors': 2, 'polaxis': 10, 'output_dim': 351,
                       'use_fpm': 0, 'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_lyot_stop': 0,
                       'use_pupil_lens': 1}

    if add_drift:
        zernike_poly_index, zernike_value_m = get_drift(1, 1, obs='ref', cycle=1)

        optics_keywords.update({'zindex': zernike_poly_index, 'zval_m': zernike_value_m})

    optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                    integrate_pixels=True)

    sim_scene = optics.get_host_star_psf(base_scene)
    outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                             write_as_L1=False)

    ##Tuning the EMCCD
    emgain = get_optimal_emgain(sim_scene.host_star_image)
    if emgain < 1:
        print(f"WARNING: detector saturated with time exposure of {frame_exp}")
        frame_exp *= emgain
        emgain = 1
        print(f"Setting time exposure to {frame_exp} second(s)")

    emccd_keywords = {'em_gain': emgain}
    detector = instrument.CorgiDetector(emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene, frame_exp)

    outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file, write_as_L1=False)

    return 1

def get_zwfs_pupil(star_properties, frame_exp, total_exp_time, bandpass = '1F', dm_case='flat', outdir=None):
    if outdir is None:
        outdir = "/Users/abidot/Desktop/corgi_outputs/zwfs/"
        outdir_noiseless = "/Users/abidot/Desktop/corgi_outputs/zwfs/noiseless/"
    else:
        outdir_noiseless = os.path.join(outdir, 'noiseless')

    base_scene = scene.Scene(star_properties) #define the astrophysical scene

    if dm_case == 'flat':
        dm_case_name = 'hlc_flat_wfe'
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    else:
        dm_case_name = 'hlc_ni_' + dm_case
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    N_obs = int(total_exp_time / frame_exp)
    N_obs_per_cycle = N_obs // 5

    for cycle in range(1, 6):
        zernike_poly_index, zernike_value_m = get_drift(frame_exp, N_obs_per_cycle, obs='ref', cycle=cycle)
        for n in range(N_obs_per_cycle):
            output_save_file = f'zwfs_pupil_ref_{n}_cycle_{cycle}.fits'
            output_ccd_save_file = f'zwfs_pupil_ref_{n}_cycle_{cycle}_ccd.fits'

            plt.title('WFE')
            plt.xlabel('Zernike noll coeff')
            plt.ylabel('WFE rms (pm)')
            plt.plot(zernike_poly_index, zernike_value_m[n] * 1e12)
            plt.show()

            optics_keywords = {'cor_type': 'zwfs', 'use_errors': 2, 'polaxis': 10, 'output_dim': 351,
                               'use_fpm': 1, 'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
                               'use_lyot_stop': 0, 'use_pupil_lens': 1, 'zindex': zernike_poly_index,
                               'zval_m': zernike_value_m[n]}

            optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                            integrate_pixels=True)

            sim_scene = optics.get_host_star_psf(base_scene)

            ##Tuning the EMCCD
            emgain = get_optimal_emgain(sim_scene.host_star_image)
            if emgain < 1:
                print(f"WARNING: detector saturated with time exposure of {frame_exp}")
                frame_exp *= emgain
                emgain = 1
                print(f"Setting time exposure to {frame_exp} second(s) and emgain to 1")

            emccd_keywords = {'em_gain': emgain}
            detector = instrument.CorgiDetector(emccd_keywords)
            sim_scene = detector.generate_detector_image(sim_scene, frame_exp)

            ### save products
            outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                                     write_as_L1=False)
            outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file,
                                     write_as_L1=False)
    return 1


if __name__ == '__main__':
    vmag_ref = 2
    vmag_sci = 0
    sptype = 'G0V'

    x_off_mas = 300
    y_off_mas = 0

    x_off_star_mas = 0 #not supported yet, set to 0
    y_off_star_mas = 0 #not supported yet, set to 0
    dmag = 25

    frame_exp = 1  # sec
    total_exp_time = 20  # sec

    ref_star_properties = {'Vmag': vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag',
                                'position_x': x_off_star_mas, 'position_y': y_off_star_mas}


    #get_clear_pupil(ref_star_properties, frame_exp)
    get_zwfs_pupil(ref_star_properties, frame_exp, total_exp_time)

    #TODO add Z1 and Z2 jitter + also generate offseted PSF ~ 5 mas