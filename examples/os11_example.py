from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
from scipy.interpolate import interp1d
from corgisim import outputs

def estimate_companion_position_roll(x_off_mas, y_off_mas, roll_0='positive', theta=-13):
    theta_rad = 2 * theta * np.pi/180
    if roll_0 == 'positive':
        theta_rad *=1
    else:
        theta_rad *=-1

    sep_mas = np.sqrt(x_off_mas**2 + y_off_mas**2)
    para_ang = np.arctan2(y_off_mas, x_off_mas)

    x_off_mas_roll = sep_mas * np.cos(para_ang + theta_rad)
    y_off_mas_roll = sep_mas * np.sin(para_ang + theta_rad)

    return x_off_mas_roll, y_off_mas_roll

def get_reference_aberration(cycle=1):
    batch_info = fits.getdata('C:/Users/abidot/Downloads/hlc_os11_v3/hlc_os11_inputs.fits').transpose()
    batch_time = batch_info[0]
    batch_id = batch_info[2]

    match_id = cycle - 1
    where_id = np.where(batch_id == match_id)

    time_ref_cycle_min = np.min(batch_time[where_id])
    time_ref_cycle_max = np.max(batch_time[where_id])

    return time_ref_cycle_min, time_ref_cycle_max


def get_science_acquisition(cycle=1, roll=1):
    # time_roll = 1 #h
    batch_info = fits.getdata('C:/Users/abidot/Downloads/hlc_os11_v3/hlc_os11_inputs.fits').transpose()
    batch_time = batch_info[0]
    batch_id = batch_info[2]

    match_id = cycle * 100 + (roll - 1)
    where_id = np.where(batch_id == match_id)

    time_sci_cycle_min = np.min(batch_time[where_id])
    time_sci_cycle_max = np.max(batch_time[where_id])

    return time_sci_cycle_min, time_sci_cycle_max


def get_drift(t_exp, N_int, obs='science', cycle=1, roll=1, lowfs_use=True):
    print("N_int", N_int)
    t_exp /= 3600  # t_exp in hour
    if obs == 'science':
        t0, t1 = get_science_acquisition(cycle, roll)
    elif obs == 'ref':
        t0, t1 = get_reference_aberration(cycle)

    noll_index = np.arange(4, 46, 1)
    a = fits.getdata('C:/Users/abidot/Downloads/hlc_os11_v3/hlc_os11_inputs.fits').transpose()
    zval_m = np.zeros((N_int, noll_index.shape[0]))
    t_interp = np.arange(t0, t1, t_exp)

    for i, idx in enumerate(noll_index):
        f_interp = interp1d(a[0], a[4 + i])
        a_interp = f_interp(t_interp)
        if idx<12 and lowfs_use: #adding
            f_lowfs_corr_interp = interp1d(a[0], a[46 + i])
            a_lowfs = f_lowfs_corr_interp(t_interp)
            a_interp += a_lowfs
        for k in range(N_int):
            print(k)
            zval_m[k, i] = a_interp[k]

    return zval_m


#######################
### Set up a scene. ###
#######################


def run_sim_obs(vmag_ref, vmag_sci,t_exp, t_tot, x_off_mas, y_off_mas, dmag, polaxis=10, verbose=True, science=True, ref=True):

    #saving outdir to modify
    outdir_noiseless = "/Users/abidot/Desktop/corgi_outputs/noiseless/"
    outdir = "/Users/abidot/Desktop/corgi_outputs/"

    # Define the host star properties
    Vmag_ref = vmag_ref
    Vmag_sci = vmag_sci
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
    cor_type = 'hlc_band1'
    zernike_poly_coef = np.arange(4, 46, 1)

    cases = ['3e-8']
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read(
        roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(
        roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')

    if science:
        N_obs = int(t_tot / t_exp)
        N_obs_per_cycle = N_obs // 4
        N_obs_per_cycle_per_roll = N_obs_per_cycle // 4

        #### simulate using corgisim
        host_star_sci_properties = {'Vmag': Vmag_sci, 'spectral_type': sptype, 'magtype': 'vegamag'}

        for cycle in range(1, 5):
            for roll in range(1, 5):
                zernike_value_m = get_drift(t_exp, N_obs_per_cycle_per_roll, obs='science', cycle=cycle, roll=roll)

                for n in range(N_obs_per_cycle_per_roll):
                    output_save_file = f'psf_sci_{n}_cycle_{cycle}_roll_{roll}.fits'
                    output_ccd_save_file = f'psf_sci_{n}_cycle_{cycle}_roll_{roll}_ccd.fits'

                    #compute the x_off, y_off of the companion with respect to roll angles
                    if roll%2==0:
                        x_off_mas_roll, y_off_mas_roll = estimate_companion_position_roll(x_off_mas, y_off_mas, roll_0='positive', theta=-13)
                    else:
                        x_off_mas_roll, y_off_mas_roll = x_off_mas, y_off_mas

                    # Construct a list of dictionaries for all companion point sources
                    point_source_info = [
                        {
                            'Vmag': vmag_sci + dmag,
                            'magtype': 'vegamag',
                            'position_x': x_off_mas_roll,
                            'position_y': y_off_mas_roll
                        }
                    ]

                    # Create a Scene object that holds all this information
                    base_scene = scene.Scene(host_star_sci_properties, point_source_info)

                    plt.title('WFE')
                    plt.xlabel('Zernike noll coeff')
                    plt.ylabel('WFE rms (pm)')
                    plt.plot(zernike_poly_coef, zernike_value_m[n]*1e12)
                    plt.show()

                    optics_keywords = {'cor_type': cor_type, 'use_errors': 2, 'polaxis': 10, 'output_dim': 51, \
                                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_fpm': 1,
                                       'use_lyot_stop': 1,
                                       'use_field_stop': 1, 'zindex': zernike_poly_coef, 'zval_m': zernike_value_m[n]}

                    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                                    integrate_pixels=True)

                    sim_scene = optics.get_host_star_psf(base_scene)
                    image_star_corgi = sim_scene.host_star_image.data

                    sim_scene = optics.inject_point_sources(base_scene, sim_scene)
                    image_comp_corgi = sim_scene.point_source_image.data
                    combined_image_corgi = image_star_corgi + image_comp_corgi

                    gain = 1000
                    emccd_keywords = {'em_gain': gain}
                    detector = instrument.CorgiDetector(emccd_keywords)

                    sim_scene = detector.generate_detector_image(sim_scene, t_exp)
                    image_tot_corgi_sub = sim_scene.image_on_detector.data

                    plt.imshow(combined_image_corgi)
                    plt.show()
                    plt.imshow(image_tot_corgi_sub)
                    plt.show()


                    ### save products
                    outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                                             write_as_L1=False)
                    outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file, write_as_L1=False)


    if ref:
        N_obs = int(t_tot / t_exp)
        N_obs_per_cycle = N_obs // 5

        #### simulate using corgisim
        host_star_sci_properties = {'Vmag': Vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag'}
        # Create a Scene object that holds all this information
        base_scene = scene.Scene(host_star_sci_properties)

        for cycle in range(1, 6):
            zernike_value_m = get_drift(t_exp, N_obs_per_cycle, obs='ref', cycle=cycle)
            for n in range(N_obs_per_cycle):
                output_save_file = f'psf_ref_{n}_cycle_{cycle}.fits'
                output_ccd_save_file = f'psf_ref_{n}_cycle_{cycle}_ccd.fits'

                plt.title('WFE')
                plt.xlabel('Zernike noll coeff')
                plt.ylabel('WFE rms (pm)')
                plt.plot(zernike_poly_coef, zernike_value_m[n] * 1e12)
                plt.show()

                optics_keywords = {'cor_type': cor_type, 'use_errors': 2, 'polaxis': 10, 'output_dim': 51, \
                                   'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_fpm': 1,
                                   'use_lyot_stop': 1,
                                   'use_field_stop': 1, 'zindex': zernike_poly_coef, 'zval_m': zernike_value_m[n]}

                optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                                integrate_pixels=True)

                sim_scene = optics.get_host_star_psf(base_scene)

                gain = 1000
                emccd_keywords = {'em_gain': gain}
                detector = instrument.CorgiDetector(emccd_keywords)

                sim_scene = detector.generate_detector_image(sim_scene, t_exp)

                ### save products
                outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                                         write_as_L1=False)
                outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file,
                                         write_as_L1=False)


if __name__ == '__main__':
    vmag_ref = 2
    vmag_sci = 0
    t_exp = 1 #sec
    t_tot = 20 #sec
    x_off_mas = 300
    y_off_mas = 0
    dmag = 25

    run_sim_obs(vmag_ref, vmag_sci, t_exp, t_tot, x_off_mas, y_off_mas, dmag, polaxis=10, verbose=True, science=True, ref=True)