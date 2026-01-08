from multiprocessing.managers import Value

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import os
from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
from scipy.interpolate import interp1d
from corgisim import outputs
import os


def get_reference_aberration(cycle=1):
    """
    Retrieve the start/end times of the requested reference star cycle from os11 scenario.
    Args:
        cycle (int): int between 1 and 6 

    Returns:
        time_ref_cycle_min, time_ref_cycle_max (floats): start/end times of the requested reference star cycle

    """
    script_dir = os.path.dirname(__file__)

    batch_info = fits.getdata(os.path.join(script_dir , 'data/hlc_os11_inputs.fits')).transpose()
    batch_time = batch_info[0]
    batch_id = batch_info[2]

    match_id = cycle - 1
    where_id = np.where(batch_id == match_id)

    time_ref_cycle_min = np.min(batch_time[where_id])
    time_ref_cycle_max = np.max(batch_time[where_id])

    return time_ref_cycle_min, time_ref_cycle_max


def get_science_acquisition(cycle=1, roll=1):
    """
        Retrieve the start/end times of the requested target star cycle/roll from os11 scenario.
        Args:
            cycle (int): int between 1 and 6 
            roll (int): Either 1 or 2 corresponding respectively to the positive and negative roll angle

        Returns:
            time_ref_cycle_min, time_ref_cycle_max (floats): start/end times of the requested reference star cycle

        """
    script_dir = os.path.dirname(__file__)
    
    batch_info = fits.getdata(os.path.join(script_dir , 'data/hlc_os11_inputs.fits')).transpose()
    batch_time = batch_info[0]
    batch_id = batch_info[2]

    match_id = cycle * 100 + (roll - 1)
    where_id = np.where(batch_id == match_id)

    time_sci_cycle_min = np.min(batch_time[where_id])
    time_sci_cycle_max = np.max(batch_time[where_id])

    return time_sci_cycle_min, time_sci_cycle_max


def get_drift(exp_time, n_frames, obs='science', cycle=1, roll=1, lowfs_use=True):
    """
    Function to interpolate the WFE estimated in OS11 on a different frame time 
    for a given observation visit (i.e. cycle and roll).
    
    Args:
        exp_time (float): Exposure time for one frame in seconds
        n_frames (int): Number of frames for a visit
        obs (str): Either 'science' or 'ref'
        cycle (int): int between 1 and 6
        roll (int): 1 or 2 corresponding respectively to the negative and positive roll angle
        lowfs_use (bool): If True, the os11 lowfs correction is applied

    Returns:
        noll_index (numpy.array): zernike noll index (n_noll_index,)
        zval_m (numpy.array): of the zernike rms value in meters (n_noll_index, n_frames)
    """
    
    print("n_frames", n_frames)
    if type(cycle)==str:
        cycle = int(cycle)
    exp_time /= 3600  # exp_time in hour
    if obs == 'science':
        t0, t1 = get_science_acquisition(cycle, roll)
    elif obs == 'ref':
        t0, t1 = get_reference_aberration(cycle)
    else:
        raise ValueError("obs must be either 'science' or 'ref'")

    noll_index = np.arange(4, 46, 1)
    script_dir = os.path.dirname(__file__)

    a = fits.getdata(os.path.join(script_dir , 'data/hlc_os11_inputs.fits')).transpose()
    zval_m = np.zeros((n_frames, noll_index.shape[0]))
    t_interp = np.arange(t0, t1, exp_time)

    for i, idx in enumerate(noll_index):
        f_interp = interp1d(a[0], a[4 + i])
        a_interp = f_interp(t_interp)
        if idx<12 and lowfs_use: #adding
            f_lowfs_corr_interp = interp1d(a[0], a[46 + i])
            a_lowfs = f_lowfs_corr_interp(t_interp)
            a_interp += a_lowfs
        for k in range(n_frames):
            zval_m[k, i] = a_interp[k]

    return noll_index, zval_m

def get_jitter(exp_time, n_frames, obs='science', cycle=1, roll=1):
    print("n_frames", n_frames)
    if type(cycle) == str:
        cycle = int(cycle)
    exp_time /= 3600  # exp_time in hour
    if obs == 'science':
        t0, t1 = get_science_acquisition(cycle, roll)
    elif obs == 'ref':
        t0, t1 = get_reference_aberration(cycle)
    else:
        raise ValueError("obs must be either 'science' or 'ref'")

    tiptilt_index = np.arange(2, 4, 1)
    script_dir = os.path.dirname(__file__)

    a = fits.getdata(os.path.join(script_dir, 'data/hlc_os11_inputs.fits')).transpose()
    zval_m = np.zeros((n_frames, tiptilt_index.shape[0]))
    t_interp = np.arange(t0, t1, exp_time)

    for i, idx in enumerate(tiptilt_index):
        f_interp = interp1d(a[0], a[78 + i])
        a_interp = f_interp(t_interp)

        for k in range(n_frames):
            zval_m[k, i] = a_interp[k]

    return tiptilt_index, zval_m


def estimate_companion_position_roll(x_off_mas, y_off_mas, roll_0='positive', theta=-13):
    """
    
    Args:
        x_off_mas (float): x-offset of the companion in mas 
        y_off_mas (float): y-offset of the companion in mas 
        roll_0 (str): Either 'positive' or 'negative'
        theta (float): Roll angle 

    Returns:

    """
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

def generate_time_series_from_os11(ref_star_properties, target_star_properties, point_source_info, frame_exp, total_exp_time, outdir, cor_type='hlc', bandpass = '1F', polaxis=10, dm_case='3e-8'):
    """

    Args:
        ref_star_properties: Dict or None
            Dictionary with the reference star properties (i.e. Star magnitude, spectral type etc...)
            if None, no reference star time series will be simulated
        target_star_properties: Dict or None
            Dictionary with the target star properties (i.e. Star magnitude, spectral type etc...)
            if None, no target star time series will be simulated
        point_source_info: List or None
            List of dictionary for the companion point source simulation (i.e. magnitude, x_offset, y_offset etc...)
        frame_exp: float
            single exposure time in seconds
        total_exp_time: float
            total exposure time in seconds
        outdir: str
            Output directory to save the time series
        cor_type: str (default is 'hlc')
            coronagraph mode ('zwfs' not supported here)
        bandpass: str (default is '1F')
            filter bandpass name
        polaxis: float (default is 10)
        dm_case: str (default is '3e-8')
            deformable mirror dark hole scenario
    """

    if cor_type == 'zwfs':
        raise ValueError("ZWFS mode not supported in this function")

    outdir_noisy = os.path.join(outdir, 'noisy')
    outdir_noiseless = os.path.join(outdir, 'noiseless')


    dm_case_name = cor_type + '_ni_' + dm_case
    dm1 = proper.prop_fits_read(
        roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(
        roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')



    if ref_star_properties is not None:
        base_scene = scene.Scene(ref_star_properties) #define the astrophysical scene for the reference star
        N_obs = int(total_exp_time / frame_exp)
        N_obs_per_cycle = N_obs // 5

        for cycle in range(1, 6):
            zernike_poly_index, zernike_value_m = get_drift(frame_exp, N_obs_per_cycle, obs='ref', cycle=cycle)
            for n in range(N_obs_per_cycle):
                output_save_file = f'psf_ref_{n}_cycle_{cycle}.fits'
                output_ccd_save_file = f'psf_ref_{n}_cycle_{cycle}_ccd.fits'

                plt.title('WFE')
                plt.xlabel('Zernike noll coeff')
                plt.ylabel('WFE rms (pm)')
                plt.plot(zernike_poly_index, zernike_value_m[n] * 1e12)
                plt.show()

                optics_keywords = {'cor_type': cor_type, 'use_errors': 2, 'polaxis': polaxis, 'output_dim': 51, \
                                   'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_fpm': 1,
                                   'use_lyot_stop': 1,
                                   'use_field_stop': 1, 'zindex': zernike_poly_index, 'zval_m': zernike_value_m[n]}

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
                outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir_noisy, filename=output_ccd_save_file,
                                         write_as_L1=False)

    if target_star_properties is not None:
        N_obs = int(total_exp_time / frame_exp)
        N_obs_per_cycle = N_obs // 4
        N_obs_per_cycle_per_roll = N_obs_per_cycle // 4

        #### simulate using corgisim

        for cycle in range(1, 5):
            for roll in range(1, 5):
                zernike_poly_index, zernike_value_m = get_drift(frame_exp, N_obs_per_cycle_per_roll, obs='science', cycle=cycle, roll=roll)

                for n in range(N_obs_per_cycle_per_roll):
                    output_save_file = f'psf_target_{n}_cycle_{cycle}_roll_{roll}.fits'
                    output_ccd_save_file = f'psf_target_{n}_cycle_{cycle}_roll_{roll}_ccd.fits'

                    if point_source_info is not None:
                        # compute the x_off, y_off of the companion with respect to roll angles
                        point_source_info_roll = point_source_info

                        if roll % 2 == 0:
                            for i in range(len(point_source_info)):
                                point_source_info_roll[i]['position_x'], point_source_info_roll[i]['position_y'] = estimate_companion_position_roll(point_source_info[i]['position_x'], point_source_info[i]['position_y'],
                                                                                                  roll_0='positive', theta=-13)

                        # Create a Scene object that holds all this information
                        base_scene = scene.Scene(target_star_properties, point_source_info_roll)

                    else:
                        base_scene = scene.Scene(target_star_properties)

                    plt.title('WFE')
                    plt.xlabel('Zernike noll coeff')
                    plt.ylabel('WFE rms (pm)')
                    plt.plot(zernike_poly_index, zernike_value_m[n] * 1e12)
                    plt.show()

                    optics_keywords = {'cor_type': cor_type, 'use_errors': 2, 'polaxis': polaxis, 'output_dim': 51, \
                                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_fpm': 1,
                                       'use_lyot_stop': 1,
                                       'use_field_stop': 1, 'zindex': zernike_poly_index, 'zval_m': zernike_value_m[n]}

                    optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                                    integrate_pixels=True)

                    sim_scene = optics.get_host_star_psf(base_scene)

                    sim_scene = optics.inject_point_sources(base_scene, sim_scene)


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
                    outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless,
                                             filename=output_save_file,
                                             write_as_L1=False) #TODO is it possible to save both noiseless host_star_image and off axis?
                    outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir_noisy, filename=output_ccd_save_file,
                                             write_as_L1=False)

    return 1