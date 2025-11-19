from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
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

