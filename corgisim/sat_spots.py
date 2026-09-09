import numbers
import numpy as np

def add_custom_pattern_dm(dm_volts, pattern_volts, scale, sign="positive"):
    """
    Add an externally supplied DM pattern (in volts) to a Roman CGI DM solution (in volts).

    This supports the "Alternate Probe" observing concept, in which externally
    designed relative-DM probe commands (delivered as ``dmrel_*.fits`` files)
    are applied through the satellite-spot observing path instead of the
    analytically generated cosine pattern (see add_cos_pattern_dm).

    The applied pattern is ``(+/- scale) * pattern_volts``. ``scale`` is
    deliberately REQUIRED (no default) because the convention for the
    delivered probe arrays is unresolved: candidate values of 1.0 and 0.3
    differ in applied probe intensity by roughly a factor of 11, so silently
    choosing one would materially change the simulated data. Callers must
    choose explicitly and record the value used.

    Parameters
    ----------
    dm_volts : numpy.ndarray
        2D array: the original DM solution, in volts.
    pattern_volts : numpy.ndarray
        2D array: relative DM pattern in volts. Must have the same shape as
        ``dm_volts`` (48x48 for the Roman CGI DMs), in the same orientation
        convention as the DM solution files (numpy [row, col] = [y, x]), and
        must be real-valued and finite. A complex-valued array (nonzero
        imaginary component) is rejected rather than silently discarded.
    scale : numbers.Real
        Real, finite scalar (int, float, or a NumPy real scalar); NaN, +/-inf
        and booleans are rejected. Multiplicative amplitude applied to the
        pattern; required, with no default (see above).
    sign : str, optional
        Either "positive" or "negative". "negative" applies
        ``-scale * pattern_volts``, forming the negative member of a
        pairwise-probing pair. Default is "positive".

    Returns
    -------
    numpy.ndarray
        2D array, in volts: the DM map with the scaled pattern added. The
        input ``dm_volts`` array is not modified.

    Raises
    ------
    ValueError
        If ``pattern_volts`` is complex-valued, is not 2D, does not match the
        shape of ``dm_volts``, or contains non-finite values; if ``scale`` is
        not finite; or if ``sign`` is neither "positive" nor "negative".
    TypeError
        If ``scale`` is not a real scalar number.
    """
    pattern = np.asarray(pattern_volts)

    if np.iscomplexobj(pattern):
        raise ValueError(
            "ERROR: custom pattern must be real-valued; got a complex-valued array "
            "(pattern_volts must not have a nonzero imaginary component)"
        )
    pattern = pattern.astype(float)

    if pattern.ndim != 2 or pattern.shape != np.shape(dm_volts):
        raise ValueError(
            f"ERROR: custom pattern shape {pattern.shape} does not match DM shape {np.shape(dm_volts)}"
        )
    if not np.all(np.isfinite(pattern)):
        raise ValueError("ERROR: custom pattern contains non-finite values")
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, numbers.Real):
        raise TypeError(
            "ERROR: scale must be a real scalar number (e.g., int, float, or a NumPy real scalar)"
        )
    if not np.isfinite(scale):
        raise ValueError(f"ERROR: scale must be finite, got {scale}")
    if sign not in ("positive", "negative"):
        raise ValueError(f"ERROR: sign must be 'positive' or 'negative', got '{sign}'")

    signed_scale = scale if sign == "positive" else -scale

    return dm_volts + signed_scale * pattern


def add_cos_pattern_dm(dm_volts, num_pairs=2, sep_lamD=7, angle_deg=[0,90], contrast=1e-6, wavelength_m=0.575e-6, gain_nm_per_V=None, sign = "positive"):
    """
    Add 2D cosine phase pattern(s) to Roman CGI DM solution (in volts). Reference: JPL codes from AJ Riggs and Vanessa Bailey

    Parameters:
        - dm_volts: 2D numpy array (original DM in volts)
        - sep_lamD: int, float, or a list of ints/floats. Number of wave cycles across the pupil diameter. Note that the dm gain value is hardcoded assuming sep_lamD=7 and two-pair satellite spots.
        - angle_deg: int, float, or a list of ints/floats. Orientation angle (degrees, 0 = along X-axis)
        - contrast: int, float, or a list of ints/floats. Expected contrast of sattelite spots, assuming amplitude_rad = 2*sqrt(contrast), where amplitude is a phase amplitude in radians
        - wavelength_m: float. Wavelength in meters
	    - gain_nm_per_V: (optional) gain to convert the DM solution from a nm unit to volts
        - sign: str. Sign of the cosine pattern, either "positive" or "negative"

    Returns:
        - dm_volts_with_pattern: 2D numpy array (in volts), updated DM map with added cosine patterns
    """

    # Roman DM properties - hardcoded, provided by John Krist.
    D_pup_x = 47.41  # in actuators
    D_pup_y = 46.74  # in actuators

    ny, nx = dm_volts.shape
    y, x = np.indices((ny, nx))
    x_center, y_center = (nx - 1) / 2, (ny - 1) / 2  # (23.5, 23.5) for 48x48

    # Shift coordinates to center
    x_shift = x - x_center
    y_shift = y - y_center

    # Ensure number of the pairs is 1 or 2
    if isinstance(num_pairs,int):
        if num_pairs not in (1,2):
            print('Warning: incorrect num_pairs input; use num_pairs=2 insead')
            num_pairs = 2
    else:
        print('Warning: incorrect num_pairs input; use num_pairs=2 insead')
        num_pairs = 2  

    # Ensure angle_deg is iterable
    if isinstance(angle_deg, (int, float)):
        if num_pairs == 2:
            print(f'Warning: input number of pairs is {num_pairs} but one angle_deg is provided; it will generate one pair of the satellite spots') 
            num_pairs = 1
        # prepare the angle list
        angle_deg -= 90 # to algin the 0-deg with x-axis; the input DM has a 90-degree offset 
        angle_list = [angle_deg]
    else: # needs to correct the angle offset of the input DM
        angle_list = []
        for angle_e in angle_deg:
            angle_e -= 90 # to algin the 0-deg with x-axis; the input DM has a 90-degree offset
            angle_list.append(angle_e) 

    # prepare the other satellite spot parameters
    if isinstance(sep_lamD, (int, float)):
        sep_lamD_list = [sep_lamD]*num_pairs 
    if isinstance(contrast, (int, float)):
        contrast_list = [contrast]*num_pairs

    # Initialize output map
    dm_volts_with_pattern = dm_volts.copy()

    # define the gain value
    if gain_nm_per_V is None:
        if num_pairs == 1:
            gain_nm_per_V = 6.11 # nm/V; empirically hardcoded for cgisim to match the scaled off-axis PSF, but it is dependent on actuator capacitance.
        else: # num_pairs is 2
            gain_nm_per_V = 5.70 # nm/V; empirically hardcoded for cgisim to match the scaled off-axis PSF, but it is dependent on actuator capacitance.
    # see Figure 15 in https://doi.org/10.1117/1.JATIS.11.3.031504
    # If you use a smaller value of sep_lamD than the default, the extracted contrast can be biased by the other satellite spots, particularly in an two-pair case.

    # Loop through all angles and add their corresponding cosine patterns
    for angle,sep_lamD,contrast in zip(angle_list,sep_lamD_list,contrast_list):
        theta = np.deg2rad(angle)

        # Effective cycles per pixel (pixel = actuator)
        # Projected D_pup along the given angle
        D_pup_proj = np.sqrt((D_pup_x * np.cos(theta))**2 + (D_pup_y * np.sin(theta))**2)
        period_pix = D_pup_proj / sep_lamD

        # Spatial coordinate along cosine axis
        coord_along_pattern = x_shift * np.cos(theta) + y_shift * np.sin(theta)

        # Generate cosine phase pattern
        amplitude_rad = 2 * np.sqrt(contrast)
        phase_pattern = amplitude_rad * np.cos(2 * np.pi * coord_along_pattern / period_pix)

        # Convert phase to stroke in meters
        stroke_m = (phase_pattern * wavelength_m) / (4 * np.pi)
        
        # Convert stroke to volts
        stroke_V = stroke_m / (gain_nm_per_V * 1e-9)

        if sign == "negative":
            stroke_V = -stroke_V

        # Add to original DM map (in volts)
        dm_volts_with_pattern += stroke_V

    return dm_volts_with_pattern
