import numpy as np

def add_cos_pattern_dm(dm_volts, sep_lamD=6, angle_deg=[0,90], contrast=1e-6, wavelength_m=0.575e-6):
    """
    Add 2D cosine phase pattern(s) to Roman CGI DM solution (in volts).

    Parameters:
        dm_volts: 2D numpy array (original DM in volts)
        sep_lamD: float, number of wave cycles across the pupil diameter
        angle_deg: int, float, or a list of ints/floats, orientation angle (degrees, 0 = same position angle as the input DM solution)
        contrast: float, expected contrast of sattelite spots, assuming amplitude_rad = 2*sqrt(contrast), where amplitude is a phase amplitude in radians
        wavelength_m: float, wavelength in meters

    Returns:
        dm_volts_with_pattern: 2D numpy array (in volts), updated DM map with added cosine patterns
    """

    gain_nm_per_V = 3.6 # nm/V, rough value provided by John Krist
    # actuator_pitch_m = 0.9906e-3  # actuator spacing in meters

    # Roman DM properties - hardcoded
    D_pup_x = 47.41  # in actuators
    D_pup_y = 46.74  # in actuators

    ny, nx = dm_volts.shape
    y, x = np.indices((ny, nx))
    x_center, y_center = (nx - 1) / 2, (ny - 1) / 2  # (23.5, 23.5) for 48x48

    # Shift coordinates to center
    x_shift = x - x_center
    y_shift = y - y_center

    # Ensure angle_deg is iterable
    if isinstance(angle_deg, (int, float)):
        angle_list = [angle_deg]
    else:
        angle_list = angle_deg

    # Initialize output map
    dm_volts_with_pattern = dm_volts.copy()

    # Loop through all angles and add their corresponding cosine patterns
    for angle in angle_list:
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

        # Add to original DM map (in volts)
        dm_volts_with_pattern += stroke_V

    return dm_volts_with_pattern
