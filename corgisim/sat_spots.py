import numpy as np

def add_cos_pattern_dm(dm_volts, sep_lamD=6, angle_deg=0, contrast=1e-6, wavelength_m=0.575e-6):
    """
    Add a 2D cosine phase pattern to a DM solution (in volts) for generating sat spots.
    
    Parameters:
        dm_volts: 2D numpy array (original DM in volts)
        sep_lamD: float, number of wave cycles across the pupil diameter
        angle_deg: float, orientation angle (degrees, 0 = along x-axis)
        contrast: float, expected contrast of sattelite spots, assuming amplitude_rad = 2*sqrt(contrast), where amplitude is a phase amplitude in radians
        wavelength_m: float, wavelength in meters
    
    Returns:
        dm_volts_with_pattern: DM map with added cosine pattern (still in volts)
    """
    gain_nm_per_V = 3.6 # nm/V
    # actuator_pitch_m = 0.9906e-3  # actuator spacing in meters

    # Roman DM properties
    D_pup_x = 47.41  # in actuators
    D_pup_y = 46.74  # in actuators

    ny, nx = dm_volts.shape
    y, x = np.indices((ny, nx))
    x_center, y_center = (nx - 1) / 2, (ny - 1) / 2  # (23.5, 23.5) for a 48x48 DM solution

    # Shift coordinates to center
    x_shift = x - x_center
    y_shift = y - y_center

    # Convert angle to radians
    theta = np.deg2rad(angle_deg)

    # Effective cycles per pixel (pixel = actuator)
    # Projected D_pup along the given angle
    D_pup_proj = np.sqrt((D_pup_x * np.cos(theta))**2 + (D_pup_y * np.sin(theta))**2)
    period_pix = D_pup_proj / sep_lamD  # pixels = actuators

    # Spatial coordinate along cosine axis
    coord_along_pattern = x_shift * np.cos(theta) + y_shift * np.sin(theta)

    # Generate cosine phase pattern
    amplitude_rad = 2 * np.sqrt(contrast) 
    phase_pattern = amplitude_rad * np.cos(2 * np.pi * coord_along_pattern / period_pix)

    # Convert phase to stroke in meters
    stroke_m = (phase_pattern * wavelength_m) / (4 * np.pi)

    # Convert stroke to volts
    stroke_V = stroke_m / (gain_nm_per_V * 1e-9)

    # Add to original DM map (still in volts)
    dm_volts_with_pattern = dm_volts + stroke_V

    return dm_volts_with_pattern
