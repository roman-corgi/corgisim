### file containing functions and data useful for simulating polarimetric observations
import numpy as np
from scipy import interpolate

def check_stokes_vector_validity(pol_state):
    """
    Check if the input stokes vector is of right length and magnitude

    Args: 
        pol_state (float array): Stokes vector describing the polarization state of a source
    
    Raises:
        ValueError: If the provided stokes vector is not of length 4 or the polarized intensity magnitude exceeds the total intensity magnitude
    """
    #checks length
    if (len(pol_state) != 4): raise ValueError(f'Invalid stokes vector length of {len(pol_state)}. Valid stokes vector is of length 4')

    #checks magnitude
    if (np.sqrt((pol_state[1] ** 2) + (pol_state[2] ** 2) + (pol_state[3] ** 2)) > pol_state[0]):
        raise ValueError(f'Invalid stokes parameters of {pol_state}. Please make sure sum of polarized intensity does not exceed that of total intensity')

def get_instrument_mueller_matrix(lam_band):
    """
    Calculate the average Mueller matrix of the instrument for a given band
    
    Args:
        lam_band (float array): Array containing the sampled wavelengths for a given band, ordered from shortest to longest. 

    Returns:
        mm_passband (4 by 4 float array): Averaged instrument Mueller matrix for the given passband
    """
    # field-independent pupil averaged Mueller matrices modeling instrument polarization
    # 21 indexes, goes from 450nm to 950nm in increments of 25nm
    # https://roman.ipac.caltech.edu/docs/Roman-Coronagraph-Optical-Model-Mueller-Matrices-450-to-950nm.pdf
    matrices = np.array([
        [[ 0.37940,  0.00530,  0.00000,  0.00000],
         [-0.00531, -0.37860, -0.00001, -0.00004],
         [ 0.00000, -0.00003,  0.34429,  0.15764],
         [ 0.00000,  0.00000,  0.15779, -0.34354]],

        [[ 0.50486,  0.00627,  0.00000,  0.00000],
         [-0.00628, -0.50430, -0.00001, -0.00002],
         [ 0.00000, -0.00002,  0.48155,  0.15007],
         [ 0.00000,  0.00000,  0.15015, -0.48101]],

        [[ 0.58203,  0.00653,  0.00000,  0.00000],
         [-0.00654, -0.58172,  0.00000,  0.00000],
         [ 0.00000, -0.00001,  0.57148,  0.10904],
         [ 0.00000,  0.00000,  0.10907, -0.57116]],

        [[ 0.62888,  0.00652,  0.00000,  0.00000],
         [-0.00652, -0.62873,  0.00000,  0.00000],
         [ 0.00000,  0.00000,  0.62630,  0.05527],
         [ 0.00000,  0.00000,  0.05528, -0.62615]],

        [[ 0.65761,  0.00640,  0.00000,  0.00000],
         [-0.00640, -0.65756,  0.00001,  0.00001],
         [ 0.00000,  0.00000,  0.65747, -0.00094],
         [ 0.00000,  0.00000, -0.00094, -0.65741]],

        [[ 0.67450,  0.00623,  0.00000,  0.00000],
         [-0.00623, -0.67448,  0.00001,  0.00001],
         [ 0.00000,  0.00000,  0.67213, -0.05384],
         [ 0.00000,  0.00000, -0.05384, -0.67211]],
        
        [[ 0.68442,  0.00603,  0.00000,  0.00000],
         [-0.00603, -0.68439,  0.00000,  0.00000],
         [ 0.00000,  0.00000,  0.67636, -0.10254],
         [ 0.00000,  0.00000, -0.10255, -0.67632]],

        [[ 0.68932,  0.00580,  0.00000, -0.00001],
         [-0.00580, -0.68923,  0.00000,  0.00002],
         [ 0.00000,  0.00000,  0.67316, -0.14607],
         [ 0.00000,  0.00000, -0.14607, -0.67307]],

        [[ 0.69043,  0.00554,  0.00000, -0.00001],
         [-0.00554, -0.69028, -0.00001,  0.00003],
         [ 0.00000, -0.00001,  0.66482, -0.18387],
         [ 0.00000,  0.00000, -0.18387, -0.66467]],
        
        [[ 0.68866,  0.00524,  0.00000, -0.00001],
         [-0.00524, -0.68842, -0.00001,  0.00003],
         [ 0.00000, -0.00001,  0.65327, -0.21534],
         [ 0.00000,  0.00000, -0.21535, -0.65303]],

        [[ 0.68453,  0.00492,  0.00000, -0.00001],
         [-0.00492, -0.68421, -0.00002,  0.00004],
         [ 0.00000, -0.00001,  0.63981, -0.24068],
         [ 0.00000,  0.00000, -0.24069, -0.63949]],

        [[ 0.67845,  0.00455,  0.00000, -0.00001],
         [-0.00455, -0.67804, -0.00002,  0.00005],
         [ 0.00000, -0.00002,  0.62560, -0.25979],
         [ 0.00000,  0.00000, -0.25980, -0.62519]],

        [[ 0.67071,  0.00416,  0.00000, -0.00001],
         [-0.00416, -0.67023, -0.00002,  0.00006],
         [ 0.00000, -0.00002,  0.61155, -0.27266],
         [ 0.00000,  0.00000, -0.27269, -0.61108]],

        [[ 0.66133,  0.00374,  0.00000, -0.00001],
         [-0.00374, -0.66079, -0.00002,  0.00006],
         [ 0.00000, -0.00003,  0.59823, -0.27921],
         [ 0.00000,  0.00000, -0.27924, -0.59770]],

        [[ 0.65085,  0.00329,  0.00000, -0.00001],
         [-0.00330, -0.65026, -0.00002,  0.00006],
         [ 0.00000, -0.00003,  0.58635, -0.27987],
         [ 0.00000,  0.00000, -0.27991, -0.58578]],

        [[ 0.63956,  0.00284,  0.00000, -0.00001],
         [-0.00284, -0.63895, -0.00002,  0.00006],
         [ 0.00000, -0.00003,  0.57627, -0.27492],
         [ 0.00000,  0.00000, -0.27496, -0.57568]],

        [[ 0.62776,  0.00238,  0.00000, -0.00001],
         [-0.00238, -0.62715, -0.00002,  0.00006],
         [ 0.00000, -0.00003,  0.56815, -0.26471],
         [ 0.00000,  0.00000, -0.26476, -0.56756]],

        [[ 0.61552,  0.00193,  0.00000, -0.00001],
         [-0.00193, -0.61493, -0.00002,  0.00006],
         [ 0.00000, -0.00003,  0.56174, -0.24957],
         [ 0.00000,  0.00000, -0.24962, -0.56116]],

        [[ 0.60343,  0.00150,  0.00000, -0.00001],
         [-0.00151, -0.60288, -0.00002,  0.00005],
         [ 0.00000, -0.00003,  0.55706, -0.23017],
         [ 0.00000,  0.00000, -0.23022, -0.55652]],

        [[ 0.59182,  0.00111,  0.00000, -0.00001],
         [-0.00112, -0.59132, -0.00002,  0.00004],
         [ 0.00000, -0.00003,  0.55380, -0.20715],
         [ 0.00000,  0.00000, -0.20719, -0.55332]],

        [[ 0.58101,  0.00077,  0.00000,  0.00000],
         [-0.00077, -0.58057, -0.00001,  0.00003],
         [ 0.00000, -0.00002,  0.55160, -0.18123],
         [ 0.00000,  0.00000, -0.18127, -0.55117]]
    ])
    
    #wavelength increments corresponding to the instrument Mueller matrix data
    lam_mm = np.array([0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 
                 0.675, 0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 
                 0.900, 0.925, 0.950])
    
    mm_passband = np.zeros((4, 4), dtype=float)
    # find interpolated value at a given wavelength in the passband for each MM coefficient at i,j
    # averages each MM coefficient at i,j over all wavelengths in the passband to obtain final MM
    for i in range(4):
        for j in range(4):
            mm_interp = interpolate.interp1d(lam_mm, matrices[:,i,j])
            mm_coefficients_i_j = mm_interp(lam_band)
            mm_passband[i, j] = np.mean(mm_coefficients_i_j)
    
    return mm_passband

def get_wollaston_mueller_matrix(angle):
    """
    Calculate the Mueller matrix response for one of the two orthogonal polarization directions the light is split into, can
    be treated like a linear polarizer oriented at a certain angle. Used to calculate polarization for point sources

    Args:
        angle (float): The linear polarization angle of transmission in degrees. Default for the wollaston prisms is 0/90 degrees
        for the prism oriented at 0 degrees and 45/135 degrees for second prism oriented at 45 degrees.

    Returns:
        The 4x4 Mueller matrix describing the transformation from light going into the wollaston to one of the two pathes the light is split into
    """
    theta = angle * (np.pi / 180) * 2
    return 0.45 * np.array([[1, np.cos(theta), np.sin(theta), 0],
                   [np.cos(theta), (np.cos(theta)) ** 2, (np.cos(theta)) * (np.sin(theta)), 0],
                   [np.sin(theta), (np.cos(theta)) * (np.sin(theta)), (np.sin(theta)) ** 2, 0],
                   [0, 0, 0, 0]])