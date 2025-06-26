import numpy as np
from astropy.io import fits
import json
import os

def get_slit_mask(config, dx_hires_um = 0.1, hires_dim_um = 800, binfac = 50):
    """
    Generate an FSAM slit mask array for spec mode simulations
    
    This function creates a high-resolution slit mask based on the specified slit 
    parameters and then bins it down to an intermediate spatial resolution.
    The function handles slit positioning offsets,
    and proper scaling based on the coronagraph configuration.
    
    Parameters
    ----------
    config : object
        CorgiOptics configuration object containing:
        - cor_type : str
            Coronagraph type ('spc-spec_band3' or other)
        - lamref_um : float
            Reference wavelength of mode in micrometers
        - ref_data_dir : str
            Directory path containing reference data files
        - slit : str
            Name of the slit to use
        - slit_x_offset_mas : float or None
            Slit offset in x-direction in milliarcseconds
        - slit_y_offset_mas : float or None
            Slit offset in y-direction in milliarcseconds
    dx_hires_um : float, optional
        High-resolution pixel size in micrometers (default: 0.1)
    hires_dim_um : float, optional
        High-resolution array dimension in micrometers (default: 800)
    binfac : int, optional  
        Binning factor for downsampling (default: 50)
    
    Returns
    -------
    binned_slit : numpy.ndarray
        2D array representing the FSAM slit transmision, with values between 0 and 1
    dx_binned_m : float
        Spatial sampling of FSAM slit array in meters
        
    Raises
    ------
    Exception
        If the requested slit is not defined in the reference parameter file
    ValueError
        If the binning factor does not evenly divide the image dimensions
        
    Notes
    -----
    - Uses Roman Space Telescope preflight proper model parameters for scaling
    - Slit parameters are loaded from 'FSAM_slit_params.json' reference file
    - The function applies user-specified offsets
    
    Examples
    --------
    >>> binned_mask, pixel_size = get_slit_mask(self, binfac=25)
    >>> print(f"Binned mask shape: {binned_mask.shape}")
    """
    if config.proper_keywords['cor_type'] == 'spc-spec_band2':
        fsam_meter_per_lamD = 1.34273E-5 / (1000 / 2048) # Roman preflight proper model manual
    else:
        fsam_meter_per_lamD = 1.48513E-5 / (1000 / 2048) # Roman preflight proper model manual

    mas_per_lamD = config.lamref_um * 1E-6 * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000    # mas per lambda0/D, defined in roman preflight proper model
    fsam_meter_per_mas = fsam_meter_per_lamD / mas_per_lamD

    slit_ref_param_fname = os.path.join(config.ref_data_dir, 'FSAM_slit_params.json')
    slit_ref_params = read_slit_params(slit_ref_param_fname)
    if config.slit not in slit_ref_params.keys():
        raise Exception('ERROR: Requested slit {:s} is not defined in {:s}'.format(config.slit, slit_ref_param_fname))

    if config.slit_x_offset_mas == None:
        config.slit_x_offset_mas = 0 
    if config.slit_y_offset_mas == None:
        config.slit_x_offset_mas = 0

    (slit_x_offset_um, slit_y_offset_um) = (1E6 * fsam_meter_per_mas * config.slit_x_offset_mas,
                                            1E6 * fsam_meter_per_mas * config.slit_y_offset_mas) 

    dx_binned_um = binfac * dx_hires_um # spatial sampling of binned array in microns
    dx_binned_m = dx_binned_um * 1E-6 # meters

    hires_dimx, hires_dimy = (int(hires_dim_um / dx_hires_um), int(hires_dim_um / dx_hires_um))
    if not (hires_dimy % binfac == 0 and hires_dimx % binfac == 0):
        raise ValueError(f"Binning factor {binfac} does not evenly divide image dimensions. "
                        f"hires_dimy ({hires_dimy}) % binfac = {hires_dimy % binfac}, "
                        f"hires_dimx ({hires_dimx}) % binfac = {hires_dimx % binfac}")

    xc = hires_dimx // 2 - 0.5 + slit_x_offset_um / dx_hires_um
    yc = hires_dimy // 2 - 0.5 + slit_y_offset_um / dx_hires_um

    xs = np.arange(hires_dimx) - xc
    ys = np.arange(hires_dimy) - yc
    XXs, YYs = np.meshgrid(xs, ys)

    slit_width_hires = 1.0 / dx_hires_um * slit_ref_params[config.slit]['width'] 
    slit_height_hires = 1.0 / dx_hires_um * slit_ref_params[config.slit]['height']
    hires_slit = ((np.abs(XXs) < slit_width_hires / 2) & 
                  (np.abs(YYs) < slit_height_hires / 2))
    binned_slit = hires_slit.reshape(hires_dimy // binfac, binfac, 
                                     hires_dimx // binfac, binfac).mean(axis=3).mean(axis=1)

    return binned_slit, dx_binned_m

def apply_prism(_3d_array):
    pass


def read_slit_params(slit_param_filename):
    """
    Loads slit parameters from a JSON file into a Python dictionary.

    Args:
        filename (str): The name of the JSON file.

    Returns:
        dict: A dictionary containing the slit data.
    """
    try:
        with open(slit_param_filename, "r") as f: # Open the file in read mode ('r')
            slit_params = json.load(f) # Load the JSON data and parse it into a dictionary
        return slit_params
    except FileNotFoundError:
        print(f"Error: The file '{slit_param_filename}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{slit_param_filename}'.")
        return None

def read_prism():
    pass

def read_subband():
    pass
