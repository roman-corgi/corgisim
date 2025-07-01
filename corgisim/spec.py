import pickle
import numpy as np
from astropy.io import fits
import json
import os
from pathlib import Path
import scipy.interpolate

def get_slit_mask(optics, dx_fsam_um = 10.0, hires_dim_um = 800, binfac = 50):
    """
    Generate an FSAM slit mask array for spec mode simulations
    
    This function creates a high-resolution slit mask based on the specified aperture
    parameters and then bins it down to an intermediate spatial resolution.
    The function handles slit positioning offsets,
    and proper scaling based on the coronagraph configuration.
    
    Parameters
    ----------
    optics : object
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
    dx_fsam_um : float, optional
        FSAM slit array spatial sampling in micrometers (default: 10.0)
    hires_dim_um : float, optional
        High-resolution array dimension in micrometers (default: 800)
    binfac : int, optional  
        Binning factor for downsampling (default: 50)
    
    Returns
    -------
    binned_slit : numpy.ndarray
        2D array representing the FSAM slit transmision, with values between 0 and 1
    dx_fsam_m : float
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
    if not hires_dim_um % dx_fsam_um == 0:
        raise ValueError(f"The spatial dimension of the binned FSAM array, hires_dim_um={hires_dim_um} microns, is not a whole number ratio to the binned spatial sampling scale, dx_fsam_um={dx_fsam_um} microns.")

    if optics.proper_keywords['cor_type'] == 'spc-spec_band2':
        fsam_meter_per_lamD = 1.34273E-5 / (1000 / 2048) # Roman preflight proper model manual
    else:
        fsam_meter_per_lamD = 1.48513E-5 / (1000 / 2048) # Roman preflight proper model manual

    mas_per_lamD = optics.lamref_um * 1E-6 * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000    # mas per lambda0/D, defined in roman preflight proper model
    fsam_meter_per_mas = fsam_meter_per_lamD / mas_per_lamD

    slit_ref_params = read_slit_params(optics.slit_param_fname)
    if optics.slit not in slit_ref_params.keys():
        raise Exception('ERROR: Requested slit {:s} is not defined in {:s}'.format(optics.slit, optics.slit_param_fname))

    if optics.slit_x_offset_mas == None:
        optics.slit_x_offset_mas = 0 
    if optics.slit_y_offset_mas == None:
        optics.slit_y_offset_mas = 0

    (slit_x_offset_um, slit_y_offset_um) = (1E6 * fsam_meter_per_mas * optics.slit_x_offset_mas,
                                            1E6 * fsam_meter_per_mas * optics.slit_y_offset_mas) 

    dx_hires_um = dx_fsam_um / binfac # spatial sampling of binned array in microns
    dx_fsam_m = dx_fsam_um * 1E-6 # meters
    hires_dimx, hires_dimy = (int(hires_dim_um / dx_hires_um), int(hires_dim_um / dx_hires_um))

    if hires_dimx % 2 == 0:
        xc = hires_dimx // 2 - 0.5 + slit_x_offset_um / dx_hires_um
        yc = hires_dimy // 2 - 0.5 + slit_y_offset_um / dx_hires_um
    else:
        xc = hires_dimx // 2 + slit_x_offset_um / dx_hires_um
        yc = hires_dimy // 2 + slit_y_offset_um / dx_hires_um

    xs = np.arange(hires_dimx) - xc
    ys = np.arange(hires_dimy) - yc
    XXs, YYs = np.meshgrid(xs, ys)

    slit_width_hires = 1.0 / dx_hires_um * slit_ref_params[optics.slit]['width'] 
    slit_height_hires = 1.0 / dx_hires_um * slit_ref_params[optics.slit]['height']
    hires_slit = ((np.abs(XXs) < slit_height_hires / 2) & 
                  (np.abs(YYs) < slit_width_hires / 2))
    binned_slit = hires_slit.reshape(hires_dimy // binfac, binfac, 
                                     hires_dimx // binfac, binfac).mean(axis=3).mean(axis=1)
    # Rotate and flip the slit array to match the orientation as applied in the Proper model
    rot_binned_slit = np.fliplr(np.rot90(binned_slit, 3))
    return rot_binned_slit, dx_fsam_m

def get_slit_mask_old(optics, dx_hires_um = 0.1, hires_dim_um = 800, binfac = 50):
    """
    Generate an FSAM slit mask array for spec mode simulations
    
    This function creates a high-resolution slit mask based on the specified aperture
    parameters and then bins it down to an intermediate spatial resolution.
    The function handles slit positioning offsets,
    and proper scaling based on the coronagraph configuration.
    
    Parameters
    ----------
    optics : object
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
    if optics.proper_keywords['cor_type'] == 'spc-spec_band2':
        fsam_meter_per_lamD = 1.34273E-5 / (1000 / 2048) # Roman preflight proper model manual
    else:
        fsam_meter_per_lamD = 1.48513E-5 / (1000 / 2048) # Roman preflight proper model manual

    mas_per_lamD = optics.lamref_um * 1E-6 * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000    # mas per lambda0/D, defined in roman preflight proper model
    fsam_meter_per_mas = fsam_meter_per_lamD / mas_per_lamD

    slit_ref_params = read_slit_params(optics.slit_param_fname)
    if optics.slit not in slit_ref_params.keys():
        raise Exception('ERROR: Requested slit {:s} is not defined in {:s}'.format(optics.slit, optics.slit_param_fname))

    if optics.slit_x_offset_mas == None:
        optics.slit_x_offset_mas = 0 
    if optics.slit_y_offset_mas == None:
        optics.slit_x_offset_mas = 0

    (slit_x_offset_um, slit_y_offset_um) = (1E6 * fsam_meter_per_mas * optics.slit_x_offset_mas,
                                            1E6 * fsam_meter_per_mas * optics.slit_y_offset_mas) 

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

    slit_width_hires = 1.0 / dx_hires_um * slit_ref_params[optics.slit]['width'] 
    slit_height_hires = 1.0 / dx_hires_um * slit_ref_params[optics.slit]['height']
    hires_slit = ((np.abs(XXs) < slit_width_hires / 2) & 
                  (np.abs(YYs) < slit_height_hires / 2))
    binned_slit = hires_slit.reshape(hires_dimy // binfac, binfac, 
                                     hires_dimx // binfac, binfac).mean(axis=3).mean(axis=1)

    return binned_slit, dx_binned_m

def apply_prism(optics, image_cube):
    """
    Apply a prism dispersion model to a multiwavelength image cube from Proper. 
    
    This function shifts each wavelength slice according to a polynomial
    dispersion model and a prism clocking angle loaded from a parameter file.
    The function interpolates the input image cube to a finer wavelength
    sampling before applying dispersion.
    
    Parameters
    ----------
    optics : object
        CorgiOptics configuration object containing:
        - prism_param_fname : str
            Path to the prism parameter file
        - lam_um : array_like
            Wavelength array in microns for the input image cube
        - wav_step_um : float
            Wavelength step size in microns for the interpolated wavelength grid
        - lamref_um : float
            Reference wavelength in microns for the observing mode
        - sampling_um : float
            Image spatial sampling in microns of the CorgiOptics configuration
        - oversampling_factor : int
            Spatial oversampling factor of the CorgiOptics configuration
            
    image_cube : ndarray, shape (n_wavelengths, n_y, n_x)
        Input multi-wavelength intensity cube from Proper simulation
        
    Returns
    -------
    dispersed_cube : ndarray, shape (n_wavelengths_interp, n_y, n_x)
        Image cube after applying prism dispersion, with finer wavelength sampling
    interp_wavs_bandpass : ndarray
        Interpolated wavelength array (in microns) for the dispersed_cube
        
    Notes
    -----
    The function performs the following operations:
    1. Loads prism parameters including clocking angle and dispersion polynomial coefficients
    2. Creates a densely sampled wavelength grid based on optics.wav_step_um
    3. Calculates wavelength-dependent dispersion shifts in model pixels 
    4. Interpolates the input image cube to the finer wavelength grid
    5. Applies 2D spatial shifts to each wavelength slice according to the dispersion model
    6. Stacks the shifted slices into a cube array
    
    The dispersion direction is determined by the prism clocking angle (theta),
    with shifts applied in both x and y directions.
    
    Examples
    --------
    >>> dispersed_cube, wavelengths = apply_prism(optics, image_cube)
    """
    prism_params = read_prism_params(optics.prism_param_fname)
    theta = prism_params['clocking_angle']

    dispersed_slices = []
    # Define a densely sampled wavelength array for the cube of dispersed images
    N_wav_interp = int(round((optics.lam_um[-1] - optics.lam_um[0]) / optics.wav_step_um))
    
    dispersion_polyfunc = np.poly1d(prism_params['pos_vs_wavlen_polycoeff'])
    interp_wavs_bandpass = np.linspace(optics.lam_um[0], optics.lam_um[-1], N_wav_interp)
    delta_wavelen = interp_wavs_bandpass - optics.lamref_um

    model_sampling_mm = optics.sampling_um / optics.oversampling_factor * 1E-3
    dispers_shift_mm = dispersion_polyfunc(delta_wavelen / optics.lamref_um)
    dispers_shift_modelpix = dispers_shift_mm / model_sampling_mm

    # Interpolate the proper image cube
    y_pts = np.arange(image_cube.shape[1])
    x_pts = np.arange(image_cube.shape[2])

    cube_grid = (optics.lam_um, y_pts, x_pts)
    interp_wavs_grid, ypts_grid, xpts_grid = np.meshgrid(interp_wavs_bandpass, y_pts, x_pts, indexing='ij')
    cube_interp_grid = (interp_wavs_grid.ravel(), ypts_grid.ravel(), xpts_grid.ravel())
    # Scale factor to conserve flux after interpolation to the densely sampled wavelength grid  
    flux_conserv_factor = len(optics.lam_um) / len(interp_wavs_bandpass)

    cube_interp_func = scipy.interpolate.RegularGridInterpolator(cube_grid, image_cube)
    # cube_interp_result = cube_interp_func(cube_interp_grid) * flux_conserv_factor
    cube_interp_result = cube_interp_func(cube_interp_grid)
    image_cube_interp = cube_interp_result.reshape(xpts_grid.shape)

    dispersed_cube = np.zeros((image_cube_interp.shape[1], image_cube_interp.shape[2]))
    # dispersed_image = np.zeros((image_cube.shape[1], image_cube.shape[2]))

    for ww, wavelen in enumerate(interp_wavs_bandpass):
        dispersion_shift = (dispers_shift_modelpix[ww] * np.sin(np.deg2rad(theta)),
                            dispers_shift_modelpix[ww] * np.cos(np.deg2rad(theta)))
        # Apply the dispersion shift
        shifted_slice = scipy.ndimage.shift(
            input=image_cube_interp[ww],
            shift=dispersion_shift,
            order=1, mode='constant',
            prefilter=False)
        
        # dispersed_image = dispersed_image + shifted_slice
        dispersed_slices.append(shifted_slice)
 
    # Stack oversamp_image_slices to make an image cube
    dispersed_cube = np.stack(dispersed_slices, axis=0)

    ## Compute the image position corresponding to the filter center wavelength
    # center_lam_um = optics.lam0_um
    # delta_wavelen_filter_center = center_lam_um - optics.lamref_um
    # disp_shift_filter_center_modelpix = dispersion_polyfunc(delta_wavelen_filter_center / optics.lamref_um) / model_sampling_mm
    # disp_shift_filter_center_2d = (-disp_shift_filter_center_modelpix * np.sin(np.deg2rad(theta)),
                                #    -disp_shift_filter_center_modelpix * np.cos(np.deg2rad(theta)))

    # xcent_ovsamp = dispersed_image.shape[1] // 2 + xoff_modelpix - disp_shift_filter_center_2d[0]
    # ycent_ovsamp = dispersed_image.shape[0] // 2 + yoff_modelpix + disp_shift_filter_center_2d[1]
    # # Transform to downsampled coordinates
    # xcent = (xcent_ovsamp + 0.5) / optics.oversampling_factor - 0.5
    # ycent = (ycent_ovsamp + 0.5) / optics.oversampling_factor - 0.5

    # fsam_image = np.zeros((dispersed_cube.shape[0],
                        #    dispersed_cube.shape[1] // optics.oversampling_factor,
                        #    dispersed_cube.shape[2] // optics.oversampling_factor))
    # for ww in range(fsam_image.shape[0]):
        # fsam_image[ww,:,:] = (dispersed_cube[ww,:,:].reshape(
                                # (fsam_image.shape[1], optics.oversampling_factor,
                                # fsam_image.shape[2], optics.oversampling_factor)).mean(3).mean(1) *
                                # optics.oversampling_factor**2)
    return dispersed_cube, interp_wavs_bandpass 

def read_slit_params(slit_param_filename):
    """
    Loads slit aperture reference parameters from a JSON file.

    Args:
        filename (str): The name of the JSON file.

    Returns:
        dict: A dictionary containing the slit data.
    """
    try:
        with open(slit_param_filename, "r") as f:
            slit_params = json.load(f)
        return slit_params
    except FileNotFoundError:
        raise  # This will re-raise the FileNotFoundError
    except json.JSONDecodeError:
        raise  # This will re-raise the JSONDecodeError

def read_prism_params(prism_param_fname):
    """
    Load prism parameters from a NumPy (.npz) file and validate required keys.
    
    This function loads prism dispersion parameters for spectroscopic
    data simulations, including polynomial coefficients for position vs wavelength
    mapping and the prism clocking angle.
    
    Parameters
    ----------
    prism_param_fname : str or Path
        Path to the prism parameter file (.npz format expected)
        
    Returns
    -------
    numpy.lib.npyio.NpzFile
        Dictionary-like object containing prism parameters with keys:
        - 'pos_vs_wavlen_polycoeff' : numpy.ndarray
            Polynomial coefficients for position vs wavelength dispersion
        - 'clocking_angle' : float
            Prism clocking angle in degrees
        - Additional parameters may be present depending on the file
        
    Raises
    ------
    FileNotFoundError
        If the prism parameter file does not exist
    ValueError
        If the file is not a valid NumPy archive or is corrupted
    KeyError
        If required prism parameters are missing from the file
    PermissionError
        If the file exists but cannot be read due to permissions
        
    Examples
    --------
    >>> prism_params = read_prism('prism_data.npz')
    >>> coeffs = prism_params['pos_vs_wavlen_polycoeff']
    >>> angle = prism_params['clocking_angle']
    
    Notes
    -----
    The function expects a NumPy .npz archive file containing at minimum
    the dispersion polynomial coefficients and clocking angle. Additional
    parameters in the file will be preserved and accessible.
    """
    # Convert to Path object for better path handling
    prism_file = Path(prism_param_fname)
    
    # Check if file exists
    if not prism_file.exists():
        raise FileNotFoundError(
            f"Prism parameter file not found: {prism_param_fname}"
        )
    # Check if file is readable
    if not prism_file.is_file():
        raise ValueError(
            f"Path exists but is not a file: {prism_param_fname}"
        )
    
    try:
        # Load the NumPy archive
        prism_params = np.load(prism_param_fname)
    except (OSError, IOError) as e:
        raise PermissionError(
            f"Cannot read prism parameter file '{prism_param_fname}': {e}"
        ) from e
    except (ValueError, pickle.UnpicklingError) as e:
        raise ValueError(
            f"Invalid or corrupted NumPy archive file '{prism_param_fname}': {e}"
        ) from e
    
    # Validate parameters
    required_params = ['pos_vs_wavlen_polycoeff', 'clocking_angle']
    missing_params = []
    
    for param in required_params:
        if param not in prism_params:
            missing_params.append(param)
    
    if missing_params:
        # Close the file if it was opened successfully
        prism_params.close()
        
        missing_str = "', '".join(missing_params)
        raise KeyError(
            f"Prism parameter file '{prism_param_fname}' is missing required "
            f"parameter{'s' if len(missing_params) > 1 else ''}: '{missing_str}'. "
            f"Available parameters: {list(prism_params.keys())}"
        )
    
    return prism_params

def read_subband_filter():
    pass
