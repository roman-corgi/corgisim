import proper
import numpy as np
import roman_preflight_proper
import corgisim
from corgisim import outputs
import copy
import os
import astropy.units as u
import corgisim.convolution as conv
from astropy.io import fits

def _get_prf_sim_info(prf_path): 
    """
    Extract the FITS header from a PRF cube file.

    Args: 
        prf_path (str): Path to the PRF cube FITS file.
    Returns:
        dict: dictionary of 
    """
    hdu = fits.open(prf_path)[0]
    sim_info_out = {}

    for line in hdu.header.get('COMMENT', []):
        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        value = value.strip()

        try:
            val = float(value)
        except ValueError:
            val = value

        sim_info_out[key.strip()] = val
    
    return sim_info_out

def _generate_prf_dictionary(radial_param, azimuthal_param, dm_solution) -> dict:
    """
    Collect metadata associated with the PRF cube, including:
    - Sampling of the PRFs in lam/D 
    - Optics keywords used to generate the PRFs
    - DM solution information
    
    Args:
        radial_param (dict): Dictionary of radial grid parameters.
        azimuthal_param (dict): Dictionary of azimuthal grid parameters.
        dm_solution (any): Initial DM solution information.
        
    Returns: 
        dict: Combined dictionary of all parameters, excluding large arrays
    """    
    # Combine all dictionaries
    combined_params = {
        **radial_param,
        **azimuthal_param,
        'dm_solution': dm_solution
    }
    combined_params['unit'] = '???' # Placeholder for unit information
    
    return combined_params

def resize_prf_cube(prf_cube, target_scene_shape):
    """
    Centre‐crop or pad a PRF cube to match scene dimensions.

    Ensures the PSF centre remains aligned by handling both even and odd
    dimension differences, padding with zeros or trimming as required.

    Parameters
    ----------
    prf_cube : ndarray, shape (n_prfs, prf_height, prf_width)
        Input cube of PRF images.
    target_scene_shape : tuple of int
        Desired output shape `(target_height, target_width)`.

    Returns
    -------
    ndarray, shape (n_prfs, target_height, target_width)
        PRF cube resized to `target_scene_shape`, centred via padding
        or cropping.
    """
    target_height, target_width = target_scene_shape
    n_prfs, prf_height, prf_width = prf_cube.shape
    
    # Calculate true centers
    prf_center_y = (prf_height - 1) / 2.0
    prf_center_x = (prf_width - 1) / 2.0
    target_center_y = (target_height - 1) / 2.0
    target_center_x = (target_width - 1) / 2.0
    
    # Calculate shifts needed to align centers
    shift_y = int(round(target_center_y - prf_center_y))
    shift_x = int(round(target_center_x - prf_center_x))
    
    # Determine crop/pad regions for height
    if shift_y >= 0:
        pad_top = shift_y
        pad_bottom = target_height - prf_height - pad_top
        crop_start_y, crop_end_y = 0, prf_height
    else:
        pad_top = 0
        pad_bottom = max(0, target_height - prf_height + abs(shift_y))
        crop_start_y = abs(shift_y)
        crop_end_y = min(prf_height, crop_start_y + target_height)
    
    # Determine crop/pad regions for width
    if shift_x >= 0:
        pad_left = shift_x
        pad_right = target_width - prf_width - pad_left
        crop_start_x, crop_end_x = 0, prf_width
    else:
        pad_left = 0
        pad_right = max(0, target_width - prf_width + abs(shift_x))
        crop_start_x = abs(shift_x)
        crop_end_x = min(prf_width, crop_start_x + target_width)
    
    # Apply cropping
    cropped_cube = prf_cube[:, crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    
    # Apply padding
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded_cube = np.pad(cropped_cube, 
                           [(0, 0), (max(0, pad_top), max(0, pad_bottom)), 
                            (max(0, pad_left), max(0, pad_right))], 
                           mode="constant", constant_values=0)
    else:
        padded_cube = cropped_cube
    
    # Final size check and correction
    current_height, current_width = padded_cube.shape[1], padded_cube.shape[2]
    if current_height != target_height or current_width != target_width:
        final_cube = np.zeros((n_prfs, target_height, target_width), dtype=prf_cube.dtype)
        copy_h = min(current_height, target_height)
        copy_w = min(current_width, target_width)
        final_cube[:, :copy_h, :copy_w] = padded_cube[:, :copy_h, :copy_w]
        return final_cube
    
    return padded_cube

def create_wavelength_grid_and_weights(wvl_um, source_sed):
    """
    Build a wavelength grid in microns and normalised spectral weights.

    If `source_sed` is None, a flat spectrum (equal weights) is assumed.

    Parameters
    ----------
    wvl_um : float or array‐like
        Single wavelength or list of wavelengths in microns.
    source_sed : array‐like or None
        Spectral energy distribution weights for each wavelength.
        If None, uses a flat spectrum (equal weight per wavelength).

    Returns
    -------
    lam_grid : numpy.ndarray
        Array of wavelengths (float) in microns.
    lam_wts : numpy.ndarray
        Normalised weights (sum to 1) corresponding to each entry in `lam_grid`.

    Raises
    ------
    ValueError
        If `source_sed` is provided but its length does not match `lam_grid`.
    """
    # Convert to array (handles both single values and lists)
    lam_grid = np.atleast_1d(u.Quantity(wvl_um, u.micron).value)
    n_lambda = len(lam_grid)
    
    # Handle spectral weights
    # TODO - not fully implemented. For now it is just checking the basic: non zero and total not zero    
    if source_sed is None:
        lam_wts = np.ones(n_lambda) / n_lambda

    else:
        source_sed = np.asarray(source_sed, float)
        if (source_sed < 0).any():
            raise ValueError("source_sed must not contain negative values")

        total = source_sed.sum()
        if total <= 0:
            raise ValueError("source_sed weights must sum to a positive value")

        if source_sed.size != n_lambda:
            raise ValueError(
                f"source_sed length ({source_sed.size}) must equal "
                f"wavelength grid length ({n_lambda})"
            )

        lam_wts = source_sed / total

    return lam_grid, lam_wts

def compute_single_off_axis_psf(optics, radius_lamD, azimuth_angle, 
                                wavelength_grid, wavelength_weights, verbose=False):
    """
    Compute a weighted off-axis PSF at a given polar position (r, θ).

    Parameters
    ----------
    radius_lamD : float
        Off-axis distance in λ/D.
    azimuth_deg : float
        Off-axis angle in degrees.
    lam_grid : ndarray
        Wavelengths to simulate (in microns).
    lam_weights : ndarray
        Normalised weights corresponding to each wavelength.

    Returns
    -------
    weighted_psf : ndarray
        2D PSF image, weighted across the spectral band.
    """

    # Convert polar to Cartesian coordinates
    dx = optics.res_mas * radius_lamD * np.cos(azimuth_angle.to_value(u.rad))
    dy = optics.res_mas * radius_lamD * np.sin(azimuth_angle.to_value(u.rad))

    grid_dim_out_tem = optics.grid_dim_out * optics.oversampling_factor
    sampling_um_tem = optics.sampling_um / optics.oversampling_factor

    # verbose for printing debug information
    if verbose:
        print(f"Computing off-axis PSF at r={radius_lamD:.2f} λ/D, θ={azimuth_angle:.1f} deg "
                f"-> dx={dx:.2f} mas, dy={dy:.2f} mas")

    optics_keywords_comp = optics.optics_keywords.copy()
    optics_keywords_comp.update({'output_dim': grid_dim_out_tem,
                                    'final_sampling_m': sampling_um_tem * 1e-6,
                                    'source_x_offset_mas': dx,
                                    'source_y_offset_mas': dy})
    
    # Run PROPER simulation
    (fields, sampling) = proper.prop_run_multi('roman_preflight', wavelength_grid, 1024,PASSVALUE= optics_keywords_comp, QUIET=optics.quiet)

    # Apply spectral weighting and bin down
    intensity = np.abs(fields)**2
    weighted_img = np.tensordot(wavelength_weights, intensity, axes=(0, 0))

    # TODO - Currently we skipped counts when calculating off-axis PSF for convolution
    # Bin down to detector resolution
    binned = weighted_img.reshape(
        (optics.grid_dim_out, optics.oversampling_factor,
            optics.grid_dim_out, optics.oversampling_factor)
    ).mean(3).mean(1) * optics.oversampling_factor**2

    return binned

def make_prf_cube(optics, radii_lamD, azimuths_deg, prf_dict, source_sed=None, output_dir=None):
    """
    Build a psf cube by evaluating the off-axis PSF at specified polar positions.

    Parameters
    ----------
    radii_lamD : array-like
        Radial offsets in λ/D.
    azimuths_deg : array-like
        Azimuthal angles in degrees.
    source_sed : array-like or None
        Spectral energy distribution weights, optional.

    Returns
    -------
    prf_cube_hdu : hdu object 
        ndarray, shape (N_positions, Ny, Nx)
        Cube of PSFs at all requested (r, θ) positions.
    """
    wavelength_grid, wavelength_weights = create_wavelength_grid_and_weights(optics.lam_um, source_sed)
    valid_positions = conv.get_valid_polar_positions(radii_lamD, azimuths_deg)   

    num_positions = len(valid_positions)
    prf_cube = np.empty((num_positions, optics.grid_dim_out, optics.grid_dim_out), dtype=np.float32)

    show_progress = num_positions > 50  # Show progress bar for larger jobs

    for i, (radius_lamD, azimuth_angle) in enumerate(valid_positions):
        prf_cube[i] = compute_single_off_axis_psf(radius_lamD, azimuth_angle, wavelength_grid, wavelength_weights)

        if show_progress:  # Only show progress for larger jobs
            progress = (i + 1) / num_positions
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r[{bar}] {i+1}/{num_positions} ({progress:.1%})', end='', flush=True)

    if show_progress:
        print()  # New line after progress bar

    prf_fname = 'prf_cube' + '_'+ optics.cgi_mode + '_'+ optics.cor_type + '_band_' + optics.bandpass + '.fits'
    prf_cube_hdu = outputs.create_hdu(prf_cube, sim_info=prf_dict)
    
    if output_dir is None:
        output_dir = './'

    prf_cube_hdu.writeto(os.path.join(output_dir, prf_fname), overwrite=False)
    print(f"PRF cube is saved to {os.path.join(output_dir, prf_fname)}")
    
    return prf_cube_hdu