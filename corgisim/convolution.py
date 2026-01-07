#This file contains the functions to generate PRFs and convolve them with a scene
#Written by DevonTax Grace at UCSB Summer 2024 - Based on notebooks by Kian Milani
#It might be incorporated into instrument.py

import numpy as np
from scipy.signal import fftconvolve
import astropy.io.fits as fits
import astropy.units as u
from astropy.io import fits
import corgisim
from synphot import units, SourceSpectrum, SpectralElement, Observation
from corgisim import outputs, spec, prf_simulation, constants
from corgisim.scene import SimulatedImage
from scipy import interpolate
import astropy.units as u

def binning(img: np.ndarray, binning_factor: int) -> np.ndarray:
    """
    Bin a 2D image by an integer factor, conserving total flux (sum over bins).

    The function trims extra rows/columns if the image dimensions are not
    divisible by the binning factor, then reshapes the image into blocks of
    size (binning_factor x binning_factor) and sums each block.

    Args:
        img (np.ndarray): Input 2D image array.
        binning_factor (int): Binning factor (block size in pixels). Must be > 0.

    Returns:
        np.ndarray: Binned 2D image with shape
        (img.shape[0] // binning_factor, img.shape[1] // binning_factor).

    Raises:
        ValueError: If `binning_factor` is not a positive integer.

    Notes:
        - The output is flux-conserving (sums within bins).
        - To get block averages instead of sums, divide the result by
          ``binning_factor ** 2``.
    """
    if not isinstance(binning_factor, int) or binning_factor <= 0:
        raise ValueError("binning_factor must be a positive integer")

    # Resize array by getting rid of extra columns and rows
    xedge = np.shape(img)[0] % binning_factor
    yedge = np.shape(img)[1] % binning_factor
    img = img[xedge:, yedge:]

    # Reshape image to new size
    binim = np.reshape(img, (int(np.shape(img)[0] // binning_factor), binning_factor,
                              int(np.shape(img)[1] // binning_factor), binning_factor))

    # Sum each bin x bin subarray
    binim = np.sum(binim, axis=3)
    binim = np.sum(binim, axis=1)

    return binim

def resize_array (array: np.ndarray, size:int, cent = None) -> np.ndarray:
    """
    Resize the array to a given size. "cent" is optional. 

    Args:
        array (np.ndarray): 
        size (int): 
        cent (tuple, optional): Centred coordinate of the array. 
            If defaults to None, cent equals to the centre of the array.

    Returns:
        Input Dimension == 3: 
            resize_cube: [description]
        Input Dimension == 2: 
            resize_array: [description]

    """
    if array.ndim == 3: 
        if cent == None:
            cent = [np.shape(array[0])[0]//2, np.shape(array[0])[1]//2]
            print(f'Centre of the image: %f' %(cent))
        
        _len = np.shape(array)[0]
        resize_cube = np.zeros((_len,size, size))
        
        for i in range (_len):
            resize_cube[i] = array[i][cent[0]- size//2:cent[0] +size//2,\
                                      cent[0] - size//2:cent[0] +size//2]
    
        return resize_cube
    
    else:
        if cent == None:
            cent = [np.shape(array)[0]//2, np.shape(array)[1]//2]
        # print(cent[0]- size//2,cent[1]+size//2,cent[0] - size//2,cent[0] +size//2)        

        resize_array = array[cent[1] - size//2:cent[1] +size//2, \
                             cent[0] - size//2:cent[0] +size//2]

        return resize_array

def build_radial_grid(iwa, owa, inner_step, mid_step, outer_step, max_radius=None):
    """
    Build a combined radial grid in units of λ/D.

    Parameters
    ----------
    iwa : float
        Inner working angle in λ/D.
    owa : float
        Outer working angle in λ/D.
    inner_step : float
        Step size in λ/D for radii ≤ `iwa`.
    mid_step : float
        Step size in λ/D between `iwa` and `owa`.
    outer_step : float
        Step size in λ/D for radii ≥ `owa`.
    max_radius : float, optional
        Maximum radius to extend the outer grid, in λ/D.
        Defaults to `max(15, 1.5 * owa)`.
    output_param : bool
        Returns the input parameters as well (for logging).

    Returns
    -------
    numpy.ndarray
        1D array of concatenated radial positions in λ/D.
    """
    if iwa < 0 or owa < 0:
        raise ValueError("IWA and OWA must both be non-negative")
    if inner_step < 0 or mid_step < 0 or outer_step < 0:
        raise ValueError("All step sizes must be strictly positive")
    if iwa >= owa:
        raise ValueError("IWA must be less than OWA")

    # TODO - check this value. Is this line needed?
    if max_radius is None:
        max_radius = max(15, owa*1.5)

    # Capture parameters
    param = {
        'iwa': iwa,
        'owa': owa,
        'inner_step': inner_step,
        'mid_step': mid_step,
        'outer_step': outer_step,
        'max_radius': max_radius
    }
    
    inner = np.arange(0, iwa + inner_step, inner_step)
    mid   = np.arange(iwa + inner_step, owa, mid_step)
    outer = np.arange(owa, max_radius + outer_step, outer_step)
    
    return np.hstack([inner, mid, outer]), param

def build_azimuth_grid(step_deg):
    """
    Generate an azimuthal grid from 0 (inclusive) to 360° (exclusive).

    Parameters
    ----------
    step_deg : float
        Step size in degrees.

    Returns
    -------
    astropy.units.Quantity
        1D array of angles from 0° up to (but not including) 360°, in Degree units.
    """
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")
    elif 360 % step_deg != 0:
        raise ValueError("step_deg must divide 360 evenly")

    param = {
        'step_deg': step_deg
    }
    
    # Generate angles from 0 to 360° with the specified step
    return np.arange(0, 360, step_deg) * u.deg, param

def get_valid_polar_positions(radii_lamD, azimuths_deg):
    """
    Generate valid polar‐coordinate pairs for PRF sampling.

    Excludes any positions at radius=0 with non‐zero angle.

    Parameters
    ----------
    radii_lamD : array‐like
        1D array of radial positions in λ/D.
    azimuths_deg : array‐like
        1D array of azimuthal angles in degrees.

    Returns
    -------
    list of tuple
        List of `(radius, angle)` pairs for each valid sampling point.
        `radius` is a float in λ/D, `angle` is an astropy Quantity in degrees.
    """
    azimuths_deg = u.Quantity(azimuths_deg, u.deg)
    radius_grid, azimuth_grid = np.meshgrid(radii_lamD, azimuths_deg, indexing="ij")
    
    # Filter out invalid positions: radius=0 with non-zero angle
    valid_mask = ~((radius_grid == 0) & (azimuth_grid != 0 * u.deg))
    valid_radii = radius_grid.ravel()[valid_mask.ravel()]
    valid_azimuths = azimuth_grid.ravel()[valid_mask.ravel()]
    
    return list(zip(valid_radii, valid_azimuths))

def nearest_id_map(r_lamD, theta_deg, radii_lamD, azimuths_deg):
    """
    Map each pixel to the index of its nearest‐neighbour PRF slice.

    Parameters
    ----------
    r_lamD : ndarray
        Radial distances of each pixel in λ/D.
    theta_deg : ndarray
        Azimuthal angles of each pixel in degrees.
    radii_lamD : array‐like
        Grid of available radial positions in λ/D.
    azimuths_deg : array‐like
        Grid of available azimuthal angles in degrees.

    Returns
    -------
    ndarray of int
        Array of the same shape as `r_lamD` containing indices into the 
        flattened PRF cube (first varying radius, then azimuth).
    """

    radial_ids = np.digitize(r_lamD, radii_lamD)
    radial_ids = np.clip(radial_ids, 1, len(radii_lamD) - 1) - 1  

    # Find nearest azimuthal bin (assuming uniform spacing)
    azimuth_step = 360.0 / len(azimuths_deg)
    azimuth_ids = ((theta_deg / azimuth_step).astype(int) % len(azimuths_deg))

    # Special case: r=0 maps to index 0 regardless of azimuth
    prf_ids = np.where(
        radial_ids == 0,
        0,  # On-axis PRF
        1 + (radial_ids - 1) * len(azimuths_deg) + azimuth_ids  # Off-axis PRFs
    )
    return prf_ids

def bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg):
    """
    Compute bilinear‐interpolation indices and weights for PRF sampling.

    Determines, for each pixel, the four surrounding PRF indices and their
    interpolation weights (α, β).
    
    α: radial weight (fraction between radial_id_low and radial_id_high)
    β: azimuthal weight (fraction between theta_id_low and theta_id_high)

    Parameters
    ----------
    r_lamD : ndarray
        Radial distances of each pixel in λ/D.
    theta_deg : ndarray
        Azimuthal angles of each pixel in degrees.
    radii_lamD : array‐like
        Grid of available radial positions in λ/D.
    azimuths_deg : array‐like
        Grid of available azimuthal angles in degrees.

    Returns
    -------
    indices : tuple of ndarray
        Four integer arrays `(k00, k10, k01, k11)` giving PRF‐cube slice
        indices for the four surrounding grid points.
    weights : tuple of ndarray
        Four float arrays `(w00, w10, w01, w11)` giving the bilinear weights 
        for each corresponding index.
    """

    n_azimuth = len(azimuths_deg)
    step   = 360 / n_azimuth

    # Find the nearest azimuthal points and weights
    theta_id_low  = (theta_deg // step).astype(int) % n_azimuth
    theta_id_high  = (theta_id_low + 1) % n_azimuth
    beta   = (theta_deg - theta_id_low * step) / step

    # Find radial points and weights
    radial_id_high = np.digitize(r_lamD, radii_lamD).clip(1, len(radii_lamD) - 1)
    radial_id_low  = radial_id_high - 1

    # Find radial weights (handle identical radii)
    dr     = radii_lamD[radial_id_high] - radii_lamD[radial_id_low]
    alpha  = np.divide(r_lamD - radii_lamD[radial_id_low], dr,
                       out=np.zeros_like(r_lamD), where=dr != 0)

    # Convert to flat PRF indices (same logic as nearest_id_map)
    def grid_to_flat_index(r_idx, theta_idx):
        return np.where(r_idx == 0, 0, 1 + (r_idx - 1) * n_azimuth + theta_idx)

    # Four corners of the bilinear interpolation
    id_00 = grid_to_flat_index(radial_id_low, theta_id_low) # radial_id_low, theta_id_low
    id_10 = grid_to_flat_index(radial_id_high, theta_id_low) # radial_id_high, theta_id_low
    id_01 = grid_to_flat_index(radial_id_low, theta_id_high) # radial_id_low, theta_id_high
    id_11 = grid_to_flat_index(radial_id_high, theta_id_high) # radial_id_high, theta_id_high

    w00 = (1 - alpha) * (1 - beta)
    w10 = alpha * (1 - beta)
    w01 = (1 - alpha) * beta
    w11 = alpha * beta

    return (id_00, id_10, id_01, id_11), (w00, w10, w01, w11)

def pixel_to_polar(img_shape, pix_scale_mas, res_mas):
    """
    Convert image‐grid pixel indices to polar coordinates.

    Computes the radial distance in λ/D and the azimuthal angle for each pixel
    relative to the image centre.

    Parameters
    ----------
    img_shape : tuple of int
        The shape of the image `(ny, nx)`.
    pix_scale_mas : float
        Detector pixel scale in milliarcseconds per pixel.
    res_mas : float
        Scale factor: 1 λ/D in milliarcseconds.

    Returns
    -------
    r_lamD : ndarray
        Radial distance of each pixel from centre, in λ/D.
    azimuth_deg : ndarray
        Azimuthal angle of each pixel in degrees (0° ≤ θ < 360°).
    """
    # technically this should be called from optics, hence this should always pass
    if pix_scale_mas <= 0 or res_mas <= 0:
        raise ValueError("pix_scale_mas and res_mas must be positive")
    
    ny, nx = img_shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices(img_shape)
    
    radii_pixels  = np.hypot(xx - cx, yy - cy)
    radii_lamD = (radii_pixels * pix_scale_mas) / res_mas

    azimuth_deg  = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360

    return radii_lamD, azimuth_deg

def convolve_with_prfs(obj, prfs_array, radii_lamD, azimuths_deg,
                       pix_scale_mas, res_mas, use_bilinear_interpolation=False):
    """
    Perform a field‐dependent PSF convolution. 

    Applies either nearest‐neighbour or bilinear interpolation of off‐axis PRFs. 
    All PRFs are assumed to be centred in their arrays.

    Parameters
    ----------
    obj : ndarray
        Input 2D image array to be convolved (e.g. scene background).
    prfs_array : ndarray
        PRF cube of shape (N_prf, prf_height, prf_width).
    radii_lamD : array‐like
        Radial sampling nodes in λ/D (must include 0).
    azimuths_deg : astropy.units.Quantity
        Azimuthal sampling nodes in degrees (0° ≤ θ < 360°).
    pix_scale_mas : float
        Detector pixel scale in milliarcseconds per pixel.
    res_mas : float
        Conversion factor: 1 λ/D → milliarcseconds.
    use_bilinear_interpolation : bool, optional
        If True, use bilinear (4‐PRF) interpolation; otherwise use nearest‐neighbour.
        Default is False.

    Returns
    -------
    conv : ndarray
        Convolved (blurred) 2D image, same shape as `obj`.
    """

    r_lamD, theta_deg = pixel_to_polar(obj.shape, pix_scale_mas, res_mas)
    prfs_resized = resize_prf_cube(prfs_array, obj.shape)
    conv = np.zeros_like(obj, dtype=float)
    
    if not use_bilinear_interpolation:
        # Nearest-neighbor convolution
        prf_ids = nearest_id_map(r_lamD, theta_deg, radii_lamD, azimuths_deg)
        
        for prf_idx in np.unique(prf_ids):
            mask = (prf_ids == prf_idx)
            if np.any(mask):
                weighted_scene = obj * mask
                conv += fftconvolve(weighted_scene, prfs_resized[prf_idx], mode="same")
    
    else:
        # Bilinear interpolation convolution
        indices, weights = bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg)
        idx_00, idx_10, idx_01, idx_11 = indices
        w_00, w_10, w_01, w_11 = weights
        
        all_indices = np.concatenate([
            idx_00.ravel(), idx_10.ravel(), idx_01.ravel(), idx_11.ravel()
        ])
        unique_prfs = np.unique(all_indices)
        
        for prf_idx in unique_prfs:
            weight_map = (
                np.where(idx_00 == prf_idx, w_00, 0.0) +
                np.where(idx_10 == prf_idx, w_10, 0.0) +
                np.where(idx_01 == prf_idx, w_01, 0.0) +
                np.where(idx_11 == prf_idx, w_11, 0.0)
            )
            
            if np.any(weight_map > 0):
                weighted_scene = obj * weight_map
                conv += fftconvolve(weighted_scene, prfs_resized[prf_idx], mode="same")
    
    return conv

#TODO - add the fourier shift in here!