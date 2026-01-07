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
    
    # Filter out invalid positions: radius=0 with non-zero angle and (0,0)
    valid_mask = radius_grid > 0.0
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

    # Flat PRF index: first varying radius, then azimuth
    prf_ids = radial_ids * len(azimuths_deg) + azimuth_ids

    return prf_ids

def bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg):
    """
    Compute bilinear‐interpolation indices and weights for PRF sampling.

    For each pixel, this function identifies the four surrounding PRF grid points on a
    polar grid (r, θ) and computes the corresponding weights. These indices and weight are used 
    to smoothly interpolate between neighbouring PRFs for field-dependent convolution.

    The interpolation is separable in radius and azimuth, and follow this form:
        α = (r - r_low) / (r_high - r_low)
        β = (θ - θ_low) / (θ_high - θ_low)

    where (r_low, θ_low) and (r_high, θ_high) are the two neighbouring grid points. 

    The four interpolation weights are then given by: 
        w00 = (1 - α) * (1 - β)
        w10 = α * (1 - β)
        w01 = (1 - α) * β
        w11 = α * β
    
    with (w00, w10, w01, w11) summing to unity for each pixel. 

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
        Four integer arrays `(id00, id10, id01, id11)`. These give the flat PRF-cube indices 
        corresponding to the four surrounding (r, θ) grid points for each pixel.

    weights : tuple of ndarray
        Four float arrays `(w00, w10, w01, w11)` giving the weights associated with each index. 
        For each pixel, the weights sum to 1.

    Notes
    -----
    - This function only computes the geometry-dependent PRF indices and weights. 
    - The PRF cube is assumed to contain only off axis PRFs. 
    - When interpolation collapses to a single radial node (r_low == r_high),
       α is set to zero, corresponding to full weight on the lower PRF.
    """

    # Number of azimuthal sampling points and corresponding angular step
    n_azimuth = len(azimuths_deg)
    step   = 360 / n_azimuth

    # --- Azimuthal indices and weights ---
    # Identify the lower azimuth index for each pixel (wrapped at 360 deg)
    theta_id_low  = (theta_deg // step).astype(int) % n_azimuth
    theta_id_high  = (theta_id_low + 1) % n_azimuth

    # Fractional azimuthal offset between the two azimuth grid points
    beta = (theta_deg - theta_id_low * step) / step

    # --- Radial indices and weights ---
    # Identify the two neighbouring radial grid points by binning. 
    # Indices are clipped to avoid the excluded on-axis PRF. 
    radial_id_high = np.digitize(r_lamD, radii_lamD).clip(1, len(radii_lamD) - 1)
    radial_id_low  = (radial_id_high - 1).clip(1, len(radii_lamD) - 1)

    # Radial spacing between the two neighbouring grid points 
    dr     = radii_lamD[radial_id_high] - radii_lamD[radial_id_low]

    # Fractional radial interpolation weight (alpha); zero when dr=0 but only perform division where dr != 0
    alpha  = np.divide(r_lamD - radii_lamD[radial_id_low], dr,
                       out=np.zeros_like(r_lamD), where=dr != 0)

    # --- Mapping from (r, θ) grid indices to flat PRF cube indices --
    # on-axis PRF is excluded. 
    def grid_to_flat_index(r_idx, theta_idx):
        return (r_idx - 1) * n_azimuth + theta_idx

    # --- Four interpolation corners ---- 
    # (radial_id_low/high, theta_id_low/high)
    id_00 = grid_to_flat_index(radial_id_low, theta_id_low)
    id_10 = grid_to_flat_index(radial_id_high, theta_id_low)
    id_01 = grid_to_flat_index(radial_id_low, theta_id_high) 
    id_11 = grid_to_flat_index(radial_id_high, theta_id_high) 

    # --- Weights --- 
    # For each pixel, the four weights sum to 1. 
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

def _set_2D_image_sim_info(optics, input_scene): 
    # Prepare additional information to be added as COMMENT headers in the primary HDU.
    # These are different from the default L1 headers, but extra comments that are used to track simulation-specific details.
    # TODO - standardize this across different simulation functions?

    sim_info = {'host_star_sptype':input_scene.host_star_sptype,
                'host_star_Vmag':input_scene.host_star_Vmag,
                'host_star_magtype':input_scene.host_star_magtype,
                'ref_flag':input_scene.ref_flag,
                'cgi_mode':optics.cgi_mode,
                'cor_type': optics.optics_keywords['cor_type'],
                'bandpass':optics.bandpass_header,
                'over_sampling_factor':optics.oversampling_factor,
                'return_oversample': optics.return_oversample,
                'output_dim': optics.optics_keywords['output_dim'],
                'nd_filter': optics.nd, 
                'SATSPOTS': optics.SATSPOTS,
                'includ_dectector_noise': 'False'
                }

    # Define specific keys from optics.optics_keywords to include in the header            
    keys_to_include_in_header = ['use_errors','polaxis','final_sampling_m', 'use_dm1','use_dm2','use_fpm',
                        'use_lyot_stop','use_field_stop','fsm_x_offset_mas','fsm_y_offset_mas','slit','prism',
                        'slit_x_offset_mas','slit_y_offset_mas']  # Specify keys to include

    subset = {key: optics.optics_keywords[key] for key in keys_to_include_in_header if key in optics.optics_keywords}
    sim_info.update(subset)

    return sim_info

def _convolve_with_prfs(obj, prfs_array, radii_lamD, azimuths_deg,
                       pix_scale_mas, res_mas, interpolate_prfs=False):
    """
    Apply a field-dependent convolution using a library of off-axis PRFs.

    This function performs a spatially varying convolution in which the
    convolution kernel depends on the pixel location in the focal plane.
    Each pixel is mapped to polar coordinates (r, theta) expressed in
    lambda over D and degrees, and is assigned an off-axis point response
    function (PRF) from a precomputed PRF cube.

    The result is constructed by grouping pixels that share the same PRF
    index (or weighted PRF combination), convolving each weighted sub-scene
    with the corresponding PRF, and summing all contributions.

    Two PRF selection modes are supported:

    - Nearest-neighbour selection (`interpolate_prfs=False`):
      Each pixel uses the single PRF whose (r, theta) sampling node is closest
      to the pixel location. 

    - Bilinear interpolation (`interpolate_prfs=True`):
      Each pixel uses a weighted combination of up to four neighbouring PRFs
      on the (r, theta) sampling grid. Weights vary smoothly with position,
      reducing discontinuities.

    Parameters
    ----------
    obj : ndarray
        Input 2D image array to be convolved.
    prfs_array : ndarray, (N_prf, prf_height, prf_width)
        PRF cube. Each slice contains one off-axis PRF. All PRFs are assumed to be centred in their arrays.
    radii_lamD : array‐like
        Radial sampling nodes in λ/D (must include 0).
    azimuths_deg : astropy.units.Quantity
        Azimuthal sampling nodes in degrees (0° ≤ θ < 360°).
    pix_scale_mas : float
        Detector pixel scale in milliarcseconds per pixel.
    res_mas : float
        Conversion factor: 1 λ/D → milliarcseconds.
    interpolate_prfs : bool, optional
        If True, interpolate between neightbouring PRFs in (r, θ) space 
        using bilinear weights (mixing up to 4 PRFs per pixel). 
        If False, use the nearest-neighbour PRF per pixel. 

        The default is False.
    
    Returns
    -------
    conv : ndarray, shape (H, W)
        Convolved image, same shape as `obj`.

    Notes
    -----
    - The PRF cube is resized internally to match the shape of `obj`
      prior to convolution.
    - The PRFs are assumed to represent the instrument intensity response to a
      unit-flux off-axis point source. Absolute flux scaling must be applied
      separately.
    - Convolution is computed via FFTs and accumulated over the set of PRF
      indices present in the field.
    """

    # Map pixels to polar coordinates (r, theta) in lambda/D and degrees
    r_lamD, theta_deg = pixel_to_polar(obj.shape, pix_scale_mas, res_mas)

    # Resize PRFs to match the shape of the object
    prfs_resized = prf_simulation.resize_prf_cube(prfs_array, obj.shape)

    # Accumulator for the field-dependent convolution result.
    conv = np.zeros_like(obj, dtype=float)
    
    if not interpolate_prfs:
        # Nearest-neighbor 
        # Each pixel is assigned directly to its nearest PRF index 
        prf_ids = nearest_id_map(r_lamD, theta_deg, radii_lamD, azimuths_deg)
        
        # Select PRF indices that are assigned to at least one pixel; each PRF is convolved once
        for prf_idx in np.unique(prf_ids):
            # True for pixels whose field position maps to this PRF index
            mask = (prf_ids == prf_idx) 
            if np.any(mask):
                weighted_scene = obj * mask
                conv += fftconvolve(weighted_scene, prfs_resized[prf_idx], mode="same")
    
    else:
        # Bilinear interpolation convolution
        # each pixel contributes to up to four neighbouring PRFs with position-dependent weights
        # Here the inputs will contain r=0
        indices, weights = bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg)
        
        # PRF indices for the four surrounding (r, θ) grid points at each pixel
        idx_00, idx_10, idx_01, idx_11 = indices
        
        # Corresponding bilinear interpolation weights per pixel (sum to 1 per pixel)
        w_00, w_10, w_01, w_11 = weights
        
        # Collect all PRF indices that appear in any of the four index maps
        # to ensure each contributing PRF is processed only once
        all_indices = np.concatenate([
            idx_00.ravel(), idx_10.ravel(), idx_01.ravel(), idx_11.ravel()
        ])
        unique_prfs = np.unique(all_indices)
        
        for prf_idx in unique_prfs:
            # Build the spatial weight map for this PRF by summing contributions
            # Pixels that do not reference this PRF receive zero weight.
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

def flux_calibration_2D_scene(optics, input_scene, conv2d):
    # NOTE: An attempt to convert flux units to physical units after convolution
    # normalize to the given contrast
    # obs: flux is in unit of photons/s/cm^2/angstrom
    obs = Observation(input_scene.twoD_scene_spectrum, optics.bp)
    counts = np.zeros((optics.lam_um.shape[0]))
    for i in range(optics.lam_um.shape[0]):
        dlam_um = optics.lam_um[1]-optics.lam_um[0]
        lam_um_l = (optics.lam_um[i]- 0.5*dlam_um) * 1e4 ## unit of anstrom
        lam_um_u = (optics.lam_um[i]+ 0.5*dlam_um) * 1e4 ## unit of anstrom
        counts[i] = (optics.polarizer_transmission * obs.countrate(area=optics.area, waverange=[lam_um_l, lam_um_u])).value

    psf_area = np.pi*(optics.res_mas/(constants.PIXEL_SCALE_ARCSEC*1e3)/2)**2 # area of the PSF FWHM in the unit of pixel
    disk_region = (conv2d > 0.5*np.max(conv2d)).sum()  # number of pixs in the disk region (>=50% maximum disk flux) 
    conv2d *= np.sum(counts, axis=0) * disk_region/psf_area   # per resolution element

    return conv2d 

def simulate_2d_scene(optics, input_scene, output_scene=None, interpolate_prfs=False):
    """
    Convolve 2D scene with a pre-computed off-axis PRF cube.

    This function reads a disk model from `input_scene.twoD_scene_info['disk_model_path']`,
    normalizes it, and performs a field-dependent 2D convolution using a precomputed PRF cube
    stored at `input_scene.twoD_scene_info['prf_path']`. The PRF sampling (radii in lambda/D and azimuthal angles)
    is reconstructed from the PRF cube metadata. The convolution can be done using either nearest-neighbor
    or interpolation between PRFs, controlled by the `interpolate_prfs` flag. 

    After convolution, the result is scaled to a count rate integrated over the bandpass defined by 
    `optics.bp` and `intput_scene.twoD_scene_spectrum`. The scaled convolved object is stored in 
    `output_scene.twoD_images` as an HDU with simulation metadata written as FITS COMMENT. 

    Parameters
    ----------
    optics: (corgisim.instrument.CorgiOptics): The optics object defining the
            instrument configuration, including the telescope and coronagraph.
    input_scene : Scene 
        Scene object containing 2D image data to be convolved.
    output_scene: SimulatedImage, optional
        If provided, the convolved image will be stored in this object.
    interpolate_prfs: bool, optional
        Whether to use interpolation between PRFs for convolution.

    Returns
    -------
    output_scene : SimulatedImage
        Output scene with `twoD_image` replaced by the convolved and scaled result.

    Raises
    ------
    ValueError
        If `input_scene.twoD_scene_info['prf_path']` is missing or None.

    Notes
    -----
    - The PRF cube is assumed to be already centred in its arrays. #TODO - add shifting? 
    - The PRF cube is assumed to be normalised to unit input flux. Absolute flux
        scaling is applied in this function using the input scene spectrum and the
        optics bandpass.
    - The disk model is normalised by its total flux prior to convolution.
    - After convolution, the image is scaled using the integrated bandpass count
        rate and converted to a per resolution element normalisation using an
        estimate of the PSF FWHM area (pixels) and a thresholded disk region mask.
    - The output is intended to represent a count rate (photoelectrons per second),
        consistent with `Observation.countrate`.
    """
    # Determine which mode to use based on provided kwargs
    if input_scene.twoD_scene_info['prf_path'] is not None:
        has_prf_cube = True 
    else:
        raise ValueError("No PRF cube path provided in scene.twoD_scene_info")
    
    # input disk model
    disk_model_data = fits.getdata(input_scene.twoD_scene_info['disk_model_path'])
    disk_model_norm = disk_model_data/np.nansum(disk_model_data, axis=(0,1)) # normalisation of the disk

    prf_cube_path = input_scene.twoD_scene_info['prf_path'] 

    prf_sim_info = prf_simulation._get_prf_sim_info(prf_cube_path) # Get the simulation information 

    print(prf_sim_info)
    # Check if PRF cube needs centering

    prf_info_is_centred = prf_sim_info.get('centred') # 'True' or 'False'
    # Convert string to boolean
    is_centred = (prf_info_is_centred == 'True') 

    if not is_centred:
        print("PRF cube is not centred. centring now...")
        from corgisim.prf_simulation import centre_prf_cube
        prf_cube = centre_prf_cube(fits.getdata(prf_cube_path), method='centroid')
    else: 
        prf_cube = fits.getdata(prf_cube_path)

    # 1. Get the radii grids for convolution 
    radii_lamD, _ = build_radial_grid(
        prf_sim_info['iwa'], 
        prf_sim_info['owa'], 
        prf_sim_info['inner_step'], 
        prf_sim_info['mid_step'], 
        prf_sim_info['outer_step'],
        prf_sim_info['max_radius']
    )

    # 2. Get the azimuth grid for convolution
    azimuths_deg, _ = build_azimuth_grid(prf_sim_info['step_deg'])

    # 3. Perform convolution
    conv2d = _convolve_with_prfs(
        obj=disk_model_norm, 
        prfs_array=prf_cube, 
        radii_lamD=radii_lamD , 
        azimuths_deg=azimuths_deg, 
        pix_scale_mas=constants.PIXEL_SCALE_ARCSEC * 1e3, 
        res_mas=optics.res_mas, 
        interpolate_prfs=interpolate_prfs
        )

    # NOTE: An attempt to convert flux units to physical units after convolution
    # 4. Flux calibration
    flux_calibrated_conv2D = flux_calibration_2D_scene(optics, input_scene, conv2d)

    if optics.cgi_mode in ['spec', 'lowfs', 'excam_efield']:
        warnings.warn(f"This mode '{optics.cgi_mode}' has not implmented yet!") # still allow the usage but warn the user about this

    if output_scene is None:
        # No output format was specified - create a new SimulatedImage object
        output_scene = SimulatedImage(input_scene)

    sim_info = _set_2D_image_sim_info(optics, input_scene)

    # Create the HDU object with the generated header information
    output_scene.twoD_image = outputs.create_hdu(flux_calibrated_conv2D, sim_info=sim_info)

    return output_scene
