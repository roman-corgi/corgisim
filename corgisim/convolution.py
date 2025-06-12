#This file contains the functions to generate PRFs and convolve them with a scene
#Written by DevonTax Grace at UCSB Summer 2024 - Based on notebooks by Kian Milani
#It might be incorporated into instrument.py

import numpy as np
from scipy.signal import fftconvolve
import astropy.io.fits as fits
import astropy.units as u
import time
import proper
import sys
proper.prop_use_fftw(DISABLE=False)

# Constants
D_ROMAN = 2.3631 * u.m

def _build_radial_grid(iwa, owa, fine, coarse, outer):
    """Concatenate inner/mid/outer λ/D radii into one 1-D array."""
    inner = np.arange(0, iwa + fine, fine)
    mid   = np.arange(iwa + fine, owa, coarse)
    outer = np.arange(owa, 15 + outer, outer)
    return np.hstack([inner, mid, outer])

def _build_azimuth_grid(step_deg):
    """Return 0 ≤ θ < 360 in constant steps as Quantity[deg]."""
    return np.arange(0, 360, step_deg) * u.deg

def _lam_grid_and_weights(c_lam_um, n_lambda, bandwidth_frac, source_sed):
    """
    Build a wavelength grid in µm and spectral weights.
    Flat spectrum if `source_sed` is None.
    """
    c_lam_um = u.Quantity(c_lam_um, u.micron).value

    if n_lambda == 1 or bandwidth_frac == 0.0:
        lam_grid = np.array([c_lam_um])
        lam_wts  = np.array([1.0])
        return lam_grid, lam_wts

    half     = bandwidth_frac / 2.0
    lam_grid = np.linspace(c_lam_um * (1 - half),
                           c_lam_um * (1 + half),
                           n_lambda)

    if source_sed is None:
        lam_wts = np.ones(n_lambda) / n_lambda
    else:
        lam_wts = np.asarray(source_sed, float)
        if lam_wts.size != n_lambda:
            raise ValueError("source_sed length must equal n_lambda")
        lam_wts /= lam_wts.sum()

    return lam_grid, lam_wts

def _pixel_polar(shape, pix_scale_mas, mas_per_lamD):
    """
    Return arrays (r_lamD, theta_deg) for every pixel of a scene frame.
    """
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices(shape)
    r_pix  = np.hypot(xx - cx, yy - cy)
    r_lamD = (r_pix * pix_scale_mas) / mas_per_lamD
    theta  = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    return r_lamD, theta

def _pad_prf_cube(prf_cube, scene_shape):
    """Centre-pad or crop a PRF cube to the full detector frame."""
    ny, nx = scene_shape
    dy = (ny - prf_cube.shape[1]) // 2
    dx = (nx - prf_cube.shape[2]) // 2
    return np.pad(prf_cube, [(0, 0), (dy, dy), (dx, dx)], mode="constant")

def _nearest_id_map(r_lamD, theta_deg, radii_lamD, azimuths_deg):
    """Map each pixel to its nearest-neighbour PRF slice index."""
    n_th   = len(azimuths_deg)
    step   = 360 / n_th
    k_th   = (theta_deg // step).astype(int) % n_th
    k_r    = np.digitize(r_lamD, radii_lamD).clip(1, len(radii_lamD) - 1) - 1
    return np.where(k_r == 0, 0, 1 + (k_r - 1) * n_th + k_th)

def _bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg):
    """
    Return index quadruple (k00,k10,k01,k11) and weight quadruple
    (w00,w10,w01,w11) for bilinear interpolation.
    """
    n_th   = len(azimuths_deg)
    step   = 360 / n_th

    th_lo  = (theta_deg // step).astype(int) % n_th
    beta   = (theta_deg - th_lo * step) / step
    th_hi  = (th_lo + 1) % n_th

    k_r_hi = np.digitize(r_lamD, radii_lamD).clip(1, len(radii_lamD) - 1)
    k_r_lo = k_r_hi - 1
    dr     = radii_lamD[k_r_hi] - radii_lamD[k_r_lo]
    alpha  = np.divide(r_lamD - radii_lamD[k_r_lo], dr,
                       out=np.zeros_like(r_lamD), where=dr != 0)

    def _idx(r_i, th_i):
        return np.where(r_i == 0, 0, 1 + (r_i - 1) * n_th + th_i)

    k00, k10 = _idx(k_r_lo, th_lo), _idx(k_r_hi, th_lo)
    k01, k11 = _idx(k_r_lo, th_hi), _idx(k_r_hi, th_hi)

    w00 = (1 - alpha) * (1 - beta)
    w10 = alpha       * (1 - beta)
    w01 = (1 - alpha) * beta
    w11 = alpha       * beta

    return (k00, k10, k01, k11), (w00, w10, w01, w11)

def build_sampling_grid(fine_sampling, coarse_sampling,
                        iwa, owa, sampling_theta,
                        resolution_elem, res_mas):
    """
        Construct a polar sampling grid on which PSFs/PRFs will be evaluated.

        Parameters
        ----------
        fine_sampling : float
            Radial step (in λ/D) inside the IWA.
        coarse_sampling : float
            Radial step (λ/D) between IWA and OWA.
        iwa, owa : float
            Inner and outer working angles, in λ/D.
        sampling_theta : float
            Azimuthal step in degrees.
        resolution_elem : float
            Radial step (λ/D) beyond the OWA (typically ≈ 1 λ/D).
        res_mas : float
            Scale factor that converts λ/D to milliarcseconds.

        Returns
        -------
        radii_lamD : ndarray
            Radial node positions in λ/D.
        radii_mas : ndarray
            Same radii converted to milliarcseconds.
        azimuths_deg : astropy.units.Quantity
            Azimuthal node positions, 0–360 deg.
    """

    radii_lamD = _build_radial_grid(iwa, owa,
                                    fine_sampling,
                                    coarse_sampling,
                                    resolution_elem)
    radii_mas  = radii_lamD * res_mas
    azimuths_deg = _build_azimuth_grid(sampling_theta)
    
    return radii_lamD, radii_mas, azimuths_deg

def make_prf_cube(radii_lamD, azimuths_deg, c_lam,
                  n_lambda=1, bandwidth_frac=0.0,
                  prf_width_px=64, det_shape=(51, 51),
                  prop_tag="roman_preflight",
                  prop_opts=None, source_sed=None):
    """
    Build a PRF cube (cropped to `det_shape`) averaged over a band.

    All wavelengths are in microns (µm).

    Parameters
    ----------
    radii_lamD, azimuths_deg, c_lam
        Polar grid and central wavelength as before.
    n_lambda, bandwidth_frac
        Define the λ grid.  If n_lambda==1 or bandwidth_frac==0 → monochromatic.
    prf_width_px
        Power-of-two gridsize for PROPER (default 64).
    det_shape
        (ny, nx) size of detector-frame PRF kernel (default 51×51).
    prop_tag / prop_opts
        Passed straight to PROPER.
    source_sed
        *Optional* 1-D iterable of length `n_lambda` that gives the
        **spectral weight** at each λ sample.  If `None` → flat spectrum.

    Returns
    -------
    prf_cube_det : ndarray, shape (N_prf, ny, nx)
        Broadband (or mono) PRF cube cropped to `det_shape`.
    """
    # wvl_grid and weights 
    lam_grid, lam_wts = _lam_grid_and_weights(c_lam,
                                              n_lambda,
                                              bandwidth_frac,
                                              source_sed)

    # polar grid list
    azimuths_deg = u.Quantity(azimuths_deg, u.deg)
    rr2d, tt2d   = np.meshgrid(radii_lamD, azimuths_deg, indexing="ij")
    mask         = ~((rr2d == 0) & (tt2d != 0 * u.deg))
    rr, tt       = rr2d.ravel()[mask], tt2d.ravel()[mask]
    n_prf        = rr.size

    prop_opts = {} if prop_opts is None else prop_opts.copy()
    prf_cube = np.empty((n_prf, prf_width_px, prf_width_px), np.float32)

    # progress bar
    def _bar(i):
        bar = "#" * int(40 * i / n_prf)
        sys.stdout.write(f"\r[{bar:40s}] {i:>4}/{n_prf:<4}")
        sys.stdout.flush()

    t0 = time.time(); _bar(0)
    for idx, (r_ld, th_deg) in enumerate(zip(rr, tt), 1):
        rad = th_deg.to_value(u.rad)
        run_opts = {**prop_opts,
                    "source_x_offset": r_ld * np.cos(rad),
                    "source_y_offset": r_ld * np.sin(rad)}

        wfs, _ = proper.prop_run_multi(prop_tag, lam_grid,
                                       prf_width_px, QUIET=True,
                                       PASSVALUE=run_opts)
        prf_cube[idx-1] = np.tensordot(lam_wts, np.abs(wfs)**2, axes=(0, 0))
        _bar(idx)

    sys.stdout.write("\n")
    print("PRF generation done in %.1f min" % ((time.time()-t0)/60.0))

    # centre-crop to detector frame
    ny_det, nx_det = det_shape
    dy = (prf_width_px - ny_det) // 2
    dx = (prf_width_px - nx_det) // 2

    return prf_cube[:, dy:dy+ny_det, dx:dx+nx_det]

def convolve_with_prfs(obj, prfs_array, radii_lamD, azimuths_deg,
                       pix_scale_mas, mas_per_lamD, weighted=False):
    """
    Field-dependent shift–add convolution.

    Parameters
    ----------
    scene           : 2-D image to blur (ny,nx)
    prfs_array      : PRF cube     (N_prf, ny_prf, nx_prf) **cropped** to detector size
    radii_lamD      : 1-D array of λ/D radial nodes (must include 0)
    azimuths_deg    : 1-D astropy Quantity (deg) azimuth nodes (0 ≤ θ < 360, constant step)
    pix_scale_mas   : detector pixel scale in mas / pix  (scalar)
    mas_per_lamD    : conversion factor: 1 λ/D → mas     (scalar)
    weighted        : if True use bilinear (4-PRF) weighting, else nearest neighbour

    Returns
    -------
    detector        : blurred image, same shape as `scene`
    """

    r_lamD, theta_deg = _pixel_polar(obj.shape,
                                     pix_scale_mas,
                                     mas_per_lamD)
    prf_pad = _pad_prf_cube(prfs_array, obj.shape)

    if not weighted:
        # nearest-neighbour convolution
        prf_ids = _nearest_id_map(r_lamD, theta_deg,
                                  radii_lamD, azimuths_deg)
        conv = np.zeros_like(obj, float)
        for k in np.unique(prf_ids):
            mask = prf_ids == k
            if mask.any():
                conv += fftconvolve(obj * mask, prf_pad[k], mode="same")
        return conv

    # bilinear-weighted convolution
    idx_maps, w_maps = _bilinear_indices_weights(r_lamD, theta_deg,
                                                 radii_lamD, azimuths_deg)
    k00,k10,k01,k11 = idx_maps
    w00,w10,w01,w11 = w_maps

    conv = np.zeros_like(obj, float)
    for k in np.unique(np.stack(idx_maps)):
        weight = (np.where(k00==k, w00, 0.0) +
                  np.where(k10==k, w10, 0.0) +
                  np.where(k01==k, w01, 0.0) +
                  np.where(k11==k, w11, 0.0))
        if weight.any():
            conv += fftconvolve(obj * weight, prf_pad[k], mode="same")
    return conv