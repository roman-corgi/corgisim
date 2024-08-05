import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from IPython.display import clear_output, display
import time
from pathlib import Path
import copy

import proper
proper.prop_use_fftw(DISABLE=False)

import roman_phasec_proper

import cgi_phasec_poppy as cgi
import cgi_phasec_poppy.imshows as imshows
from cgi_phasec_poppy.imshows import *

from importlib import reload


def generating_prfs(iwa, owa, sampling_theta, resolution_elem, n_lam,
                    c_lam, bandwidth, options, prf_width, sampling_plot=False, save_path="prfs.fits"):
    
    ''' 
    Function to generate PRF including optional argument to
    make a sampling grid that the PRFs will be generated on

    Arguments (to be revised):
    iwa = inner working angle
    owa = outer working angle
    sampling_theta = used for sampling grid
    resolution_elem: (resolution element) Conversion factor 
    from radians to milliarcseconds
    n_lam = number of wavelength samples
    c_lam = central wavelength
    bandwidth = fractional bandwidth
    options = dictionary input for changing coronagraph
    type to the correct band
    prf_width = desired width of prf
    sampling_plot = defaults to not plot, set true to show plotting
    save_path = path to save the FITS file containing all PRFs

    '''
    # Create the sampling grid the PSFs will be made on
    sampling1 = 0.1 # Fine sampling interval for the innermost region
    sampling2 = 0.2 # Coarser sampling interval for the intermediate region
    sampling3 = resolution_elem # Sampling interval for the outer region
    offsets1 = np.arange(0, iwa+1, sampling1) # Region from center to about inner w.a.
    offsets2 = np.arange(iwa+1, owa, sampling2) # Region from about inner w.a. to outer w.a.
    offsets3 = np.arange(owa, 15+sampling3, sampling3) # Region from the outer w.a. to beyond

    r_offsets = np.hstack([offsets1, offsets2, offsets3]) # Combined array of all radial offsets
    nr = len(r_offsets) # Total number of radial offsets

    thetas = np.arange(0, 360, sampling_theta) * u.deg # Array of angular offsets from 0 to 360 degrees w/ specified interval
    nth = len(thetas) # Total number of angular offsets

    # Total number of PRFs required for the grid
    # Calculated based on the number of radial and angular offsets
    prfs_required = (nr - 1) * nth + 1
    display(prfs_required)

    # Plots the field angles for grid
    theta_offsets = []
    for r in r_offsets[1:]:
        theta_offsets.append(thetas.to(u.radian).value)
    theta_offsets = np.array(theta_offsets)
    theta_offsets.shape

    # Generating PRFs
    minlam = c_lam * (1 - bandwidth/2)
    maxlam = c_lam * (1 + bandwidth/2)
    lam_array = np.linspace(minlam, maxlam, n_lam)
    lam_array = np.array([c_lam])
    
    psfs_array = np.zeros(shape=((len(r_offsets)-1)*len(thetas) + 1, prf_width, prf_width))

    count = 0
    start = time.time()
    for i, r in enumerate(r_offsets):
        for j, th in enumerate(thetas):
            xoff = r * np.cos(th)
            yoff = r * np.sin(th)
            options.update({'source_x_offset': xoff.value, 'source_y_offset': yoff.value})
        
            (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, prf_width, QUIET=True, PASSVALUE=options)

            prfs = np.abs(wfs)**2
            prf = np.sum(prfs, axis=0) / n_lam 

            psfs_array[count] = prf
            count += 1

            if r < r_offsets[1]: 
                break # skip first set of PSFs if radial offset is 0 at the start

    # Saves all PRFs to a single FITS file
    hdu = fits.PrimaryHDU(psfs_array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(save_path, overwrite=True)

    # Optional embedded function to plot sampling PRFs
    def sampling_plots_prfs():
        fig = plt.figure(dpi=125, figsize=(4,4))

        ax1 = plt.subplot(111, projection='polar')
        ax1.plot(theta_offsets, r_offsets[1:], '.')
        ax1.set_yticklabels([])
        ax1.set_rticks([iwa, owa, max(r_offsets)])  # Less radial ticks
        ax1.set_rlabel_position(55)  # Move radial labels away from plotted line
        ax1.set_thetagrids(thetas[::2].value)
        ax1.grid(axis='x', visible=True, color='black', linewidth=1)
        ax1.grid(axis='y', color='black', linewidth=1)
        ax1.set_title('Distribution of PRFs', va='bottom')
        ax1.set_axisbelow(False)

        # Plots 2 band images
        (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, prf_width, QUIET=False, PASSVALUE=options)
        prf_pixelscale_m = pxscls_m[0] * u.m / u.pix

        patches = [Circle((0, 0), iwa, color='c', fill=False), Circle((0, 0), owa, color='c', fill=False)]
        imshow2(prf, prf, 'HLC PSF: Band 1', 'HLC PSF: Band 1',
                lognorm1=True, lognorm2=True, 
                pxscl1=prf_pixelscale_m.to(u.mm/u.pix), pxscl2=resolution_elem, patches2=patches)
    
    if sampling_plot:
        sampling_plots_prfs()