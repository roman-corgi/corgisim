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

proper.prop_use_fftw(DISABLE=False)


def generating_prfs(fine_sampling, coarse_sampling, iwa, owa,
                    sampling_theta, resolution_elem, n_lam,
                    c_lam, bandwidth, options, prf_width, 
                    sampling_plot=False, save_path="prfs.fits"):
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
    sampling_plot = boolean to control plotting of sampling grid
    save_path = path to save the FITS file containing all PRFs

    '''
    #Create the sampling grid the PSFs will be made on
    sampling1 = fine_sampling  #sampling interval for the innermost region
    sampling2 = coarse_sampling  #sampling interval for the intermediate region
    sampling3 = resolution_elem  #sampling interval for the outer region
    offsets1 = np.arange(0, iwa + 1, sampling1)  #Region from center to about inner w.a.
    offsets2 = np.arange(iwa + 1, owa, sampling2)  #Region from about inner w.a. to outer w.a.
    offsets3 = np.arange(owa, 15 + sampling3, sampling3)  #Region from the outer w.a. to beyond

    r_offsets = np.hstack([offsets1, offsets2, offsets3])  #Combined array of all radial offsets
    nr = len(r_offsets)  #Total number of radial offsets

    thetas = np.arange(0, 360, sampling_theta) * u.deg  #Array of angular offsets from 0 to 360 degrees w/ specified interval
    nth = len(thetas)  #Total number of angular offsets

    #Total number of PRFs required for the grid
    #Calculated based on the number of radial and angular offsets
    prfs_required = (nr - 1) * nth + 1
    display(prfs_required)

    #Plots the field angles for grid
    theta_offsets = []
    for r in r_offsets[1:]:
        theta_offsets.append(thetas.to(u.radian).value)
    theta_offsets = np.array(theta_offsets)
    theta_offsets.shape

    #Generates the PRFs
    minlam = c_lam * (1 - bandwidth / 2)
    maxlam = c_lam * (1 + bandwidth / 2)
    lam_array = np.linspace(minlam, maxlam, n_lam)
    lam_array = np.array([c_lam])


    psfs_array = np.zeros(shape=((len(r_offsets)-1)*len(thetas) + 1, prf_width, prf_width)) 

    count = 0
    start = time.time()
    for i, r in enumerate(r_offsets):
        for j, th in enumerate(thetas):
            if count >= psfs_array.shape[0]:
                break
            
            xoff = r * np.cos(th)
            yoff = r * np.sin(th)
            options.update({'source_x_offset': xoff.value, 'source_y_offset': yoff.value})

            (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, prf_width, QUIET=True, PASSVALUE=options)

            prfs = np.abs(wfs) ** 2
            prf = np.sum(prfs, axis=0) / n_lam

            psfs_array[count] = prf
            count += 1

            if r < r_offsets[1]:
                break  #skips first set of PSFs if radial offset is 0 at the start

    #Saves all PRFs to a single FITS file
    hdu = fits.PrimaryHDU(psfs_array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(save_path, overwrite=True)

    #Optional to embed function to plot sampling PRFs
    def sampling_plots_prfs():
        fig = plt.figure(dpi=125, figsize=(4, 4))

        ax1 = plt.subplot(111, projection='polar')
        ax1.plot(theta_offsets, r_offsets[1:], '.')
        ax1.set_yticklabels([])
        ax1.set_rticks([iwa, owa, max(r_offsets)])  #Less radial ticks
        ax1.set_rlabel_position(55)  #Move radial labels away from plotted line
        ax1.set_thetagrids(thetas[::2].value)
        ax1.grid(axis='x', visible=True, color='black', linewidth=1)
        ax1.grid(axis='y', color='black', linewidth=1)
        ax1.set_title('Distribution of PRFs', va='bottom')
        ax1.set_axisbelow(False)

        #Plots 2 band images
        (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, prf_width, QUIET=False, PASSVALUE=options)
        prf_pixelscale_m = pxscls_m[0] * u.m / u.pix

        patches = [Circle((0, 0), iwa, color='c', fill=False), Circle((0, 0), owa, color='c', fill=False)]
        imshow2(prf, prf, ' ', ' ',
                lognorm1=True, lognorm2=True,
                pxscl1=prf_pixelscale_m.to(u.mm / u.pix), pxscl2=resolution_elem, patches2=patches)

    if sampling_plot:
        sampling_plots_prfs()


def convolve_2d_scene(prf_file, scene_info):
    mas_per_lamD = (c_lam / bandwidth * u.radian).to(u.mas)
    iwa_mas = iwa * mas_per_lamD
    owa_mas = owa * mas_per_lamD
    psf_pixelscale_mas = resolution_elem * mas_per_lamD / u.pix

    #Loads the PRFs from the FITS file
    prfs_array = fits.getdata(prf_file)
    npsf = prfs_array.shape[1]

    #Loads the input image
    input_image = fits.getdata(scene_info)
    
    #coordinates of the center of the input image calculated here
    cx, cy = input_image.shape[1] // 2, input_image.shape[0] // 2
    #[1] gives the width of the image
    #[0] gives the height

    #Extracts the region to convolve, centered around the middle of the input image
    disk_sim = np.zeros_like(input_image)
    #is an array of zeros with the same shape as input_image
    #later used to store sum of convolved images
    
    #Convolves each PRF with the image and sum the results
    for prf in prfs_array:
        '''
        Loops thru each prf in the prf array
        -> prf_padded: Ensures that the PSF 
        is centered and aligned properly with the input 
        image during convolution.
        -> np.pad is used to add padding to prf so that
        it becomes the same size as the input image.
        '''
        prf_padded = np.pad(prf, [(input_image.shape[0] // 2 - prf.shape[0] // 2,
                                  input_image.shape[0] // 2 - prf.shape[0] // 2),
                                 (input_image.shape[1] // 2 - prf.shape[1] // 2,
                                  input_image.shape[1] // 2 - prf.shape[1] // 2)],
                           mode='constant', constant_values=0)
        disk_sim += convolve2d(input_image, prf_padded, mode='same')
        '''
        convolve2d function performs element-wise multiplication in the 
        frequency domain, effectively blurring the input image by the PSF. 
        The result is then transformed back to the spatial domain.
        -> mode = same: output size will be the same as the input image size.

        The code accumulates the results of convolving the input image with
        each PSF, providing a composite image that represents the combined
        effect of all PSFs.
        '''

    #Saves convolved image to fits
    hdu = fits.PrimaryHDU(disk_sim)
    hdul = fits.HDUList([hdu])
    output_file = 'convolved.fits'
    hdul.writeto(output_file, overwrite=True)

    #Zooms in region based on iwa and owa for better plotting demonstration
    zoom_radius_mas = owa_mas.value * 1.5  #Extend beyond OWA to include some margin
    zoom_radius_pixels = int(zoom_radius_mas / psf_pixelscale_mas.value)

    #Crops the image for zoomed-in view
    zoomed_disk_sim = disk_sim[cy-zoom_radius_pixels:cy+zoom_radius_pixels, cx-zoom_radius_pixels:cx+zoom_radius_pixels]

    #Plots the zoomed-in result
    zoomed_xpix = (np.arange(-zoom_radius_pixels, zoom_radius_pixels) * psf_pixelscale_mas.value) / 1000
    zoomed_ypix = (np.arange(-zoom_radius_pixels, zoom_radius_pixels) * psf_pixelscale_mas.value) / 1000

    fig, ax = plt.subplots()
    im = ax.imshow(zoomed_disk_sim, cmap='magma', extent=[np.min(zoomed_xpix), np.max(zoomed_xpix), 
                                                          np.max(zoomed_ypix), np.min(zoomed_ypix)])
    ax.invert_yaxis()
    circ1 = Circle((0, 0), iwa_mas.value / 1000, color='r', fill=False)
    circ2 = Circle((0, 0), owa_mas.value / 1000, color='r', fill=False)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.set_ylabel('arcsec')
    ax.set_xlabel('arcsec')
    fig.colorbar(im, orientation='vertical')
    plt.show()