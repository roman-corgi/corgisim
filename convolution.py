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

def convolve_2d_scene(iwa, owa, sampling_theta, resolution_elem,
                      n_lam, c_lam, bandwidth, options, prf_width,
                      image_file, sampling_plot=False):
    
    def generating_prfs():
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
        sampling1 = 0.1  #Fine sampling interval for the innermost region
        sampling2 = 0.2  #Coarser sampling interval for the intermediate region
        sampling3 = resolution_elem  #Sampling interval for the outer region
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

        #Generating PRFs
        minlam = c_lam * (1 - bandwidth / 2)
        maxlam = c_lam * (1 + bandwidth / 2)
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
                    break #Skip first set of PSFs if radial offset is 0 at the start

        #Saves all PRFs to a single FITS file
        hdu = fits.PrimaryHDU(psfs_array)
        hdul = fits.HDUList([hdu])
        hdul.writeto("prfs.fits", overwrite=True)

        #Optional plotting embedded function for sampling PRFs
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
            imshow2(prf, prf, 'HLC PSF: Band 1', 'HLC PSF: Band 1',
                    lognorm1=True, lognorm2=True,
                    pxscl1=prf_pixelscale_m.to(u.mm / u.pix), pxscl2=resolution_elem, patches2=patches)

        if sampling_plot:
            sampling_plots_prfs()

        return psfs_array

    #Generates the PRFs
    prfs_array = generating_prfs()
    npsf = prfs_array.shape[1]

    #Convert units
    mas_per_lamD = (c_lam / bandwidth * u.radian).to(u.mas)
    iwa_mas = iwa * mas_per_lamD
    owa_mas = owa * mas_per_lamD
    psf_pixelscale_mas = resolution_elem * mas_per_lamD / u.pix

    #Loads the input image
    input_image = fits.getdata(image_file)
    px = input_image.shape[0] // 2 - npsf // 2
    py = px + npsf

    #Extract the region to convolve
    disk = input_image[px:py, px:py]

    #Performs the convolution
    disk_sim = np.zeros_like(disk)
    for prf in prfs_array: #1D to 2D 
        disk_sim += np.reshape(np.convolve(disk.flatten(), prf.flatten(), mode='same'), (npsf, npsf))

    #Saves convolved image
    hdu = fits.PrimaryHDU(disk_sim)
    hdul = fits.HDUList([hdu])
    output_file = f'test_convolved.fits' #Need to rename when finalized
    hdul.writeto(output_file, overwrite=True)

    #For plotting the result
    xpix = (np.arange(-npsf // 2, npsf // 2) * psf_pixelscale_mas.value) / 1000
    ypix = (np.arange(-npsf // 2, npsf // 2) * psf_pixelscale_mas.value) / 1000

    fig, ax = plt.subplots()
    im = ax.imshow(disk_sim, cmap='magma', extent=[np.min(xpix), np.max(xpix), np.max(ypix), np.min(ypix)])
    ax.invert_yaxis()
    circ1 = Circle((0, 0), iwa_mas.value / 1000, color='r', fill=False)
    circ2 = Circle((0, 0), owa_mas.value / 1000, color='r', fill=False)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.set_ylabel('arcsec')
    ax.set_xlabel('arcsec')
    fig.colorbar(im, orientation='vertical')
    plt.show()