import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from IPython.display import display


def generating_prfs(iwa, owa, sampling_theta, resolution_elem):
    ''' 
    Function to generate PRF including making a sampling grid
    that the PRFs will be generated on

    Argument (to be revised):
    resolution_elem: (resolution element) Conversion factor 
    from radians to milliarcseconds

    '''

    prf_pixelscale_lamD = resolution_elem #Pixel scale in units of lambda/diameter


    #Create the sampling grid the PSFs will be made on
    sampling1 = 0.1 #Fine sampling interval for the innermost region
    sampling2 = 0.2 #Coarser sampling interval for the intermediate region
    sampling3 = prf_pixelscale_lamD #Sampling interval for the outer region
    offsets1 = np.arange(0,iwa+1,sampling1) #Region from center to about inner w.a.
    offsets2 = np.arange(iwa+1,owa,sampling2) #Region from about inner w.a. to outer w.a.
    offsets3 = np.arange(owa,15+sampling3,sampling3) #Region from the outer w.a. to beyond

    r_offsets = np.hstack([offsets1, offsets2, offsets3]) #Combined array of all radial offsets
    nr = len(r_offsets) #Total number of radial offsets

    thetas = np.arange(0,360,sampling_theta) * u.deg #Array of angular offsets from 0 to 360 degrees w/ specified interval
    nth = len(thetas) #Total number of angular offsets

    '''
    Total number of PSFs required for the grid. 
    Calculated based on the number of radial and angular offsets.
    '''
    psfs_required = (nr - 1) * nth + 1
    display(psfs_required)

    # Plotting field angles
    theta_offsets = []
    for r in r_offsets[1:]:
        theta_offsets.append(thetas.to(u.radian).value)
    theta_offsets = np.array(theta_offsets)
    theta_offsets.shape

    fig = plt.figure(dpi=125, figsize=(4,4))

    ax1 = plt.subplot(111, projection='polar')
    ax1.plot(theta_offsets, r_offsets[1:], '.', )
    ax1.set_yticklabels([])
    ax1.set_rticks([iwa, owa, max(r_offsets)],)  # Less radial ticks
    ax1.set_rlabel_position(55)  # Move radial labels away from plotted line
    ax1.set_thetagrids(thetas[::2].value)
    ax1.grid(axis='x', visible=True, color='black', linewidth=1)
    ax1.grid(axis='y', color='black', linewidth = 1)
    ax1.set_title('Distribution of PRFs', va='bottom')
    ax1.set_axisbelow(False)