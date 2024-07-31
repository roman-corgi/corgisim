def generating_prfs(wavelength, apt_diameter, resolution_elem):
    ''' 
    Function to generate PRF including making a sampling grid
    that the PRFs will be generated on

    Arguments (to be revised):
    wavelength: central wavelength of the light
    apt_diameter: (aperature diameter) diameter of the 
    telescope's primary mirror
    resolution_elem: (resolution element) Conversion factor 
    from radians to milliarcseconds

    '''
    #Need to manipulate when given general data
    mas_per_lamD = (wavelength_c/D*u.radian).to(u.mas) #for resolution_elem variable

    npsf = 0 #Number of pixels across the PSF
    psf_pixelscale = 0 #Physical scale of each pixel
    psf_pixelscale_lamD = 0 #Pixel scale in units of lambda/diameter
    psf_pixelscale_mas = psf_pixelscale_lamD*resolution_elem/u.pix #Pixel scale in milliarcseconds per pixel

    '''
    inner working angle and outer working angle, 
    working angle abbreviated in comments as "w.a."
    '''
    iwa = 3 #smallest angle from optical axis where measurements are taken
    owa = 9 #largest angle from optical axis where measurements are taken

    #Create the sampling grid the PSFs will be made on
    sampling1 = 0 #Fine sampling interval for the innermost region
    sampling2 = 0.2 #Coarser sampling interval for the intermediate region
    sampling3 = psf_pixelscale_lamD #Sampling interval for the outer region
    offsets1 = np.arange(0,iwa+1,sampling1) #Region from center to about inner w.a.
    offsets2 = np.arange(iwa+1,owa,sampling2) #Region from about inner w.a. to outer w.a.
    offsets3 = np.arange(owa,15+sampling3,sampling3) #Region from the outer w.a. to beyond

    r_offsets = np.hstack([offsets1, offsets2, offsets3]) #Combined array of all radial offsets
    nr = len(r_offsets) #Total number of radial offsets
    r_offsets_mas = r_offsets*mas_per_lamD #Convert radial offsets to milliarcseconds

    sampling_theta = 0 #Angular sampling interval in degrees
    thetas = np.arange(0,360,sampling_theta)*u.deg #Array of angular offsets from 0 to 360 degrees w/ specified interval
    nth = len(thetas) #Total number of angular offsets

    '''
    Total number of PSFs required for the grid. 
    Calculated based on the number of radial and angular offsets.
    '''
    psfs_required = (nr-1)*nth + 1
    display(psfs_required)

    # Plotting field angles
    theta_offsets = [] #Array of angular positions for each radial offset
    for r in r_offsets[1:]: #Each radial offset gets all the angular positions
        theta_offsets.append(thetas.to(u.radian).value)
    theta_offsets = np.array(theta_offsets)
    theta_offsets.shape

    #Code to plot sampling grid
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