from corgisim import scene
from corgisim import instrument
from astropy.io import fits

#######################
### Set up a scene. ###
#######################

#Define the host star properties
host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
#Make a list of point sources
point_source_info = [{'v_mag': 1, 'location': [0, 0]}, {'v_mag': 1, 'location': [0, 0]}]
#Read in a 2D scene
twoD_scene_hdu = fits.open('disk_image.fits')

#Create a Scene object that holds all this information
base_scene = scene.Scene(host_star_properties, point_source_info, twoD_scene_hdu)

##########################################
### Define the state of the instrument ###
##########################################

#Set up the configuration of the instrument and the detector
proper_keywords = {"filter":"Band 1","roll_angle": 36.5, "dm1_filename": "dm1.fits", "dm2_filename": "dm2.fits"}
emccd_keywords = {"read_noise": 3}

# Set up the optics
optics = instrument.CorgiOptics(proper_keywords)

# Set up the detector
detector = instrument.CorgiDetector(emccd_keywords)

##########################
### Simulate the scene ###
##########################

# Populate scene.host_star_image with an astropy HDU that contains a noiseless on-axis PSF
simulated_scene_with_psf = optics.get_psf(base_scene, on_the_fly=True, oversample = 1, return_oversample = False)

# Populate scene.point_source_image with an astropy HDU that contains the point sources
scene_with_point_sources = optics.inject_point_sources(base_scene, on_the_fly=True, oversample = 1, return_oversample = False)

# Populate scene.background_scene with an astropy HDU that contains the simulated scene and associated metadata in the header
scene_with_2D = optics.simulate_2D_scene(base_scene, on_the_fly=True, oversample = 1, return_oversample = False)

total_scene = scene.combine_simulated_scence_list([simulated_scene_with_psf, scene_with_point_sources, scene_with_2D])

############################
### Simulate the readout ###
############################
final_image = detector.generate_detector_image(total_scene)

#TO DO: delete this comment
