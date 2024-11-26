from corgisim import scene
from corgisim import instrument
from astropy.io import fits


### Set up a scene. 
host_star_properties = {'v_mag': 1, 'spectral_type': 'G2V', 'ra': 0, 'dec': 0}
point_source_info = [{'v_mag': 1, 'location': [0, 0]}, {'v_mag': 1, 'location': [0, 0]}]
background_scene_hdu = fits.open('disk_image.fits')

sim_scene = scene.Scene(host_star_properties, point_source_info, background_scene_hdu)

### Define the state of the instrument 

#Set up the configuration of the instrument and the detector
proper_keywords = {"filter":"Band 1"}
emccd_keywords = {"read_noise": 3}

# Set up the instrument
inst = instrument.CGI(proper_keywords, emccd_keywords)

### Simulate the scene. 

# Populate scene.host_star_image with an astropy HDU that contains a noiseless on-axis PSF
scene_with_psf = inst.get_psf(sim_scene, on_the_fly=True, oversample = 1, return_oversample = False)

# Populate scene.point_source_image with an astropy HDU that contains the point sources
scene_with_point_sources = inst.inject_point_sources(scene_with_psf, on_the_fly=True, oversample = 1, return_oversample = False)'

# Populate scene.background_scene with an astropy HDU that contains the simulated scene and associated metadata in the header
scene_with_2D = inst.simulate_2D_scene(scene_with_point_sources, on_the_fly=True, oversample = 1, return_oversample = False)


### Place the scene on the detector and simulate the readout.
total_scene = inst.place_scene_on_detector(scene_with_2D)


### TBD: How to simulate an observation sequence - likely a series of functions in observations.py

# Simulate the readout
final_image = inst.generate_detector_image(total_scene)


