

def Scene():
    '''
    
    A class that defines the an astrophysical scene
    - Information about the host star (brightness, spectral type, etc.)
    - A list of point sources (brightness, location, spectra?, etc.)
    - A 2D background scene that will be convolved with the off-axis PSFs
        - This could cotain a

    Arguments: 
        host_star_properties: A dictionary that contains information about the host star, such as: 
            brightness, spectral type, ra?, dec?, etc. 
        point_sources_info: A list of dictionaries that contain information about the point sources, such as:
            brightness, location, spectra?, etc.
        background_scene: An astropy HDU that contains a background scene as data and a header full of relevant information, such as: 
            pixel_scale, etc. 
    '''
    def __init__(self, host_star_properties, point_source_info, background_scene_hdu):
        self.host_star_dict = host_star_properties
        self.point_source_list = point_source_info
        self.background_scene = background_scene_hdu