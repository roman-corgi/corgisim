

class Scene():
    ''' 
    A class that defines the an astrophysical scene
    - Information about the host star (brightness, spectral type, etc.)
    - A list of point sources (brightness, location, spectra?, etc.)
    - A 2D background scene that will be convolved with the off-axis PSFs
        - Format needs to be determined. Likely a fits hdu with specific header keywords. 
        - Input with North-Up East-Left orientation.

    Arguments: 
        host_star_properties: A dictionary that contains information about the host star, such as: 
            brightness, spectral type, ra?, dec?, etc. 
        point_sources_info: A list of dictionaries that contain information about the point sources, such as:
            brightness, location, spectra?, etc.
        background_scene: An astropy HDU that contains a background scene as data and a header full of relevant information, such as: 
            pixel_scale, etc. 
    '''
    def __init__(self, host_star_properties=None, point_source_info=None, twoD_scene_hdu=None):
        self.host_star_dict = host_star_properties
        self.point_source_list = point_source_info
        self.twoD_scene = twoD_scene_hdu



class SimulatedScene(): 
    '''
    A class that defines a simulated scene. 

    Arguments: 
        input_scene: A corgisim.scene.Scene object that contains the information scene to be simulated.
    '''
    def __init__(self, input_scene):

        self.input_scene = input_scene

        #The following three attributes will hold astropy HDUs with the simulated images.
        self.host_star_image = None
        self.point_source_image = None
        self.twoD_image = None

        #This will be basically the sum of the above three images at the right location on the detector
        self.total_image = None



def combine_simulated_scenes_list(scene_list):
    '''
    Function that takes a list of Simulated_Scene objects and combines them into a single Simulated_Scene object. 

    Arguments: 
        scene_list: A list of corgisim.scene.Simulated_Scene objects.


    Returns:
        corgisim.scene.Simulated_Scene: A scene object with the host_star_image, point_source_image, and background_scene attributes
                                        populated with the sum of the input scenes.
    '''
