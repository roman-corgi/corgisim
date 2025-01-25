import proper
import numpy as np

class Scene():
    ''' 
    A class that defines the an astrophysical scene
    - Information about the host star (brightness, spectral type, etc.)
    - A list of point sources (brightness, location, spectra?, etc.)
    - A 2D background scene that will be convolved with the off-axis PSFs
        - Format needs to be determined. Likely a fits hdu with specific header keywords. 
        - Input with North-Up East-Left orientation.

    Arguments: 
        host_star_properties (dict): A dictionary that contains information about the host star, such as: 
            brightness, spectral type, ra?, dec?, etc. 
        point_sources_info (list): A list of dictionaries that contain information about the point sources, such as:
            brightness, location, spectra?, etc.
        background_scene (HDUList): An astropy HDU that contains a background scene as data and a header full of relevant information, such as: 
            pixel_scale, etc. 
    '''
    def __init__(self, host_star_properties=None, point_source_info=None, twoD_scene_hdu=None):
        self._host_star_Dist = host_star_properties['Dist']  ## host star distance (pc)
        self._host_star_Teff = host_star_properties['Teff']  ## host star effective temperature (K)
        self._host_star_Rs = host_star_properties['Rs']  ## host star radius (solar radii)
        self._point_source_list = point_source_info
        self._twoD_scene = twoD_scene_hdu

            
    @property
    def host_star_Dist(self):
        """
        A Method returns host star distance in pc.
        Returns:
            Distance (float): in parsecs (pc).
        """
        return self._host_star_Dist

                
    @host_star_Dist.setter
    def host_star_Dist(self, value):
        """
        Setter method to validate and set the host star distance.
        Args:
            value (float): The host star distance in parsecs (pc). Must be greater than 0.
        Raises:
            ValueError: If `value` is not greater than 0.
        """
        if value <= 0:
            raise ValueError("Distance must be larger than 0!")
        self._host_star_Dist = float(value)
        
    @property
    def host_star_Teff(self):
        """
        A Method returns host star temperature in K.
        Returns:
            Temperature (float): in K.
        """
        return self._host_star_Teff

                
    @host_star_Teff.setter
    def host_star_Teff(self, value):
        """
        Setter method to validate and set the host star temperature.
        Args:
            value (float): The host star temperature in K. Must be greater than 3000K and less than 12000K.
        Raises:
            ValueError: If `value` is less than 3000K or greater than 12000K.
        """
        if (value < 3000) or (value > 12000):
            raise ValueError("Host star temperature is out of range (3000-12000K)!")
        self._host_star_Teff = float(value)

    @property
    def host_star_Rs(self):
        """
        A Method returns host star radius in solar Radii.
        Returns:
            Rs (float): in solar Radii.
        """
        return self._host_star_Rs

                
    @host_star_Rs.setter
    def host_star_Rs(self, value):
        """
        Setter method to validate and set the host star radius.
        Args:
            value (float): The host star radius in solar radii. Must be greater than 0.1 solar radii  and less than 10 solar radii.
        Raises:
            ValueError: If `value` is less than 0.1 solar radii or greater than 10 solar radii.
        """
        if (value < 0.1) or (value > 10):
            raise ValueError("Host star radius is out of range (0.1-10 solar radii)!")
        self._host_star_Rs = float(value)

    


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
    pass


