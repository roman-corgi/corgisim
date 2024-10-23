


class cgi():
    '''
    A class that defines the current configuration of CGI. 

    It should include basically everything about the instrument configuration. 

    Will likely be fairly similar to Jorge's corgisims_core class. 

    '''

    def __init__(self, proper_keywords, emccd_keywords):
        '''
        Initialize the class with the information from the CGI xml file. 

        Initialize the class with two dictionaries: 
        - proper_keywords: A dictionary with the keywords that are used to set up the proper model
        - emccd_keywords: A dictionary with the keywords that are used to set up the emccd model
        '''


    def simulate_2D_scene(data, scene):
        '''
        Function that simulates a 2D scene with the current configuration of CGI. 

        Arguments: 
        data: A dictionary that contains the data to be simulated. 
        scene: A corgisim.scene.Scene object that contains the scene to be simulated.

        Returns: 
        A dictionary that contains the simulated scene. 
        '''

