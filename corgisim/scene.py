import proper
import numpy as np
import re
from synphot.models import BlackBodyNorm1D, Box1D, ConstFlux1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
from corgisim import pol
import cgisim
import os

class Scene():
    ''' 
    A class that defines the an astrophysical scene
    - Information about the host star (brightness, spectral type, etc.)
    - A list of point sources (brightness, location, spectra?, etc.)
    - A 2D background scene that will be convolved with the off-axis PSFs
        - Format needs to be determined. Likely a fits hdu with specific header keywords. 
        - Input with North-Up East-Left orientation.

    Arguments: 
        host_star_properties (dict): A dictionary that contains information about the host star, including: 
            - "Vmag" (float): The V-band magnitude of the host star.
            - "spectral_type" (str): Spectral type of the host star. Must follow the format: "<Class><Subclass>[<Luminosity Class>]".
                Valid components include:
                - Spectral classes: O, B, A, F, G, K, M, L, T, Y
                - Subclasses: Integer from 0 to 9, optionally with a decimal (e.g., 3.5)
                - Luminosity classes (optional): I, II, III, IV, V, VI, VII
                - Example valid spectral types: "G2V", "M5III", "A0", "T7", "L3.5V", "B2IV"
            - "magtype" (str): the magnitude type: 
                'vegamag' for Vega magnitude system.
                'ABmag' for AB magnitude system
            - "ref_flag" (boolean):optional, whether the input scene is a reference star (True) or a science target (False). Default is fasle
            - "pol_state" (float array): optional, vector of length 4 consisting of the I, Q, U, and V components of the stokes parameter
                describing how the starlight is polarized, default is unpolarized or [1,0,0,0]

        point_sources_info (list): A list of dictionaries, each representing an off-axis point source in the scene. Each dictionary must contain:
            - "Vmag" (float): The apparent V-band magnitude of the source.
            - "magtype" (str): The photometric system used for the magnitude. Must be one of:
                - "vegamag": Vega magnitude system
                - "ABmag": AB magnitude system
            - "position_x" (float): 
                X-coordinate of the source position in mas, relative to the host star.
            - "position_y" (float): 
                Y-coordinate of the source position in mas, relative to the host star.
            - "Custom_Spectrum" (optional): 
                A custom spectrum for the source. If provided, this spectrum will override the default spectrum generated based on Vmag.
            - "pol_state" (float array): optional, vector of length 4 consisting of the I, Q, U and V components of the stokes parameter
                describing how the source light is polarized, default is unpolarized or [1,0,0,0]
            Notes:
                - The coordinates should be provided in the same reference frame and orientation as the background scene (typically North-up, East-left).
                - All magnitudes must be consistent with their respective magnitude type.
                - If no custom spectrum is provided, a default flat spectrum will be generated based on the V-band magnitude.

        background_scene (HDUList): An astropy HDU that contains a background scene as data and a header full of relevant information, such as: 
            pixel_scale, etc. 

    Raises:
        ValueError: If the provided spectral type is invalid.
        ValueError: If the provided stokes vector is not of length 4 or the polarized intensity magnitude exceeds the total intensity magnitude
    '''
    def __init__(self, host_star_properties=None, point_source_info=None, twoD_scene_hdu=None):
        
        self._host_star_Vmag = host_star_properties['Vmag']  ## host star Vband magnitude

        # Validate the magnitude type
        if host_star_properties['magtype'] not in ['vegamag', 'ABmag']:
            raise ValueError("Invalid magnitude type. Valid options are: 'vegamag' or 'ABmag'.")
        # Store the magnitude type
        self._host_star_magtype = host_star_properties['magtype']  # Type of magnitude (Vega or AB)

        ### check if input spectral type is valid
        if is_valid_spectral_type(host_star_properties['spectral_type']):
           self._host_star_sptype = host_star_properties['spectral_type']  

        # Set the reference flag from host_star_properties, defaulting to False if not provided
        self.ref_flag = host_star_properties.get('ref_flag', False)
        
        ### Retrieve the stellar spectrum based on spectral type and V-band magnitude
        ### The self.stellar_spectrum attribute is an instance of the SourceSpectrum class (from synphot), 
        ### used to store and retrieve the wavelength and stellar flux.
        self.stellar_spectrum = self.get_stellar_spectrum( self._host_star_sptype, self._host_star_Vmag, magtype =self._host_star_magtype)

        #Set the polarization state from host_star_properties, default to [1,0,0,0] if none provided
        self.host_star_pol_state = host_star_properties.get('pol_state', np.array([1,0,0,0]))
        
        pol.check_stokes_vector_validity(self.host_star_pol_state)

        #normalizes stokes vector so that I = 1
        self.host_star_pol_state = np.divide(self.host_star_pol_state, self.host_star_pol_state[0])

        #self._point_source_list = point_source_info
        # Extract V-band magnitude and magnitude type from point source info
        if point_source_info is not None:
            n_off_axis_source = len( point_source_info)
            print(f"Adding {n_off_axis_source} off-axis sources")
            # Extract V-band magnitudes from point source info
            self._point_source_Vmag = [source['Vmag'] for source in point_source_info]
            self._point_source_magtype =[source['magtype'] for source in point_source_info]# Type of magnitude ('vegamag' or 'ABmag')
            self.point_source_x = [source['position_x'] for source in point_source_info]
            self.point_source_y = [source['position_y'] for source in point_source_info]
            # Extract optional custom spectrum, if provided
            self.point_source_spectrum = [source.get('Custom_Spectrum', None) for source in point_source_info]  

            # Generate the off-axis source spectrum using provided parameters
            self.off_axis_source_spectrum = self.get_off_axis_source_spectrum(self._point_source_Vmag,
                                                                            spectrum=self.point_source_spectrum,
                                                                            magtype=self._point_source_magtype)
            
            #Set the polarization state of sources, default to [1,0,0,0] if none provided
            self.point_source_pol_state = [source.get('pol_state', np.array([1,0,0,0])) for source in point_source_info]

            #check validity of source stoke vector and normalizes it
            for source in range(n_off_axis_source):
                pol.check_stokes_vector_validity(self.point_source_pol_state[source])
                self.point_source_pol_state[source] = np.divide(self.point_source_pol_state[source], self.point_source_pol_state[source][0])

        
        self._twoD_scene = twoD_scene_hdu

        
    @property
    def host_star_sptype(self):
        """
        A Method returns host star spectral type
        Returns:
            sptype (str): specral type
        """
        return self._host_star_sptype

                
    @host_star_sptype.setter
    def host_star_sptype(self, value):
        """
        Setter method to validate and set the host star spectral type.
        Args:
            value (str): The host star spectral type
        
        """
        if is_valid_spectral_type(value):
            self._host_star_sptype = float(value)
            

    @property
    def host_star_Vmag(self):
        """
        A Method returns host star apparent magnitude
        Returns:
            sptype (float):  host star apparent magnitude
        """
        return self._host_star_Vmag

                
    @host_star_Vmag.setter
    def host_star_Vmag(self, value):
        """
        Setter method to validate and set the host star spectral type.
        Args:
            value (str): The host star apparent magnitude
        
        """
        self._host_star_Vmag = float(value)

    @property
    def host_star_magtype(self):
        """
        A Method returns type of magnitude
        Returns:
            magtype (str):  type of magnitude
        """
        return self._host_star_magtype

                
    @host_star_magtype.setter
    def host_star_Vmag(self, value):
        """
        Setter method to validate and set the type of magnitude.
        Args:
            value (str): The type of magnitude
        """
        self._host_star_magtype = float(value)

    @property
    def point_source_Vmag(self):
         "List of V-band magnitudes for off-axis point sources."
         return self._point_source_Vmag

                
    @point_source_Vmag.setter
    def point_source_Vmag(self, value):
        """
        Set the list of V-band magnitudes for off-axis point sources.
        Args:
            value (list[float]): A list of apparent V-band magnitudes.
        Raises:
            TypeError: If the input is not a list.
        """
        if not isinstance(value, list):
            raise TypeError("point_source_Vmag must be a list of floats")

        self._point_source_Vmag = value
        

    @property
    def point_source_magtype(self):
        "List of magnitude type for off-axis point sources."
        return self._point_source_magtype

                
    @point_source_magtype.setter
    def point_source_magtype(self, value):
        """
        Set the list of V-band magnitudes for off-axis point sources.
        Args:
            value (list[float]): A list of  magnitude type
        Raises:
            TypeError: If the input is not a list.
        """
        if not isinstance(value, list):
            raise TypeError("point_source_Vmag must be a list of floats")

        self._point_source_magtype = value


    def get_stellar_spectrum(self, sptype, magnitude, magtype = 'vegamag', return_teff=False):
        """
        Retrieves or interpolates a stellar spectrum based on spectral type and magnitude.

        This function either fetches the predefined temperature, metallicity, and gravity values 
        for a given spectral type or interpolates these values if the spectral type is not directly 
        available in the lookup table. It then generates a blackbody spectrum and normalizes it 
        to a given magnitude.

        Args:
            sptype (str): The spectral type of the star (e.g., "G2V", "M5III").
            magnitude (float): The magnitude of the star in the specified system.
            magtype (str, optional): The magnitude type ('vegamag' by default).
            return_teff (bool, optional): If True, also return the effective temperature inferred from the spectral type. Default is False. 

        Returns:
            sp_scale (SourceSpectrum): The scaled stellar spectrum.
            v0 (float, optional): Returned only if `return_teff` is True. The effective temperature of the star in Kelvin.


        Raises:
            ValueError: If the spectral type format is invalid.
        """
        # Mapping of spectral types to temperature, metallicity, and surface gravity
        sptype_teff_mapping = {
            # https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
            "O0V": (45000, 0.0, 4.0),  # Bracketing for interpolation
            "O3V": (45000, 0.0, 4.0),
            "O5V": (43000, 0.0, 4.5),
            "O7V": (36500, 0.0, 4.0),
            "O9V": (32500, 0.0, 4.0),
            "B0V": (31500, 0.0, 4.0),
            "B1V": (26000, 0.0, 4.0),
            "B3V": (17000, 0.0, 4.0),
            "B5V": (15700, 0.0, 4.0),
            "B8V": (12500, 0.0, 4.0),
            "A0V": (9700, 0.0, 4.0),
            "A1V": (9200, 0.0, 4.0),
            "A3V": (8550, 0.0, 4.0),
            "A5V": (8080, 0.0, 4.0),
            "F0V": (7220, 0.0, 4.0),
            "F2V": (6810, 0.0, 4.0),
            "F5V": (6510, 0.0, 4.0),
            "F8V": (6170, 0.0, 4.5),
            "G0V": (5920, 0.0, 4.5),
            "G2V": (5770, 0.0, 4.5),
            "G5V": (5660, 0.0, 4.5),
            "G8V": (5490, 0.0, 4.5),
            "K0V": (5280, 0.0, 4.5),
            "K2V": (5040, 0.0, 4.5),
            "K5V": (4410, 0.0, 4.5),
            "K7V": (4070, 0.0, 4.5),
            "M0V": (3870, 0.0, 4.5),
            "M2V": (3550, 0.0, 4.5),
            "M5V": (3030, 0.0, 5.0),
            "M9V": (2400, 0.0, 5.0),   # Bracketing for interpolation
            "O0IV": (45000, 0.0, 3.8),  # Bracketing for interpolation
            "B0IV": (30000, 0.0, 3.8),
            "B8IV": (12000, 0.0, 3.8),
            "A0IV": (9500, 0.0, 3.8),
            "A5IV": (8250, 0.0, 3.8),
            "F0IV": (7250, 0.0, 3.8),
            "F8IV": (6250, 0.0, 4.3),
            "G0IV": (6000, 0.0, 4.3),
            "G8IV": (5500, 0.0, 4.3),
            "K0IV": (5250, 0.0, 4.3),
            "K7IV": (4000, 0.0, 4.3),
            "M0IV": (3750, 0.0, 4.3),
            "M9IV": (3000, 0.0, 4.7),    # Bracketing for interpolation
            "O0III": (45000, 0.0, 3.5),  # Bracketing for interpolation
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "A0III": (9100, 0.0, 3.5),
            "F0III": (7000, 0.0, 3.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "M0III": (3750, 0.0, 1.5),
            "M6III": (3000, 0.0, 1.0),  # Bracketing for interpolation
            "M9III": (2400, 0.0, 1.0),  # Bracketing for interpolation
            "O0I": (45000, 0.0, 5.0),  # Bracketing for interpolation
            "O6I": (39000, 0.0, 4.5),
            "O8I": (34000, 0.0, 4.0),
            "B0I": (26000, 0.0, 3.0),
            "B5I": (14000, 0.0, 3.0),
            "A0I": (9750, 0.0, 2.0),
            "A5I": (8500, 0.0, 2.0),
            "F0I": (7750, 0.0, 2.0),
            "F5I": (7000, 0.0, 1.5),
            "G0I": (5500, 0.0, 1.5),
            "G5I": (4750, 0.0, 1.0),
            "K0I": (4500, 0.0, 1.0),
            "K5I": (3750, 0.0, 0.5),
            "M0I": (3750, 0.0, 0.0),
            "M2I": (3500, 0.0, 0.0),
            "M5I": (3000, 0.0, 0.0), # Bracketing for interpolation 
            "M9I": (2400, 0.0, 0.0), # Bracketing for interpolation 
            }  

        sptype_list = list(sptype_teff_mapping.keys())

        # Attempt to auto-append "V" if no luminosity class is given
        if sptype not in sptype_teff_mapping:
            if len(sptype) == 2:
            #if not any(sptype.endswith(cls) for cls in ['I', 'II', 'III', 'IV', 'V','VI','VII','VIII']):
                sptype += 'V'  # assume main sequence if no class specified
        
        if sptype in sptype_list:
            #print('aaa',sptype)
            v0, v1, v2 = sptype_teff_mapping[sptype]
        else:
             # Interpolate values for undefined sptype
    
            sptype_list.sort(key=sort_sptype)
            rank_list = np.array([sort_sptype(st) for st in sptype_list])
        
            # Find the rank of the input spec type
            rank = sort_sptype(sptype)  
            # Grab values from tuples and interpolate based on rank
            tup_list0 = np.array([sptype_teff_mapping[st][0] for st in sptype_list])
            tup_list1 = np.array([sptype_teff_mapping[st][1] for st in sptype_list])
            tup_list2 = np.array([sptype_teff_mapping[st][2] for st in sptype_list])
            v0 = np.interp(rank, rank_list, tup_list0)
            v1 = np.interp(rank, rank_list, tup_list1)
            v2 = np.interp(rank, rank_list, tup_list2)
            
        # Create a blackbody spectrum using the interpolated or retrieved temperature
        # sp wavelengh unit is the default for synphot: angstrom
        # sp flux unit is the default for synphot: photlam (photons/s/cm^2/anstrom)
        sp = SourceSpectrum(BlackBodyNorm1D, temperature=v0 )
        # Define the V band bandpass
        v_band = SpectralElement.from_filter('johnson_v')
        
        # Scale the spectrum based on magnitude 
        if magtype == 'vegamag':
            # read vega spetrum
            vega_spec = SourceSpectrum.from_vega()
            # sp_scale has the same units as sp
            sp_scale = sp.normalize(renorm_val= magnitude * VEGAMAG, band=v_band,vegaspec = vega_spec )
        if magtype == 'ABmag':
            raise ValueError("AB magnitude system has not been implemented yet. Please use Vega magnitudes instead.")

        if return_teff:
             #### This is for testing the spytpe and teff mapping
            return sp_scale, v0
        else:
            return sp_scale

   

    def get_off_axis_source_spectrum(self, vmag, magtype, spectrum=None):
        """
        Generate a list of off-axis source spectra scaled to given V-band magnitudes.

        Parameters
        ----------
        vmag : list of float
            Desired V-band magnitudes for the off-axis sources.
        magtype : list of str
            Magnitude system used for each source. 
        spectrum : list or None, optional
            Custom input spectra. If None, flat spectra will be used and scaled to match `vmag`.
            Custom spectra are not yet supported and will raise an error if provided.

        Returns
        -------
        list of synphot.SourceSpectrum
            A list of `SourceSpectrum` objects scaled to the desired V magnitudes.
        """
        if not isinstance(vmag, list):
            raise TypeError("vmag must be a list of floats.")
        if not isinstance(magtype, list):
            raise TypeError("magtype must be a list of strings.")
        if len(vmag) != len(magtype):
            raise ValueError("vmag and magtype must have the same length.")

        if spectrum is None:
            spectrum = [None] * len(vmag)
        elif not isinstance(spectrum, list):
            raise TypeError("spectrum must be a list if provided.")
        elif len(spectrum) != len(vmag):
            raise ValueError("spectrum must be the same length as vmag and magtype.")

        # Prepare Vega reference and V-band filter (shared for all)
        vega_spec = SourceSpectrum.from_vega()
        v_band = SpectralElement.from_filter('johnson_v')

        scaled_spectra = []
        for m, spec, mtype in zip(vmag, spectrum, magtype):
            if spec is not None:
                raise NotImplementedError("Custom spectra are not yet supported.")

            if mtype != 'vegamag':
                raise ValueError(f"Unsupported magnitude type '{mtype}'. Only 'vegamag' is supported.")

            # Create and normalize flat spectrum
            flat_spec = SourceSpectrum(ConstFlux1D, amplitude=1 * units.PHOTLAM)
            scaled_spec = flat_spec.normalize(
                renorm_val=m * units.VEGAMAG,
                band=v_band,
                vegaspec=vega_spec
            )
            scaled_spectra.append(scaled_spec)

        return scaled_spectra
    


class SimulatedImage(): 
    '''
    A class that defines a simulated scene. 

    Arguments: 
        input_scene: A corgisim.scene.Scene object that contains the information scene to be simulated.

    '''
    def __init__(self, input_scene):

        self.input_scene = input_scene


        #The following three attributes will hold astropy HDUs with the simulated images.
        self.host_star_image = None
        #self.host_star_image_on_detector = None
        self.point_source_image = None
        self.twoD_image = None


        #This will be basically the sum of the above three images at the right location on the detector
        self.total_image = None
        self.image_on_detector = None


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




def sort_sptype(typestr):
        letter = typestr[0]
        lettervals = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}
        value = lettervals[letter] * 1.0
        value += (int(typestr[1]) * 0.1)
        if "III" in typestr:
            value += 30
        elif 'II' in typestr:
            value += 10  #### class II use teh same teff mapping as class I
        elif 'IV' in typestr:
            value += 40
        elif "V" in typestr:
            value += 50
        elif "I" in typestr:
            value += 10
        
        return value


def is_valid_spectral_type(spectral_type):
    """
    Validate if the input stellar spectral type is in a correct format.

    Args:
        spectral_type (str): The spectral type string (e.g., "G2V").

    Returns:
        bool: True if valid, False otherwise.

    Raises:
        ValueError: If the spectral type is not valid.
    """
    # Define valid spectral classes
    valid_classes = "OBAFGKM"

    # Define valid luminosity classes
    valid_luminosity_classes = "I|II|III|IV|V|VI|VII"

    # Regular expression pattern to match spectral types
    pattern = rf"^([{valid_classes}])([0-9](?:\.\d)?)(?:({valid_luminosity_classes}))?$"

    match = re.match(pattern, spectral_type)

    if not match:
        error_message = (
            f"Invalid spectral type: '{spectral_type}'.\n"
            f"Spectral types must follow the format: '<Class><Subclass>[<Luminosity Class>]'.\n"
            f"- Valid spectral classes: {', '.join(valid_classes)}\n"
            f"- Valid subclasses: Integer from 0 to 9, optionally with a decimal (e.g., 3.5)\n"
            f"- Valid luminosity classes (optional): {valid_luminosity_classes}\n"
            f"\nExample valid spectral types: 'G2V', 'M5III', 'A0', 'T7', 'L3.5V', 'B2IV'."
        )
        raise ValueError(error_message)

    return bool(match)

