from corgisim import scene
from corgisim import instrument
import proper
import roman_preflight_proper
import pytest
import shutil
import os

def test_install():
    """
    Test the installation
    """    

    # Copy the prescription files
    path_directory = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.isfile( path_directory + '/roman_preflight.py')):
        prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight.py'
        shutil.copy( prescription_file, path_directory )

    if not (os.path.isfile(path_directory + '/roman_preflight_compact.py')):
        prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight_compact.py'
        shutil.copy( prescription_file, path_directory )

    assert(os.path.isfile(path_directory + '/roman_preflight.py'))
    assert(os.path.isfile(path_directory + '/roman_preflight_compact.py'))

    #Check your installation by creating a scene
    #Define the host star properties
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
    cor_type = 'hlc_band1'
    
    #### simulate using corgisim
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)
    assert isinstance(base_scene, scene.Scene)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                       'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
   
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    assert isinstance(optics, instrument.CorgiOptics)

    sim_scene = optics.get_host_star_psf(base_scene)
    image = sim_scene.host_star_image.data

    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 30
    detector = instrument.CorgiDetector( emccd_keywords)
    assert isinstance(detector, instrument.CorgiDetector)


if __name__ == '__main__':
    test_install()
  