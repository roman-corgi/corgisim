from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os
import corgisim
from corgisim import outputs

def test_L1_product_fits_format():
    """Test the headers of saved L1 product FITS file
    """
    #print('testrun')
    #### simulate using corgisim
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1b'
    cor_type = 'hlc_band1'

    mag_companion = [25,25]
    ###the position of companions in unit of mas
    ####550nm/2.3m = 29.4 mas
    ###we used sep = 3 lambda/D here 
    dx= [3*49.3,-3*49.3]
    dy= [3*49.3,-3*49.3]
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]},
                         {'Vmag': mag_companion[1], 'magtype': 'vegamag','position_x':dx[1] , 'position_y':dy[1]}]


    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    proper_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True)
    sim_scene = optics.get_psf(base_scene)

    sim_scene = optics.inject_point_sources(base_scene,sim_scene)

    gain =1000
    emccd_keywords ={'em_gain':gain}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene, exptime,full_frame=True,loc_x=300, loc_y=300)
    
    ### save the L1 product fits file to test/testdata folder
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
    outputs.save_hdu_to_fits(sim_scene.image_on_detector,outdir=outdir, write_as_L1=True)
    
    ### read the L1 product fits file
    prihdr = sim_scene.image_on_detector[0].header
    exthdr = sim_scene.image_on_detector[1].header
    time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"CGI_{prihdr['VISITID']}_{time_in_name}_L1_.fits"


    f = os.path.join( outdir , filename)
 
    with fits.open(f) as hdul:
        data = hdul[1].data
        prihr = hdul[0].header
        exthr = hdul[1].header
        
        # Check that the dtype is exactly uint16
    assert data.dtype == np.uint16, f"Expected np.uint16, but got {data.dtype}"
    assert exthr['BITPIX'] == 16, f"Expected BITPIX=16, but got {exthr['BITPIX']}"
    assert data.shape[0] == 1200, f"Expected data shape[0]=2200, but got {data.shape[0]}"
    assert data.shape[1] == 2200, f"Expected data shape[1]=1200, but got {data.shape[1]}"  

    ### delete file after testing
    print('Deleted the FITS file after testing')
    os.remove(f)
    


if __name__ == '__main__':
    #run_sim()
    test_L1_product_fits_format()
