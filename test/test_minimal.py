import corgisim
import pytest
from corgisim import scene, instrument, observation, outputs, inputs, spec
import matplotlib.pyplot as plt
import numpy as np
import proper
import os
from corgisim.scene import SimulatedImage
import roman_preflight_proper
from astropy.io import fits
import cgisim

def test_excam_mode():
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
    cor_type = 'hlc_band1'

    mag_companion = [25]
    ###the position of companions in unit of mas
    ####550nm/2.3m = 29.4 mas
    ###we used sep = 3 lambda/D here 
    dx= [3*49.3]
    dy= [3*49.3]
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]}]


    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1, }
                

    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, oversampling_factor = 3, if_quiet=True)
    
    efields = optics.get_e_field()

    assert type(efields) == np.ndarray
    assert efields.shape == (7, optics_keywords['output_dim'],  optics_keywords['output_dim'])
     
    sim_scene = optics.get_host_star_psf(base_scene)

    assert(isinstance(sim_scene.host_star_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene.host_star_image.data, np.ndarray)  )
    assert np.any(sim_scene.host_star_image.data > 0)

    #test satellite spots
    polaxis = 10
    output_dim = 51
    contrast = 1e-5

    # calculate offset
    wavelength = 0.575e-6 # meter, assuming band 1
    lam_D = np.degrees(wavelength/2.3)*3600*1000 # in mas
    shift = [0, 6*lam_D] # shift in [x,y]

    satspot_keywords = {'num_pairs':2, 'sep_lamD': 7, 'angle_deg': [0,90], 'contrast': contrast, 'wavelength_m': wavelength}

    ##define the corgi.optics class that hold all information about the instrument paramters                    
    optics_with_spots = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, satspot_keywords=satspot_keywords, oversampling_factor = 3, if_quiet=True)

    sim_scene_with_spots = optics_with_spots.get_host_star_psf(base_scene)
    image_star_with_spots = sim_scene_with_spots.host_star_image.data

    assert(isinstance(sim_scene_with_spots.host_star_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene_with_spots.host_star_image.data, np.ndarray)  )
    assert np.any(sim_scene_with_spots.host_star_image.data > 0)
    
    assert np.any(sim_scene_with_spots.host_star_image.data != sim_scene.host_star_image.data)

    sim_scene = optics.inject_point_sources(base_scene,sim_scene)

    assert(isinstance(sim_scene.point_source_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene.point_source_image.data, np.ndarray)  )
    assert np.any(sim_scene.point_source_image.data > 0)
    assert np.any(sim_scene.host_star_image.data != sim_scene.point_source_image.data)

    emccd_keywords ={}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene, exptime,full_frame=True,loc_x=300, loc_y=300)

    assert(isinstance(sim_scene.point_source_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene.point_source_image.data, np.ndarray)  )
    assert np.any(sim_scene.point_source_image.data > 0)
    assert np.any(sim_scene.host_star_image.data != sim_scene.point_source_image.data)
    ### save the L1 product fits file to test/testdata folder
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
    outputs.save_hdu_to_fits(sim_scene.image_on_detector,outdir=outdir, write_as_L1=True)
    ### read the L1 product fits file
    prihdr = sim_scene.image_on_detector[0].header
    exthdr = sim_scene.image_on_detector[1].header
    time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"


    f = os.path.join( outdir , filename)
 
    with fits.open(f) as hdul:
        data = hdul[1].data
        prihr = hdul[0].header
        exthr = hdul[1].header

    assert data.dtype == np.uint16, f"Expected np.uint16, but got {data.dtype}"
    assert exthr['BITPIX'] == 16, f"Expected BITPIX=16, but got {exthr['BITPIX']}"
    assert data.shape[0] == 1200, f"Expected data shape[0]=2200, but got {data.shape[0]}"
    assert data.shape[1] == 2200, f"Expected data shape[1]=1200, but got {data.shape[1]}"
    assert exthr['FSMX'] == 0.0, f"Expected data FSMX=10, but got {exthr['FSMX']}" 
    assert exthr['FSMY'] == 0.0, f"Expected data FSMY=10, but got {exthr['FSMY']}"
    assert prihr['PSFREF'] == False, f"Expected data PSFREF=False, but got {prihr['PSFREF']}"
    assert prihr['PHTCNT'] == True, f"Expected data PSFREF=True, but got {prihr['PHTCNT']}"

    assert exthdr['KGAINPAR'] == 8.7, f"Expected data KGAINPAR=8.7, but got {exthdr['KGAINPAR']}"
    assert exthdr['EMGAIN_C'] == 1000, f"Expected data EMGAIN_C=1000, but got {exthdr['EMGAIN_C']}"
    assert exthdr['EMGAIN_A'] == 1000, f"Expected data EMGAIN_A=1000, but got {exthdr['EMGAIN_A']}"
    assert exthdr['ISPC'] == 1, f"Expected header ISPC=1, but got {exthdr['ISPC']}"

    assert exthdr['SPAM_H'] ==  1001.3, f"Expected data SPAM_H=1001.3, but got {exthdr['SPAM_H']}"
    assert exthdr['SPAM_V']== 16627,  f"Expected data SPAM_V = 16627, but got {exthdr['SPAM_V']}"
    assert exthdr['SPAMNAME'] =='OPEN' , f"Expected data SPAMNAME ='OPEN', but got {exthdr['SPAMNAME']}"
    assert exthdr['SPAMSP_H']== 1001.3, f"Expected data SPAMSP_H=1001.3, but got {exthdr['SPAMSP_H']}"
    assert exthdr['SPAMSP_V'] == 16627, f"Expected data SPAMSP_V=16627, but got {exthdr['SPAMSP_V']}"

    assert exthdr['LSAM_H'] ==  36898.7, f"Expected data LSAM_H=36898.7, but got {exthdr['LSAM_H']}"
    assert exthdr['LSAM_V']== 4636.2,  f"Expected data LSAM_V = 4636.2, but got {exthdr['LSAM_V']}"
    assert exthdr['LSAMNAME'] =='NFOV' , f"Expected data LSAMNAME ='NFOV', but got {exthdr['LSAMNAME']}"
    assert exthdr['LSAMSP_H']== 36898.7, f"Expected data LSAMSP_H=36898.7, but got {exthdr['LSAMSP_H']}"
    assert exthdr['LSAMSP_V'] == 4636.2, f"Expected data LSAMSP_V=4636.2, but got {exthdr['LSAMSP_V']}"

    assert exthdr['CFAM_H'] == 55829.2, f"Expected data CFAM_H=55829.2, but got {exthdr['CFAM_H']}"
    assert exthdr['CFAM_V'] == 10002.7, f"Expected data CFAM_V=10002.7, but got {exthdr['CFAM_V']}"
    assert exthdr['CFAMNAME'] == '1F', f"Expected data CFAMNAME='1F', but got {exthdr['CFAMNAME']}"
    assert exthdr['CFAMSP_H'] == 55829.2, f"Expected data CFAMSP_H=55829.2, but got {exthdr['CFAMSP_H']}"
    assert exthdr['CFAMSP_V'] == 10002.7, f"Expected data CFAMSP_V=10002.7, but got {exthdr['CFAMSP_V']}"

    assert exthdr['DPAM_H'] == 38917.1, f"Expected data DPAM_H=38917.1, but got {exthdr['DPAM_H']}"
    assert exthdr['DPAM_V'] == 26016.9, f"Expected data DPAM_V=26016.9, but got {exthdr['DPAM_V']}"
    assert exthdr['DPAMNAME'] == 'IMAGING', f"Expected data DPAMNAME='IMAGING', but got {exthdr['DPAMNAME']}"
    assert exthdr['DPAMSP_H'] == 38917.1, f"Expected data DPAMSP_H=38917.1, but got {exthdr['DPAMSP_H']}"
    assert exthdr['DPAMSP_V'] == 26016.9, f"Expected data DPAMSP_V=26016.9, but got {exthdr['DPAMSP_V']}"

    assert exthdr['FPAM_H'] ==  6757.2, f"Expected data FPAM_H= 6757.2, but got {exthdr['FPAM_H']}"
    assert exthdr['FPAM_V'] == 22424, f"Expected data FPAM_V=22424, but got {exthdr['FPAM_V']}"
    assert exthdr['FPAMNAME'] == 'HLC12_C2R1', f"Expected data FPAMNAME='HLC12_C2R1', but got {exthdr['FPAMNAME']}"
    assert exthdr['FPAMSP_H'] ==  6757.2, f"Expected data FPAMSP_H= 6757.2, but got {exthdr['FPAMSP_H']}"
    assert exthdr['FPAMSP_V'] == 22424, f"Expected data FPAMSP_V=22424, but got {exthdr['FPAMSP_V']}"

    assert exthdr['FSAM_H'] ==  29387, f"Expected data FSAM_H=29387, but got {exthdr['FSAM_H']}"
    assert exthdr['FSAM_V'] == 12238, f"Expected data FSAM_V=12238, but got {exthdr['FSAM_V']}"
    assert exthdr['FSAMNAME'] == 'R1C1', f"Expected data FSAMNAME='R1C1', but got {exthdr['FSAMNAME']}"
    assert exthdr['FSAMSP_H'] ==  29387, f"Expected data FSAMSP_H=29387, but got {exthdr['FSAMSP_H']}"
    assert exthdr['FSAMSP_V'] == 12238, f"Expected data FSAMSP_V=12238, but got {exthdr['FSAMSP_V']}"
    assert exthdr['FSMPRFL'] == 'NFOV' f"Expected data FSMPRFL=NFOV, but got {exthdr['FSMPRFL']}"

    os.remove(f)

def test_cpgs_obs():

    script_dir = os.getcwd()

    #Test with target and reference
    filepath = 'test/test_data/cpgs_mock.xml'
    abs_path =  os.path.join(script_dir, filepath)

    scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(abs_path)

    assert detector_target.photon_counting == True
    
    len_list = 0 
    for visit in visit_list:
        len_list += visit['number_of_frames']

    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == len_list
    assert isinstance(simulatedImage_list[0], SimulatedImage)

    #Test with an off-axis light source
    mag_companion = [10] 
    dx= [3*49.3]
    dy= [3*49.3]
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]}]
    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path, point_source_info=point_source_info)

    i=0
    for visit in visit_list:
        for _ in range(visit['number_of_frames']):        
        #Check that the target has a point source and the target doesn't  
            if simulatedImage_list[i].input_scene.ref_flag :
                assert '_point_source_Vmag' not in simulatedImage_list[i].input_scene.__dict__
                assert simulatedImage_list[i].point_source_image == None

            else:
                assert simulatedImage_list[i].input_scene._point_source_Vmag == mag_companion
                assert simulatedImage_list[i].input_scene._point_source_magtype == ['vegamag']
                assert simulatedImage_list[i].input_scene.point_source_dra == dx
                assert simulatedImage_list[i].input_scene.point_source_ddec == dy
                assert isinstance(simulatedImage_list[i].point_source_image, fits.hdu.image.PrimaryHDU)

            line_roll = [line for line in simulatedImage_list[i].image_on_detector.header['COMMENT'] if 'roll_angle' in line]
            _, roll = line_roll[0].split(':', 1)
            assert float(roll) == visit["roll_angle"]
                
            i += 1      

def test_spec_mode():
    input1 = inputs.Input()

    base_scene = scene.Scene(input1.host_star_properties)
    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'
    bandpass = '3F'
    cases = ['2e-8']      
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    polaxis = 10
    # output_dim define the size of the output image
    output_dim = 121
    overfac = 3
    optics_keywords_slit_prism ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim, 
                  'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,
                  'slit':'R1C2', 'prism':'PRISM3', 'wav_step_um':2E-3}

    optics_slit_prism = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords_slit_prism, if_quiet=True,
                                small_spc_grid = 1, oversampling_factor = overfac, return_oversample = False)

    sim_scene_slit_prism = optics_slit_prism.get_host_star_psf(base_scene)

    assert(isinstance(sim_scene_slit_prism.host_star_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene_slit_prism.host_star_image.data, np.ndarray)  )
    assert np.any(sim_scene_slit_prism.host_star_image.data > 0)

def test_spc_mode():
    input1 = inputs.Input()
    base_scene = scene.Scene(input1.host_star_properties)
    sptype = input1.host_star_properties['spectral_type']
    cgi_mode = 'excam'
    bandpass_corgisim = '4F'
    bandpass_cgisim = '4'
    cor_type = 'spc-wide'
    cases = ['2e-8']       
    rootname = 'spc-wide_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)
    image_star_corgi = sim_scene.host_star_image.data

    assert(isinstance(sim_scene.host_star_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene.host_star_image.data, np.ndarray)  )
    assert np.any(sim_scene.host_star_image.data > 0)

def test_pol_mode():
    Vmag = 8
    sptype = 'G0V'
    mag_companion = 25

    #hlc
    companion_x_pos = 148
    companion_y_pos = 148

    # Companion Stokes parameters I Q U V
    companion_pol = np.array([1, 0.3, 0.1, 0])

    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':'vegamag'}
    point_source_info = [{'Vmag': mag_companion, 'magtype': 'vegamag','position_x':companion_x_pos , 'position_y':companion_y_pos, 'pol_state': companion_pol}]
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ##define instrument parameters
    cgi_mode = 'excam'

    bandpass_corgisim = '1F'
    cor_type = 'hlc'
    output_dim = 51  
    rootname = 'hlc_ni_3e-8'

    #define which wollaston prism to use
    prism = 'POL0' 

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    optics_keywords = {'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':output_dim, 'prism':prism,\
                        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_keywords_0_90 = {'cor_type':cor_type, 'use_errors':1, 'polaxis':-10, 'output_dim':output_dim, 'prism':prism,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_0_90 = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords_0_90, if_quiet=True)

    sim_scene_0_90 = optics_0_90.get_host_star_psf(base_scene)

    image_star_corgi_x = sim_scene_0_90.host_star_image.data[0]
    image_star_corgi_y = sim_scene_0_90.host_star_image.data[1]
    assert(isinstance(sim_scene_0_90.host_star_image, fits.hdu.image.PrimaryHDU)  )
    assert(isinstance(sim_scene_0_90.host_star_image.data[0], np.ndarray)  )
    assert(isinstance(sim_scene_0_90.host_star_image.data[1], np.ndarray)  )

    assert np.any(sim_scene_0_90.host_star_image.data[0] > 0)
    assert np.any(sim_scene_0_90.host_star_image.data[1] > 0)

    sim_scene_0_90 = optics_0_90.inject_point_sources(base_scene, sim_scene_0_90)
    image_comp_corgi_x = sim_scene_0_90.point_source_image.data[0]
    image_comp_corgi_y = sim_scene_0_90.point_source_image.data[1]

    assert(isinstance(sim_scene_0_90.point_source_image, fits.hdu.image.PrimaryHDU)  )

    assert np.any(image_comp_corgi_x != image_star_corgi_x)
    assert np.any(image_comp_corgi_y != image_star_corgi_y)
    
if __name__ == '__main__':
    test_excam_mode()
    test_cpgs_obs()
    test_spc_mode()
    test_spec_mode()
    test_pol_mode()