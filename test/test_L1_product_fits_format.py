import corgisim
from corgisim import scene, instrument, outputs, inputs, observation
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os, shutil

def test_L1_product_fits_format():
    """Test the headers of saved L1 product FITS file
    """
    #print('testrun')
    #### simulate using corgisim
    #### testing the defalut value pass to header
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
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

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':51,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1, }
                

    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)

    sim_scene = optics.inject_point_sources(base_scene,sim_scene)

    
    emccd_keywords ={}
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
    filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"


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
    assert exthr['FSMX'] == 0.0, f"Expected data FSMX=10, but got {exthr['FSMX']}" 
    assert exthr['FSMY'] == 0.0, f"Expected data FSMY=10, but got {exthr['FSMY']}"
    assert prihr['PSFREF'] == False, f"Expected data PSFREF=False, but got {prihr['PSFREF']}"
    assert prihr['PHTCNT'] == True, f"Expected data PSFREF=True, but got {prihr['PHTCNT']}"
    assert prihr['SATSPOTS'] == 0, f"Expected data PSFREF=True, but got {prihr['PHTCNT']}"
    assert prihr['ROLL'] == 0.0, f"Expected data ROLL=0, but got {prihr['ROLL']}"

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



    ### delete file after testing
    print('Deleted the FITS file after testing headers populated with default values')
    os.remove(f)

    ####################################################################################################
    #### testing the non-defalut(input) value pass to header
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1B'
    cor_type = 'hlc_band1'

    mag_companion = [25,25]
    ###the position of companions in unit of mas
    ####550nm/2.3m = 29.4 mas
    ###we used sep = 3 lambda/D here 
    dx= [3*49.3,-3*49.3]
    dy= [3*49.3,-3*49.3]
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag','ref_flag':True}
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]},
                         {'Vmag': mag_companion[1], 'magtype': 'vegamag','position_x':dx[1] , 'position_y':dy[1]}]


    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties, point_source_info)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':51,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1,
                    'fsm_x_offset_mas':10.0,'fsm_y_offset_mas':20.0 }
                ##pass fsm_x_offset_mas and fsm_y_offset_mas for no zero value as test

    roll_angle=10.0 ##degree
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True, roll_angle=roll_angle)
    sim_scene = optics.get_host_star_psf(base_scene)

    sim_scene = optics.inject_point_sources(base_scene,sim_scene)

    gain =100
    e_per_dn=1.0
    emccd_keywords ={'em_gain':gain,'e_per_dn':e_per_dn}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords, photon_counting = False)
    sim_scene = detector.generate_detector_image(sim_scene, exptime,full_frame=True,loc_x=300, loc_y=300)
    
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
        
        # Check that the dtype is exactly uint16
    assert data.dtype == np.uint16, f"Expected np.uint16, but got {data.dtype}"
    assert exthr['BITPIX'] == 16, f"Expected BITPIX=16, but got {exthr['BITPIX']}"
    assert data.shape[0] == 1200, f"Expected header shape[0]=2200, but got {data.shape[0]}"
    assert data.shape[1] == 2200, f"Expected header shape[1]=1200, but got {data.shape[1]}"
    assert exthr['FSMX'] == 10.0, f"Expected header FSMX=10, but got {exthr['FSMX']}" 
    assert exthr['FSMY'] == 20.0, f"Expected header FSMY=10, but got {exthr['FSMY']}"
    assert prihr['PSFREF'] == True, f"Expected header PSFREF=False, but got {prihr['PSFREF']}"
    assert prihr['PHTCNT'] == False, f"Expected header PSFREF=False, but got {prihr['PHTCNT']}"
    assert prihdr['FRAMET'] == exptime, f"Expected header FRAMET = {exptime}, but got {prihdr['FRAMET']}"
    assert prihr['ROLL'] == 10.0, f"Expected data ROLL=10, but got {prihr['ROLL']}"

    assert exthdr['KGAINPAR'] == e_per_dn, f"Expected data KGAINPAR={e_per_dn}, but got {exthdr['KGAINPAR']}"
    assert exthdr['EMGAIN_C'] == gain, f"Expected data EMGAIN_C={gain}, but got {exthdr['EMGAIN_C']}"
    assert exthdr['EMGAIN_A'] == gain, f"Expected data EMGAIN_A={gain}, but got {exthdr['EMGAIN_A']}"
    assert exthdr['ISPC'] == 0, f"Expected header ISPC=0, but got {exthdr['ISPC']}"

    ### delete file after testing
    print('Deleted the FITS file after testing headers populated with non-dafult values(inputs)')
    os.remove(f)

    ####################################################################################################
    #### testing the headers for pupil images

    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass = '1F'
    cor_type = 'hlc_band1'
    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    #Define the host star properties
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag','ref_flag':True}
    

    #Create a Scene object that holds all this information
    base_scene = scene.Scene(host_star_properties)

    ####setup the wavelength for the simulation, nlam=1 for monochromatic image, nlam>1 for broadband image 
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':51,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':0, 'use_lyot_stop':0,  'use_field_stop':0,\
                    'use_pupil_lens':1}
                ##pass fsm_x_offset_mas and fsm_y_offset_mas for no zero value as test


    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True,oversampling_factor =1)
    sim_scene = optics.get_host_star_psf(base_scene)

    gain =100
    e_per_dn=1.0
    emccd_keywords ={'em_gain':gain,'e_per_dn':e_per_dn}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords, photon_counting = False)
    sim_scene = detector.generate_detector_image(sim_scene, exptime,full_frame=True,loc_x=300, loc_y=300)
    
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
        
        # Check that the dtype is exactly uint16
    assert data.dtype == np.uint16, f"Expected np.uint16, but got {data.dtype}"
    assert exthdr['SPAM_H'] ==  1001.3, f"Expected data SPAM_H=1001.3, but got {exthdr['SPAM_H']}"
    assert exthdr['SPAM_V']== 16627,  f"Expected data SPAM_V = 16627, but got {exthdr['SPAM_V']}"
    assert exthdr['SPAMNAME'] =='OPEN' , f"Expected data SPAMNAME ='OPEN', but got {exthdr['SPAMNAME']}"
    assert exthdr['SPAMSP_H']== 1001.3, f"Expected data SPAMSP_H=1001.3, but got {exthdr['SPAMSP_H']}"
    assert exthdr['SPAMSP_V'] == 16627, f"Expected data SPAMSP_V=16627, but got {exthdr['SPAMSP_V']}"

    assert exthdr['LSAM_H'] ==  20822, f"Expected data LSAM_H=20822, but got {exthdr['LSAM_H']}"
    assert exthdr['LSAM_V']== 17393.9,  f"Expected data LSAM_V = 17393.9, but got {exthdr['LSAM_V']}"
    assert exthdr['LSAMNAME'] =='OPEN' , f"Expected data LSAMNAME ='OPEN', but got {exthdr['LSAMNAME']}"
    assert exthdr['LSAMSP_H']== 20822, f"Expected data LSAMSP_H=20822, but got {exthdr['LSAMSP_H']}"
    assert exthdr['LSAMSP_V'] == 17393.9, f"Expected data LSAMSP_V=17393.9, but got {exthdr['LSAMSP_V']}"

    assert exthdr['CFAM_H'] == 55829.2, f"Expected data CFAM_H=55829.2, but got {exthdr['CFAM_H']}"
    assert exthdr['CFAM_V'] == 10002.7, f"Expected data CFAM_V=10002.7, but got {exthdr['CFAM_V']}"
    assert exthdr['CFAMNAME'] == '1F', f"Expected data CFAMNAME='1F', but got {exthdr['CFAMNAME']}"
    assert exthdr['CFAMSP_H'] == 55829.2, f"Expected data CFAMSP_H=55829.2, but got {exthdr['CFAMSP_H']}"
    assert exthdr['CFAMSP_V'] == 10002.7, f"Expected data CFAMSP_V=10002.7, but got {exthdr['CFAMSP_V']}"

    assert exthdr['DPAM_H'] == 62626.4, f"Expected data DPAM_H=62626.4, but got {exthdr['DPAM_H']}"
    assert exthdr['DPAM_V'] == 21024.3, f"Expected data DPAM_V=21024.3, but got {exthdr['DPAM_V']}"
    assert exthdr['DPAMNAME'] == 'PUPIL', f"Expected data DPAMNAME='PUPIL', but got {exthdr['DPAMNAME']}"
    assert exthdr['DPAMSP_H'] == 62626.4, f"Expected data DPAMSP_H=62626.4, but got {exthdr['DPAMSP_H']}"
    assert exthdr['DPAMSP_V'] == 21024.3, f"Expected data DPAMSP_V=21024.3, but got {exthdr['DPAMSP_V']}"

    assert exthdr['FPAM_H'] ==  3509.4, f"Expected data FPAM_H= 3509.4, but got {exthdr['FPAM_H']}"
    assert exthdr['FPAM_V'] == 32824.7, f"Expected data FPAM_V= 32824.7, but got {exthdr['FPAM_V']}"
    assert exthdr['FPAMNAME'] == 'OPEN_12', f"Expected data FPAMNAME='OPEN_12', but got {exthdr['FPAMNAME']}"
    assert exthdr['FPAMSP_H'] ==  3509.4, f"Expected data FPAMSP_H= 3509.4, but got {exthdr['FPAMSP_H']}"
    assert exthdr['FPAMSP_V'] ==  32824.7, f"Expected data FPAMSP_V= 32824.7, but got {exthdr['FPAMSP_V']}"

    assert exthdr['FSAM_H'] ==  30677.2, f"Expected data FSAM_H=30677.2, but got {exthdr['FSAM_H']}"
    assert exthdr['FSAM_V'] == 2959.5, f"Expected data FSAM_V=2959.5, but got {exthdr['FSAM_V']}"
    assert exthdr['FSAMNAME'] == 'OPEN', f"Expected data FSAMNAME='OPEN', but got {exthdr['FSAMNAME']}"
    assert exthdr['FSAMSP_H'] ==  30677.2, f"Expected data FSAMSP_H=30677.2, but got {exthdr['FSAMSP_H']}"
    assert exthdr['FSAMSP_V'] == 2959.5, f"Expected data FSAMSP_V=2959.5, but got {exthdr['FSAMSP_V']}"

    ### delete file after testing
    print('Deleted the FITS file after testing headers populated with pupil images sumulation')
    os.remove(f)


def test_L1_product_from_CPGS():
    """Test the saving of files from CPGS
    """

    script_dir = os.getcwd()

    filepath = 'test/test_data/cpgs_short_sequence.xml'
    abs_path =  os.path.join(script_dir, filepath)
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata/cpgs')
    
    scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(abs_path)
    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path, full_frame=True, loc_x=300, loc_y=300, save_as_fits=True, output_dir=outdir)

    #Check that there are as many simulated images as files
    assert len(simulatedImage_list) == len([name for name in os.listdir(outdir) if os.path.isfile(outdir+'/'+name)])

    #Check that the names are correct
    #for simulatedImage in simulatedImage_list:
    i = 0
    for visit in visit_list:
        for _ in range(visit['number_of_frames']):

            prihdr = simulatedImage_list[i].image_on_detector[0].header
            exthdr = simulatedImage_list[i].image_on_detector[1].header
            time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
            filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"

            f = os.path.join( outdir , filename)
            assert os.path.isfile(f)
            assert prihdr['ROLL'] == visit["roll_angle"]
            i += 1
    # Delete the files 
    shutil.rmtree(outdir)

    # test at the observation sequence level
    # n_frames and exp_time values are not critical
    n_frames = 100
    exp_time = 30

    simulatedImage_list_sequence = observation.generate_observation_sequence( scene_target, optics, detector_target, exp_time, n_frames, save_as_fits=True, output_dir=outdir, full_frame=True, loc_x=300, loc_y=300)
    
    #Check that there are as many simulated images as files
    assert len(simulatedImage_list_sequence) == len([name for name in os.listdir(outdir) if os.path.isfile(outdir+'/'+name)]) == n_frames

    #Check that the names are correct
    for simulatedImage in simulatedImage_list_sequence:
        prihdr = simulatedImage.image_on_detector[0].header
        exthdr = simulatedImage.image_on_detector[1].header
        time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
        filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"

        f = os.path.join( outdir , filename)
        assert os.path.isfile(f)
     
        with fits.open(f) as hdul:
            data_file = hdul[1].data
            prihr_file = hdul[0].header
            exthr_file = hdul[1].header
    # Delete the files 
    shutil.rmtree(outdir)

if __name__ == '__main__':
    #run_sim()
    test_L1_product_fits_format()
    test_L1_product_from_CPGS()