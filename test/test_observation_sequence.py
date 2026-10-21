from corgisim import scene, instrument, inputs, observation
from corgisim.scene import SimulatedImage
from astropy.io import fits
import proper
import roman_preflight_proper
import pytest
import numpy as np
import os

def test_generate_observation_sequence():

    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    gain =1000
    emccd_keywords ={'em_gain':gain, }

    base_scene = scene.Scene(host_star_properties)
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    detector = instrument.CorgiDetector( emccd_keywords, photon_counting=True)
    
    exp_time = 20
    n_frames = 1
    
    # Test a single frame
    with pytest.warns(UserWarning): 
        simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == n_frames
    assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

    # Test a single full frame 
    simulatedImage_list_fullframe = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames,full_frame=True, loc_x=300, loc_y=300)

    assert isinstance(simulatedImage_list_fullframe, list)
    assert len(simulatedImage_list_fullframe) == n_frames
    assert isinstance(simulatedImage_list_fullframe[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector, fits.hdu.hdulist.HDUList)

    assert len(simulatedImage_list_fullframe[n_frames-1].image_on_detector) == 2 # Primary and Image HDU
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].data, np.ndarray)

    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EXPTIME'] == exp_time
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EMGAIN_C'] == gain

    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['PHTCNT'] == detector.photon_counting
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['OPGAIN'] == gain
    assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['FRAMET'] == exp_time

    # Test satellite spots generation 
    satellite_spots_frame_time = 10
    satellite_spots_number_of_frames = 2
    satellite_spots_gain = 500
    detector_satspots = instrument.CorgiDetector( {'em_gain':satellite_spots_gain})

    satspot_keywords = {'num_pairs':2, 'sep_lamD': 7, 'angle_deg': [0,90], 'contrast': 1e-5}

    simulatedImage_list_fullframe = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames,
                                                                              satspots_are_present = True, satspots_detector = detector_satspots, satspots_exptime = satellite_spots_frame_time, satspots_number_of_frames = satellite_spots_number_of_frames, satspot_keywords = satspot_keywords,
                                                                              full_frame= True,loc_x=300, loc_y=300 )


    assert isinstance(simulatedImage_list_fullframe, list)
    assert len(simulatedImage_list_fullframe) == n_frames + 3*satellite_spots_number_of_frames
    assert isinstance(simulatedImage_list_fullframe[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector, fits.hdu.hdulist.HDUList)

    assert len(simulatedImage_list_fullframe[n_frames-1].image_on_detector) == 2 # Primary and Image HDU
    assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].data, np.ndarray)
    
    for i in range(0,3*satellite_spots_number_of_frames):
        prihdr = simulatedImage_list_fullframe[i].image_on_detector[0]
        exthdr = simulatedImage_list_fullframe[i].image_on_detector[1]

        assert exthdr.header['EMGAIN_C'] == satellite_spots_gain
        assert exthdr.header['EXPTIME'] == satellite_spots_frame_time
        assert exthdr.header['SATSPOTS'] == 1
        assert exthdr.header['ISPC'] == False

        assert prihdr.header['PHTCNT'] == False
        assert prihdr.header['OPGAIN'] == satellite_spots_gain
        assert prihdr.header['FRAMET'] == satellite_spots_frame_time

    assert simulatedImage_list_fullframe[-1].image_on_detector[1].header['EXPTIME'] == exp_time
    assert simulatedImage_list_fullframe[-1].image_on_detector[1].header['EMGAIN_C'] == gain
    assert simulatedImage_list_fullframe[-1].image_on_detector[1].header['SATSPOTS'] == 0

    assert simulatedImage_list_fullframe[-1].image_on_detector[0].header['PHTCNT'] == detector.photon_counting
    assert simulatedImage_list_fullframe[-1].image_on_detector[0].header['OPGAIN'] == gain
    assert simulatedImage_list_fullframe[-1].image_on_detector[0].header['FRAMET'] == exp_time

    
    # Test several frames
    wavelength = 0.575e-6 # meter, assuming band 1
    lam_D = np.degrees(wavelength/2.3)*3600*1000 # in mas
    fov_shape = image_star.shape
    xcen = (fov_shape[1]-1)/2
    ycen = (fov_shape[0]-1)/2
    pix_scale = 0.0218*1000 #mas/pix
    guess_x = xcen+shift[0]/pix_scale
    guess_y = ycen+shift[1]/pix_scale
    n_frames = 5
    simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames,
                                                                              satspots_are_present = True, satspots_detector = detector_satspots, satspots_exptime = satellite_spots_frame_time, satspots_number_of_frames = satellite_spots_number_of_frames, satspot_keywords = satspot_keywords,
                                                                              full_frame= False )
    # Test correctness of satellite spots

    # Positive    
    peak_positive = []
    for i in range(satellite_spots_number_of_frames, 2*satellite_spots_number_of_frames ):
        satellite_spot_image_positive = simulatedImage_list[i].image_on_detector.data
        fitresult=[]
        for angle in satspot_keywords['angle_deg']: # note: this line does not work if angle_deg is not list
            sep = satspot_keywords['sep_lamD']*lam_D # mas

            guess_x1 = xcen+sep*np.cos(np.radians(angle))/pix_scale
            guess_x2 = xcen-sep*np.cos(np.radians(angle))/pix_scale

            guess_y1 = ycen+sep*np.sin(np.radians(angle))/pix_scale
            guess_y2 = ycen-sep*np.sin(np.radians(angle))/pix_scale

            for guess_x,guess_y in zip([guess_x1,guess_x2],[guess_y1,guess_y2]):
                peak,fwhm,x,y=gaussfit2d(image_star_with_spots,guess_x,guess_y)
                fitresult.append([peak,fwhm,x,y])
        peak_satspots_ave = np.mean(np.array(fitresult)[:,0])
        peak_positive.append(peak_satspots_ave)
    assert np.allclose(peak_positive, peak_positive[0], rtol=0.1)        
    #Negative

    # for i in range(2*satellite_spots_number_of_frames,3*satellite_spots_number_of_frames ):
    #     satellite_spot_image_negative = simulatedImage_list[i].image_on_detector.data

    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == n_frames
    assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

def test_generate_observation_scenario_from_cpgs():
    script_dir = os.getcwd()

    #Test with target and reference
    filepath = 'test/test_data/cpgs_ref_pol_satspots.xml'
    abs_path =  os.path.join(script_dir, filepath)
   
    scene_target, scene_reference, optics, detector_target, detector_reference, visit_list, satellite_dict_target, satellite_dict_reference = inputs.load_cpgs_data(abs_path, output_dim=51, fast_gain_mode = True, gain_CIC_Q=0.0)
    len_list = 0 
    for visit in visit_list:
        len_list += visit['number_of_frames']
        if visit['isReference']:
            len_list+=(satellite_dict_reference['satellite_spots_number_of_frames']*3)
        else:
            len_list+=(satellite_dict_target['satellite_spots_number_of_frames']*3)
            
    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path, output_dim=51, fast_gain_mode = True, gain_CIC_Q=0.0)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == len_list
    assert isinstance(simulatedImage_list[0], SimulatedImage)

    #Test with an off-axis light source
    mag_companion = [10] 
    dx= [3*49.3]
    dy= [3*49.3]
    point_source_info = [{'Vmag': mag_companion[0], 'magtype': 'vegamag','position_x':dx[0] , 'position_y':dy[0]}]
    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path, point_source_info=point_source_info, output_dim=51, fast_gain_mode = True, gain_CIC_Q=0.0)

    for simulatedImage in simulatedImage_list:
        #Check that the target has a point source and the target doesn't  
        if simulatedImage.input_scene.ref_flag :
            assert '_point_source_Vmag' not in simulatedImage.input_scene.__dict__
            assert simulatedImage.point_source_image == None  
        else:
            assert simulatedImage.input_scene._point_source_Vmag == mag_companion
            assert simulatedImage.input_scene._point_source_magtype == ['vegamag']
            assert simulatedImage.input_scene.point_source_dra == dx
            assert simulatedImage.input_scene.point_source_ddec == dy
            assert isinstance(simulatedImage.point_source_image, fits.hdu.image.PrimaryHDU)  


    #Test with only target
    filepath = 'test/test_data/cpgs_without_reference.xml'
    abs_path =  os.path.join(script_dir, filepath)
    scene_target, optics, detector_target, visit_list, satellite_dict_target = inputs.load_cpgs_data(abs_path, output_dim=51, fast_gain_mode = True, gain_CIC_Q=0.0)
    len_list = 0 
    for visit in visit_list:
        len_list += visit['number_of_frames']
        len_list+=(satellite_dict_target['satellite_spots_number_of_frames']*3)

    simulatedImage_list = observation.generate_observation_scenario_from_cpgs(abs_path, output_dim=51, fast_gain_mode = True, gain_CIC_Q=0.0)
    assert isinstance(simulatedImage_list, list)
    assert len(simulatedImage_list) == len_list
    assert isinstance(simulatedImage_list[0], SimulatedImage)




if __name__ == '__main__':
    test_generate_observation_sequence()
    test_generate_observation_scenario_from_cpgs()