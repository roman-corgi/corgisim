## import packages
from corgisim import scene
from corgisim import instrument,observation
import matplotlib.pyplot as plt
import numpy as np
import proper
from corgisim import outputs
import roman_preflight_proper
roman_preflight_proper.copy_here()


if __name__ == '__main__':

    ################### Define the Target and Reference Star Properties ######################
    # --- Create the Astrophysical Scene for Target ---
    # VISTYPE = CGIVST_CAL_TGTREF_PHOT
    target_star_name = 'ups And'                 # Name of the host star
    target_Vmag = 4.1                           # V-band magnitude of the host star
    target_sptype = 'F9'                       # Spectral type of the host star
    target_ref_flag = False                      # if the target is a reference star or not, default is False
    target_host_star_properties = {'Vmag': target_Vmag,
                                    'spectral_type': target_sptype,
                                    'magtype': 'vegamag',
                                    'ref_flag': target_ref_flag,
                                    'target_name': target_star_name} 
    # --- Define the companion properties ---
    dx = 161.2    # X positions relative to host star in sky coordinates (dRA) in mas for each companion
    dy = 158.6    # Y positions relative to host star in sky coordinates (dDec) in mas for each companion
    #24.52
    companion_info = [{'Vmag': 24.5,
                       'magtype': 'vegamag',
                       'position_x':dx,
                       'position_y': dy}]
    target_scene = scene.Scene(target_host_star_properties, companion_info)

    # --- Create the Astrophysical Scene for Reference Star ---
    ref_star_name = 'eta UMa'
    ref_Vmag = 1.86
    ref_sptype = 'B3'
    ref_ref_flag = True
    ref_host_star_properties = {'Vmag': ref_Vmag,
                                 'spectral_type': ref_sptype,
                                 'magtype': 'vegamag',
                                 'ref_flag': ref_ref_flag,
                                 'target_name': ref_star_name}
    ref_scene = scene.Scene(ref_host_star_properties)

    ################### Define the Instrument Parameters ######################
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1F'

    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    polaxis = 10 #no polarization
    output_dim = 101
    roll_angle = 0
    nd_filter = 0 # no ND filter
    visit_type = 'CGIVST_TDD_OBS'

    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, "nd":nd_filter}    
                  
    optics_target = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                    roll_angle=roll_angle, visit_type=visit_type)

    optics_ref = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                    roll_angle=roll_angle, visit_type=visit_type)


    ################### Define Detector Parameters ###################

    target_emccd_keywords ={'em_gain':8000}
    detector_target = instrument.CorgiDetector(target_emccd_keywords, photon_counting=True)

    target_sat_emccd_keywords ={'em_gain':1679.848}
    detector_target_sat = instrument.CorgiDetector(target_sat_emccd_keywords, photon_counting=False)

    ref_emccd_keywords ={'em_gain':1123.938}
    detector_ref = instrument.CorgiDetector(ref_emccd_keywords, photon_counting=False)

    ref_sat_emccd_keywords ={'em_gain':391.228}
    detector_ref_sat = instrument.CorgiDetector(ref_sat_emccd_keywords, photon_counting=False)
    
    ################### Define some common parameters for the observation sequence ###################
    # --- Define the output directory for the simulated images ---
    outdir ='/Users/jingwenzhang/Desktop/sim_script/upsAnd_sims_pc_full1'
    # --- Define the location of star on detector ---
    # here we assume the star is centered on the detector, so the location is (512, 512) for a 1024x1024 detector
    loc_x = 512
    loc_y = 512
    # --- define roll angle A and B  ---
    roll_angle_A = -15
    roll_angle_B = 15
    
    # --- define exposure time and frame number for each visit  ---
    t_frame_target =  3.46 # seconds, Exposure Time per Frame for target
    N_frames_target = 2415 # Number of frames for target #153

    t_frame_target_sat =  22.678 # seconds, Exposure Time per Frame for target
    N_frames_target_sat = 33 # Number of frames for target #153

    t_frame_ref = 22.74 # seconds, Exposure Time per Frame for refence star
    N_frames_ref = 97 #42 # Number of frames for reference star

    t_frame_ref_sat = 21.45 # seconds, Exposure Time per Frame for refence star
    N_frames_ref_sat = 21 # Number of frames for reference star

    ################### Visit 1: reference star roll A
    optics_ref.roll_angle = roll_angle_A
    visit_id_1 = '0020001001001901001'
    optics_ref.visit_id = visit_id_1
    # satllite spot frames
    optics_ref.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit1 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref_sat, 
                                                       t_frame_ref_sat, N_frames_ref_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=N_frames_ref_sat)


    # non-satelliet spot frames
    optics_ref.satspot_keywords = None
    visit1 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref, 
                                                       t_frame_ref,N_frames_ref, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)
    ################### Visit 2: target star roll A
    optics_target.roll_angle = roll_angle_A
    visit_id_2 = '0020001001001901002'
    optics_target.visit_id = visit_id_2

    # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit2 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit2 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)
    
    ################### Visit 3: target star roll B
    optics_target.roll_angle = roll_angle_B
    visit_id_3 = '0020001001001901003'
    optics_target.visit_id = visit_id_3

   # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit3 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit3 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)

    ################### Visit 4: target star roll A
    optics_target.roll_angle = roll_angle_A
    visit_id_4 = '0020001001001901004'
    optics_target.visit_id = visit_id_4

    # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit4 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit4 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)
    
    ################### Visit 5: target star roll B
    optics_target.roll_angle = roll_angle_B
    visit_id_5 = '0020001001001901005'
    optics_target.visit_id = visit_id_5

   # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit5 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit5 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)



    ################### Visit 6: reference star roll B
    optics_ref.roll_angle = roll_angle_B
    visit_id_6 = '0020001001001901006'
    optics_ref.visit_id = visit_id_6

    # satllite spot frames
    optics_ref.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit6 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref_sat, 
                                                       t_frame_ref_sat, N_frames_ref_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=N_frames_ref_sat)

    # non-satelliet spot frames
    optics_ref.satspot_keywords = None
    visit6 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref, 
                                                       t_frame_ref,N_frames_ref, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)     

    ################### Visit 7: target star roll B
    optics_target.roll_angle = roll_angle_B
    visit_id_7 = '0020001001001901007'
    optics_target.visit_id = visit_id_7

   # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit7 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit7 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)

    ################### Visit 8: target star roll A
    optics_target.roll_angle = roll_angle_A
    visit_id_8 = '0020001001001901008'
    optics_target.visit_id = visit_id_8

    # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit8 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit8 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)

    ################### Visit 9: target star roll B
    optics_target.roll_angle = roll_angle_B
    visit_id_9 = '0020001001001901009'
    optics_target.visit_id = visit_id_9

   # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit9 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit9 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)

    ################### Visit 10: target star roll A
    optics_target.roll_angle = roll_angle_A
    visit_id_10 = '0020001001001901010'
    optics_target.visit_id = visit_id_10

    # satllite spot frames
    optics_target.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit10 = observation.generate_observation_sequence(target_scene, optics_target, detector_target_sat, 
                                                       t_frame_target_sat, N_frames_target_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames= N_frames_target_sat)

    # non-satelliet spot frames
    optics_target.satspot_keywords = None
    visit10 = observation.generate_observation_sequence(target_scene, optics_target, detector_target, 
                                                       t_frame_target, N_frames_target, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)    
    ################### Visit 11: reference star roll A
    optics_ref.roll_angle = roll_angle_A
    visit_id_11 = '0020001001001901011'
    optics_ref.visit_id = visit_id_11
    # satllite spot frames
    optics_ref.satspot_keywords = {'num_pairs':2, 'sep_lamD': 6.25, 'angle_deg': [0,90], 'contrast': 1e-7, 'wavelength_m': 0.575e-6}
    visit11 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref_sat, 
                                                       t_frame_ref_sat, N_frames_ref_sat, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=N_frames_ref_sat)


    # non-satelliet spot frames
    optics_ref.satspot_keywords = None
    visit11 = observation.generate_observation_sequence(ref_scene, optics_ref, detector_ref, 
                                                       t_frame_ref,N_frames_ref, save_as_fits=True, 
                                                       output_dir=outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,n_satspot_frames=None)                                             


                                                                               
