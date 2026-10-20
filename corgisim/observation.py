### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 
import corgisim
import os
from corgisim import scene, instrument, inputs, observation, outputs
from corgidrp import mocks
from datetime import datetime
import copy 
import warnings
def generate_observation_sequence ( scene, optics, detector, exp_time, n_frames, 
                                    satspots_are_present = False, satspots_detector = None, satspots_exptime = None, satspots_number_of_frames = None, satspot_keywords = None,
                                    vistype = 'CGIVST_TDD_OBS', visit_id= '0200001001001001001', save_as_fits= False, output_dir=None, full_frame= False, loc_x=512, loc_y=512, write_as_L1=True ):
    optics.visit_id = visit_id
    optics.visit_type = vistype
    if save_as_fits == False: 
        warnings.warn(
                    "save_as_fits is set to False, so files are not automatically written to the disk. "
                    "If you want to re-use these files later, save them using outputs.save_hdu_to_fits ",
                    UserWarning,
                )   
    optics.visit_id
    if satspots_are_present:
        optics.SATSPOTS = 1
    sim_scene = optics.get_host_star_psf(scene)
    point_source_present = False

    if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
        point_source_present = True
        sim_scene = optics.inject_point_sources(scene,sim_scene)
    
    simulatedImage_list = []
    if satspots_are_present and (satspots_detector is None or satspots_exptime is None or satspots_number_of_frames is None): 
            raise ValueError('Satellite spots are presents but arguments are missing. Please specify satspot keywords, detector, exposure time and number of frames' )


    #else:
    if save_as_fits:
        # Save the images as fits in output_dir if specified, in corgisim/test/testdata if not
        # Simulation needs to be full frame to be written as L1
        if output_dir == None:
            local_path = corgisim.lib_dir
            outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
            print("No output directory specified. FITS files saved in ", outdir)
        else:
            outdir = output_dir

    if satspots_are_present:
        # For each frame, we take background (no satellite spots), positive and negative
        # Background 
        for i in range(0, satspots_number_of_frames):
            sim_image = satspots_detector.generate_detector_image(sim_scene,satspots_exptime,full_frame=full_frame,loc_x=loc_x, loc_y=loc_y)
            simulatedImage_list.append(copy.deepcopy(sim_image))
            if save_as_fits:
                if write_as_L1: 
                    outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=True)
                else: 
                    dt = datetime.today()
                    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
                    time_in_name = outputs.isotime_to_yyyymmddThhmmsss(dt_str)
                    filename = f"cgi_{visit_id}_{time_in_name}_not_l1_.fits"
                    outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=False, filename=filename)

        for sign in ["positive", "negative"]:
            satspot_keywords["sign"] = sign
            optics.add_satspot(satspot_keywords=satspot_keywords)
            #Get the new PSF with point source if present
            sim_scene = optics.get_host_star_psf(scene)
            if point_source_present:
                sim_scene = optics.inject_point_sources(scene,sim_scene)

            for i in range(0, satspots_number_of_frames):
                sim_image = satspots_detector.generate_detector_image(sim_scene,satspots_exptime, full_frame=full_frame,loc_x=loc_x, loc_y=loc_y)
                simulatedImage_list.append(copy.deepcopy(sim_image))
                optics.remove_satspot(satspot_keywords=satspot_keywords)
                if save_as_fits:
                    if write_as_L1: 
                        outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=True)
                    else: 
                        dt = datetime.today()
                        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
                        time_in_name = outputs.isotime_to_yyyymmddThhmmsss(dt_str)
                        filename = f"cgi_{visit_id}_{time_in_name}_not_l1_.fits"
                        outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=False, filename=filename)                        
        sim_scene = optics.get_host_star_psf(scene)  
        if point_source_present:  
            sim_scene = optics.inject_point_sources(scene, sim_scene) 
            
    for i in range(0, n_frames):
        sim_image = detector.generate_detector_image(sim_scene,exp_time,full_frame=full_frame,loc_x=loc_x, loc_y=loc_y)
        simulatedImage_list.append(copy.deepcopy(sim_image))

        if save_as_fits:
            if write_as_L1: 
                outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=True)
            else: 
                dt = datetime.today()
                dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
                time_in_name = outputs.isotime_to_yyyymmddThhmmsss(dt_str)
                filename = f"cgi_{visit_id}_{time_in_name}_not_l1_.fits"
                outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir ,write_as_L1=False, filename=filename)

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath, save_as_fits= False, save_as_list=False, output_dir=None, full_frame=False, loc_x=512, loc_y=512, point_source_info=None, output_dim=201, polaxis=0, fast_gain_mode='auto', gain_CIC_Q='roman',write_as_L1=True ):
    """Generates an observation scenario by loading instrument, scene, and visit
    information from a CPGS file.

    This function attempts to load both target and reference star information.
    If only target information is available, it proceeds with that.

    Args:
        - filepath (str): The path to the CPGS XML file.
        - save_as_fits (bool): wether or not to save the generated images as fits file
        - save_as_list (bool): wether or not to save the generated images as list of file. In order to save memory, default behaviour is not to save as a list if we're saving as fits
        - output_dir: if saving as fits file, where to save them 
        - full_frame (bool, optional): If True, a full-frame detector image will be generated.
        - loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - point_sources_info (list): A list of dictionaries, each representing an off-axis point source in the scene.
                             
    Returns:
        - list[corgisim.scene.SimulatedImage]: A list of SimulatedImage objects, representing the complete observation scenario across all visits defined in the CPGS file.
    """
    # Get the detector, scene and optics used in generate obeservation sequence from CPGS file
    simulatedImage_list = []
    # Try to get target and reference
    try:
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list,satellite_dict_target, satellite_dict_reference = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list, satellite_dict_target = inputs.load_cpgs_data(filepath,output_dim=output_dim, polaxis=polaxis, fast_gain_mode=fast_gain_mode, gain_CIC_Q=gain_CIC_Q)

    if point_source_info is not None:
        host_star_properties = {'Vmag': scene_target._host_star_Vmag, 'spectral_type': scene_target._host_star_sptype, 'magtype': scene_target._host_star_magtype, 'ref_flag': False}
        scene_target = scene.Scene(host_star_properties, point_source_info)

    #Satellit spot configuration
    satspots_are_present = (satellite_dict_target is not None) 
    if satspots_are_present:
        contrast1 = 1e-7
        contrast2 = 1e-5
        sep1 = 6.25
        sep2 = 13
        match satellite_dict_target['satellite_spot_conf']:
            case 0:
                sep_lamD = sep1
                angle_deg = [0,90]
                contrast = contrast1
            case 1:
                sep_lamD = sep2
                angle_deg = [0,90]
                contrast = contrast1
            case 2:
                sep_lamD = sep1
                angle_deg = [45,135]
                contrast = contrast1
            case 3:
                sep_lamD = sep2
                angle_deg = [45,135]
                contrast = contrast1
            case 4:
                sep_lamD = sep1
                angle_deg = [0,90]
                contrast = contrast2
            case 5:
                sep_lamD = sep2
                angle_deg = [0,90]
                contrast = contrast2
            case 6:
                sep_lamD = sep1
                angle_deg = [45,135]
                contrast = contrast2
            case 7:
                sep_lamD = sep2
                angle_deg = [45,135]
                contrast = contrast2
            case _:
                raise KeyError('Unknown satellite spots configuration')

        satspot_keywords = {'num_pairs':2, 'sep_lamD': sep_lamD, 'angle_deg': angle_deg, 'contrast': contrast}

        detector_satspots_target = instrument.CorgiDetector(emccd_keywords={'em_gain':satellite_dict_target['satellite_spots_gain']}, photon_counting=False) 
        if 'satellite_dict_reference' in locals():
            detector_satspots_reference = instrument.CorgiDetector(emccd_keywords={'em_gain':satellite_dict_reference['satellite_spots_gain']}, photon_counting=False) 

    prihdr, exthdr = mocks.create_default_L1_headers()
    visit_id_root = prihdr['PROGNUM']+ prihdr['EXECNUM']+prihdr['CAMPAIGN']+prihdr['SEGMENT']+prihdr['OBSNUM']
    for visit in visit_list:
        visit_id = visit_id_root + visit['visit_num']
        # make a new subfolder for each visit 
        subfolder_name = 'V' + visit_id
        subfolder_path =  os.path.join(str(output_dir or ''), subfolder_name)
        if save_as_fits:
            os.makedirs(subfolder_path, exist_ok=True)

        optics.roll_angle = visit['roll_angle']
        simulatedImage_visit = []
        if visit['isReference']:
            simulatedImage_visit = generate_observation_sequence(scene_reference, optics, detector_reference,  visit['exp_time'], visit['number_of_frames'], 
                                                                satspots_are_present = satspots_are_present, satspots_detector = detector_satspots_reference, satspots_exptime = satellite_dict_reference['satellite_spots_frame_time'], satspots_number_of_frames = satellite_dict_reference['satellite_spots_number_of_frames'], satspot_keywords = satspot_keywords,
                                                                visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y , write_as_L1= write_as_L1)

        else:
            simulatedImage_visit = generate_observation_sequence(scene_target, optics, detector_target,  visit['exp_time'], visit['number_of_frames'], 
                                                                satspots_are_present = satspots_are_present, satspots_detector = detector_satspots_target, satspots_exptime = satellite_dict_target['satellite_spots_frame_time'], satspots_number_of_frames = satellite_dict_target['satellite_spots_number_of_frames'], satspot_keywords = satspot_keywords,
                                                                visit_id= visit_id, vistype= visit['vistype'], save_as_fits= save_as_fits, output_dir=subfolder_path, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y, write_as_L1= write_as_L1 )


        if not save_as_fits or (save_as_fits and save_as_list): # If we are writing the files, we are not storing the images unless explicitly told to do so
            simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list

    