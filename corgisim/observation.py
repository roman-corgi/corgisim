### Functions that will be run to simulate an observation. 
## This will likely contain functions simmilar to the functionality in Jorge's corgisims_obs 
import corgisim
import os
from corgisim import scene, instrument, inputs, observation, outputs, wavefront_estimation
import copy

def check_cpgs_os11_compatibility(visit_list):
    """Check the compatibility of the CPGS observation scenario with the OS11 scenario.
    Raise ValueError if the CPGS is not compatible.

    Args:
        - visit_list (list): List of dictionaries containing information about the CPGS visits.

    """

    ref_angles = [
        v["roll_angle"]
        for v in visit_list
        if v["isReference"]
    ]

    if len(set(ref_angles)) > 1:
        raise ValueError(
            f"Reference visits have inconsistent "
            f"position angles: {sorted(set(ref_angles))}"
        )

    ref_indices = [
        i for i, v in enumerate(visit_list)
        if v["isReference"]
    ]

    for i in range(len(ref_indices) - 1):

        n_between = (
                ref_indices[i + 1]
                - ref_indices[i]
                - 1
        )

        if n_between > 4:
            raise ValueError(
                f"More than 4 science visits "
                f"between reference visits "
                f"{visit_list[ref_indices[i]]['visit_id']} "
                f"and "
                f"{visit_list[ref_indices[i + 1]]['visit_id']}"
            )
    print("CPGS scenario compatible with os11, adding the speckle variation to the time series")
    return 1

def add_batch_info_to_visit(visit_list):
    """Add the batch information to each visit of the CPGS visit_list to make it correspond with .

        Args:
            - visit_list (list): List of dictionaries containing information about the CPGS visits.
        Returns:
            - visit_list (list): same list as the input but with batch information added to it.
    """

    batch_id = 0
    n_science = 0

    for v in visit_list:

        if v["isReference"]:

            # new batch
            batch_id += 1
            n_science = 0

            v["batch_id"] = batch_id
            v["roll_id"] = None

        else:

            n_science += 1

            roll_id = n_science

            v["batch_id"] = batch_id
            v["roll_id"] = roll_id

    return visit_list


def generate_observation_sequence(scene, optics, detector, exp_time, n_frames, save_as_fits= False, output_dir=None, full_frame= False, loc_x=None, loc_y=None, zernike_index=None, zernike_value_m=None):
    """Generates a sequence of simulated observations and places them on a detector.

    This function orchestrates the simulation of a given astrophysical scene through
    the instrument optics and then onto the detector. It first generates the host star's
    PSF, then injects any defined off-axis point sources into the simulated scene.
    Finally, it uses the detector model to create a detector image, optionally
    generating either a sub-array or a full-frame image for each exposure.
    Each observation sequence corresponds to a single visit at a specific roll angle.

    Args:
        scene (corgisim.scene.Scene): The scene object containing information about
            the host star and any specified point sources.
        optics (corgisim.instrument.CorgiOptics): The optics object defining the
            instrument configuration, including the telescope and coronagraph.
        detector (corgisim.instrument.CorgiDetector): The detector object defining
            the detector characteristics and noise properties.
        exp_time (float): The exposure time for each individual frame in seconds.
        n_frames (int): The total number of frames to generate in this observation sequence.
        full_frame (bool, optional): If True, a full-frame detector image will be generated.
            If False (default), a sub-array image is generated.
        loc_x (int, optional): The x-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            x-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.
        loc_y (int, optional): The y-coordinate for the center of the sub-array in pixels
            if `full_frame` is False. If `full_frame` is True, this specifies the
            y-coordinate of the full frame's origin (top-left pixel). Required if `full_frame` is True.
        zernike_index (List of int, optional): List of zernike polynomial index.
            If None, no speckles variation will be generated.
        zernike_value_m (numpy 2d array (n_index, n_frames), optional): List of the zernike wavefront aberration values in meters.
            If None, no speckles variation will be generated.

    Returns:
        list[corgisim.scene.SimulatedImage]: A list of :py:class:`corgisim.scene.SimulatedImage` objects,
        where each object represents a single generated observation frame with its image data
        and associated FITS header information.
    """

    sim_scene = optics.get_host_star_psf(scene)
    if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
        sim_scene = optics.inject_point_sources(scene,sim_scene)
    
    simulatedImage_list = []
    
    if full_frame == False :
        for i in range(0, n_frames):
            if zernike_index is not None:
                optics.optics_keywords.update({'zindex': zernike_index, 'zval_m': zernike_value_m[i]})
                sim_scene = optics.get_host_star_psf(scene)
                if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
                    sim_scene = optics.inject_point_sources(scene, sim_scene)

            sim_image = detector.generate_detector_image(sim_scene,exp_time)
            simulatedImage_list.append(copy.deepcopy(sim_image))
    else:
        if save_as_fits:
            # Save the images as fits in output_dir if specified, in corgisim/test/testdata if not
            # Simulation needs to be full frame to be written as L1
            if output_dir == None:
                local_path = corgisim.lib_dir
                outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
                print("No output directory specified. FITS files saved in ", outdir)
            else:
                outdir = output_dir

        for i in range(0, n_frames):
            if zernike_index is not None:
                optics.optics_keywords.update({'zindex': zernike_index, 'zval_m': zernike_value_m[i]})
                sim_scene = optics.get_host_star_psf(scene)
                if hasattr(scene, 'point_source_dra') or hasattr(scene, 'point_source_ddec'):
                    sim_scene = optics.inject_point_sources(scene, sim_scene)

            sim_image = detector.generate_detector_image(sim_scene,exp_time,full_frame=True,loc_x=loc_x, loc_y=loc_y)
            simulatedImage_list.append(copy.deepcopy(sim_image))

            if save_as_fits:
                outputs.save_hdu_to_fits(sim_image.image_on_detector,outdir=outdir, write_as_L1=True)

    return simulatedImage_list

def generate_observation_scenario_from_cpgs(filepath, save_as_fits= False, output_dir=None, full_frame=False, loc_x=None, loc_y=None, point_source_info=None, speckle_variation=False):
    """Generates an observation scenario by loading instrument, scene, and visit
    information from a CPGS file.

    This function attempts to load both target and reference star information.
    If only target information is available, it proceeds with that.

    Args:
        - filepath (str): The path to the CPGS XML file.
        - loc_x (int): The horizontal coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - loc_y (int): The vertical coordinate (in pixels) of the center where the sub_frame will be inserted, needed when full_frame=True, and image from CorgiOptics has size is smaller than 1024×1024
        - point_sources_info (list): A list of dictionaries, each representing an off-axis point source in the scene.
        - speckle_variation (bool): If True, time varying wavefront aberrations are added to the simulations.
                             
    Returns:
        - list[corgisim.scene.SimulatedImage]: A list of SimulatedImage objects, representing the complete observation scenario across all visits defined in the CPGS file.
    """
    # Get the detector, scene and optics used in generate obeservation sequence from CPGS file
    simulatedImage_list = []
    # Try to get target and reference
    try:
        scene_target, scene_reference, optics, detector_target, detector_reference, visit_list = inputs.load_cpgs_data(filepath)
    # If error, only get the target        
    except ValueError:
        scene_target, optics, detector_target, visit_list = inputs.load_cpgs_data(filepath)

    if point_source_info is not None:
        host_star_properties = {'Vmag': scene_target._host_star_Vmag, 'spectral_type': scene_target._host_star_sptype, 'magtype': scene_target._host_star_magtype, 'ref_flag': False}
        scene_target = scene.Scene(host_star_properties, point_source_info)

    if speckle_variation:
        check_cpgs_os11_compatibility(visit_list)
        visit_list = add_batch_info_to_visit(visit_list)

    for visit in visit_list:
        optics.roll_angle = visit['roll_angle']
        if visit['isReference']:
            if speckle_variation:
                zernike_index, zernike_value_m = wavefront_estimation.get_drift(exp_time=visit['exp_time'], n_frames=visit['number_of_frames'], obs='ref', cycle=visit['batch_id'])
            else:
                zernike_index, zernike_value_m = None, None
            simulatedImage_visit = generate_observation_sequence(scene_reference, optics, detector_reference, visit['exp_time'], visit['number_of_frames'],save_as_fits= save_as_fits, output_dir=output_dir, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y, zernike_index=zernike_index, zernike_value_m=zernike_value_m)
        else:
            if speckle_variation:
                zernike_index, zernike_value_m = wavefront_estimation.get_drift(exp_time=visit['exp_time'], n_frames=visit['number_of_frames'], obs='science', cycle=visit['batch_id'], roll=visit['roll_id'])
            else:
                zernike_index, zernike_value_m = None, None
            simulatedImage_visit = generate_observation_sequence(scene_target, optics, detector_target, visit['exp_time'], visit['number_of_frames'],save_as_fits= save_as_fits, output_dir=output_dir, full_frame= full_frame,loc_x=loc_x, loc_y=loc_y, zernike_index=zernike_index, zernike_value_m=zernike_value_m)

        simulatedImage_list.extend(simulatedImage_visit)

    return simulatedImage_list
