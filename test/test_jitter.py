import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
from corgisim import scene, instrument, inputs, observation, jitter
from corgisim.scene import SimulatedImage
from astropy.io import fits
import proper
import roman_preflight_proper
import pandas as pd
import os
"""
Jitter and Finite Stellar Diameter Tests:
    
    test_offsets_and_areas_against_example: tests use of jitter.py against an example
    test_obs_with_finite_stellar_diam: tests an observation sequence with finite stellar diameter included
    test_all_pol_obs_with_finite_stellar_diam: tests that all polarization cases run with finite stellar diameter included (except for the one case in test_minimal)
    test_jittered_weights: compares the calculated weights for a jitter+finite stellar case against an example
    test_obs_with_jitter: tests that a case with only jitter runs without issue
    test_pol_obs_with_finite_stellar_diam_and_jitter: tests a jitter+finite stellar diameter case
"""

def test_offsets_and_areas_against_example():
    '''
     This function tests the functions in jitter.py that calculate the x and
     y coordinates of the offsets and the normalized area of the region 
     represented by each offset against an example that approximately 
     reproduces the jitter offsets and regions shown in a figure in John
     Krist's paper.
    '''
    ###############################################################################
    # Precalculated example
    this_file_dir = os.path.dirname(__file__) # This file's folder
    filepath = 'test_data/example_jitter_data_offsets_and_areas.txt'
    abs_path = os.path.join(this_file_dir,filepath)
    example_data = pd.read_csv(abs_path)
    example_x_offsets = example_data['x_off']
    example_y_offsets = example_data['y_off']
    example_As = example_data['Anorm']
    
    # This example uses the following parameters for each ring of offsets:
    
    # Ring 0 (Centered at the origin)
    # Define the radius of the ring
    r_ring0 = 0.075
    
    # Ring 1 has 6 regions centered at a radius of 0.15 mas
    # This ring is centered on the y axis.
    # Specify the radial width of each region
    dr_ring1 = 0.15
    # Specify the angle of the first region center in degrees
    theta_ring1_centers_start = 90
    # Specify the number of regions
    regnum_ring1 = 6
    # Determine the ring parameters
    x_ring1_centers, y_ring1_centers,\
     r_ring1_centers, r_ring1_outer,\
     x_ring1_outer, yu_ring1_outer,\
     yl_ring1_outer, boundary_coords1 = jitter.Determine_ring_params(r_ring0,dr_ring1,regnum_ring1,theta_ring1_centers_start)
    
    # Ring 2 has 8 regions
    # This ring is centered on the x axis
    # Specify the radial width of each region
    dr_ring2 = 0.15
    # Specify the angle of the first region center in degrees
    theta_ring2_centers_start = 0
    # Specify the number of regions
    regnum_ring2 = 8
    # Determine the ring parameters
    x_ring2_centers, y_ring2_centers,\
     r_ring2_centers, r_ring2_outer,\
     x_ring2_outer, yu_ring2_outer,\
     yl_ring2_outer, boundary_coords2 = jitter.Determine_ring_params(r_ring1_outer,dr_ring2,regnum_ring2,theta_ring2_centers_start)
    
    # Ring 3 has 12 regions
    # This ring is not centered on either axis. It has region borders along the x and y axes
    # Specify the radial width of each region
    dr_ring3 = 0.15
    # Specify the number of regions
    regnum_ring3 = 12
    # Specify the angle of the first region center in degrees
    # In this case, there are three regions in each quadrant, so one region is centered at 45 degrees.
    theta_ring3_centers_start = 45
    # Determine the ring parameters
    x_ring3_centers, y_ring3_centers,\
     r_ring3_centers, r_ring3_outer,\
     x_ring3_outer, yu_ring3_outer,\
     yl_ring3_outer, boundary_coords3 = jitter.Determine_ring_params(r_ring2_outer,dr_ring3,regnum_ring3,theta_ring3_centers_start)
    
    # Ring 4 has 14 regions
    # This ring is centered on the x axis.
    # Specify the radial width of each region
    dr_ring4 = 0.15
    # Specify the number of regions
    regnum_ring4 = 14
    # Specify the angle of the first region center in degrees
    theta_ring4_centers_start = 0
    # Determine the ring parameters
    x_ring4_centers, y_ring4_centers,\
     r_ring4_centers, r_ring4_outer,\
     x_ring4_outer, yu_ring4_outer,\
     yl_ring4_outer, boundary_coords4 = jitter.Determine_ring_params(r_ring3_outer,dr_ring4,regnum_ring4,theta_ring4_centers_start)
     
    # Ring 5 has 12 regions.
    # This ring has three regions per quadrant, so one region is centered at 45 degrees.
    # Specify the radial width of each region
    dr_ring5 = 0.2
    # Specify the number of regions
    regnum_ring5 = 12
    # Specify the angle of the first region center in degrees
    theta_ring5_centers_start = 45
    # Determine the ring parameters
    x_ring5_centers, y_ring5_centers,\
     r_ring5_centers, r_ring5_outer,\
     x_ring5_outer, yu_ring5_outer,\
     yl_ring5_outer, boundary_coords5 = jitter.Determine_ring_params(r_ring4_outer,dr_ring5,regnum_ring5,theta_ring5_centers_start)
     
    # Ring 6 has 10 regions.
    # This ring is centered on the x axis.
    # Specify the radial width of each region
    dr_ring6 = 0.4
    # Specify the number of regions
    regnum_ring6 = 10
    # Specify the angle of the first region center in degrees
    theta_ring6_centers_start = 0
    # Determine the ring parameters
    x_ring6_centers, y_ring6_centers,\
     r_ring6_centers, r_ring6_outer,\
     x_ring6_outer, yu_ring6_outer,\
     yl_ring6_outer, boundary_coords6 = jitter.Determine_ring_params(r_ring5_outer,dr_ring6,regnum_ring6,theta_ring6_centers_start)
     
    # Ring 7 has 14 regions.
    # This ring is centered on the y axis.
    # Specify the radial width of each region
    dr_ring7 = 0.4
    # Specify the number of regions
    regnum_ring7 = 14
    # Specify the angle of the first region center in degrees
    theta_ring7_centers_start = 90
    # Determine the ring parameters
    x_ring7_centers, y_ring7_centers,\
     r_ring7_centers, r_ring7_outer,\
     x_ring7_outer, yu_ring7_outer,\
     yl_ring7_outer, boundary_coords7 = jitter.Determine_ring_params(r_ring6_outer,dr_ring7,regnum_ring7,theta_ring7_centers_start)
    
    # Ring 8 has 10 regions.
    # This ring is centered on the x axis.
    # Specify the radial width of each region
    dr_ring8 = 0.8
    # Specify the number of regions
    regnum_ring8 = 10
    # Specify the angle of the first region center in degrees
    theta_ring8_centers_start = 0
    # Determine the ring parameters
    x_ring8_centers, y_ring8_centers,\
     r_ring8_centers, r_ring8_outer,\
     x_ring8_outer, yu_ring8_outer,\
     yl_ring8_outer, boundary_coords8 = jitter.Determine_ring_params(r_ring7_outer,dr_ring8,regnum_ring8,theta_ring8_centers_start)
    
    # Ring 9 has 14 regions.
    # This ring is centered on the y axis.
    # Specify the radial width of each region
    dr_ring9 = 0.8
    # Specify the number of regions
    regnum_ring9 = 14
    # Specify the angle of the first region center in degrees
    theta_ring9_centers_start = 90
    # Determine the ring parameters
    x_ring9_centers, y_ring9_centers,\
     r_ring9_centers, r_ring9_outer,\
     x_ring9_outer, yu_ring9_outer,\
     yl_ring9_outer, boundary_coords9 = jitter.Determine_ring_params(r_ring8_outer,dr_ring9,regnum_ring9,theta_ring9_centers_start)
    
    # Ring 10 has 10 regions.
    # This ring is centered on the x axis.
    # Specify the radial width of each region
    dr_ring10 = 1.6
    # Specify the number of regions
    regnum_ring10 = 10
    # Specify the angle of the first region center in degrees
    theta_ring10_centers_start = 0
    # Determine the ring parameters
    x_ring10_centers, y_ring10_centers,\
     r_ring10_centers, r_ring10_outer,\
     x_ring10_outer, yu_ring10_outer,\
     yl_ring10_outer, boundary_coords10 = jitter.Determine_ring_params(r_ring9_outer,dr_ring10,regnum_ring10,theta_ring10_centers_start)
    
    # Ring 11 has 14 regions.
    # This ring is centered on the y axis.
    # Specify the radial width of each region
    dr_ring11 = 1.6
    # Specify the number of regions
    regnum_ring11 = 14
    # Specify the angle of the first region center in degrees
    theta_ring11_centers_start = 90
    # Determine the ring parameters
    x_ring11_centers, y_ring11_centers,\
     r_ring11_centers, r_ring11_outer,\
     x_ring11_outer, yu_ring11_outer,\
     yl_ring11_outer, boundary_coords11 = jitter.Determine_ring_params(r_ring10_outer,dr_ring11,regnum_ring11,theta_ring11_centers_start)
    ###############################################################################
    # Test that the manually specified regions match those calculated by
    # jitter.Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, \
    #                                    N_offsets_per_ring,starting_offset_ang_by_ring,\
    #                                    r_ring0=0.075,dr_rings=None)
    
    # Define the function inputs to match the example above
    N_rings_of_offsets = 11
    N_offsets_per_ring = np.array([regnum_ring1,regnum_ring2,regnum_ring3,regnum_ring4,regnum_ring5,regnum_ring6,regnum_ring7,regnum_ring8,regnum_ring9,regnum_ring10,regnum_ring11])
    starting_offset_ang_by_ring = np.array([theta_ring1_centers_start,theta_ring2_centers_start,theta_ring3_centers_start,theta_ring4_centers_start,theta_ring5_centers_start,theta_ring6_centers_start,theta_ring7_centers_start,theta_ring8_centers_start,theta_ring9_centers_start,theta_ring10_centers_start,theta_ring11_centers_start])
    r_ring0=0.075
    dr_rings=np.array([dr_ring1,dr_ring2,dr_ring3,dr_ring4,dr_ring5,dr_ring6,dr_ring7,dr_ring8,dr_ring9,dr_ring10,dr_ring11])
    outer_radius_of_offset_circle = r_ring0+np.sum(dr_rings)
    
    # Use the function to calculate the parameters for all of the regions in all of the rings
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict,yl_outer_dict, boundary_coords_dict = \
         jitter.Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, N_offsets_per_ring, starting_offset_ang_by_ring,r_ring0,dr_rings)
         
    # For easier iterating, reshape the offsets and areas into lists
    x_offsets_list = [];
    y_offsets_list = [];
    A_offsets_list = [];
    for iring in np.arange(11+1):
        # Extract the data for the ring
        y_offsets_iring = y_offsets[iring]
        x_offsets_iring = x_offsets[iring]
        A_offsets_iring = A_offsets[iring]
        # The zeroth ring has only one data point (the onaxis source)
        if iring==0:
            # Append the data for the zeroth ring
            x_offsets_list = np.append(x_offsets_list,x_offsets_iring)
            y_offsets_list = np.append(y_offsets_list,y_offsets_iring)
            A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
        else:
            # The remaining rings have multiple data points
            # Iterate over each region
            Nregions_iring = x_offsets_iring.shape[0]
            for ireg in range(Nregions_iring):
                # Append the data for region ireg in ring iring
                x_offsets_list = np.append(x_offsets_list,x_offsets_iring[ireg])
                y_offsets_list = np.append(y_offsets_list,y_offsets_iring[ireg])
                A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
         
    # Check that the offsets and areas match those calculated above
    assert np.allclose(example_x_offsets,x_offsets_list)
    assert np.allclose(example_y_offsets,y_offsets_list)
    assert np.allclose(r_ring0**2/outer_radius_of_offset_circle**2,A_offsets_list[0])
    assert np.allclose(example_As[1:-1],A_offsets_list[1:-1])
    
    # The normalized area should be nearly equal to 1, allowing for some rounding error
    total_area_norm = np.sum(A_offsets_list)
    assert np.allclose(total_area_norm,1.0,atol=0.1) == True
    
###############################################################################
def test_finite_stellar_diam_weights():
    '''
    This function checks that sum(W_i * A_i) = 1 for the finite stellar diameter
    case, where W_i is the weight of the ith offset source and A_i is the area
    of the region represented by the ith offset source.
    In the specific case where the star has a finite diameter and where we are
    not considering jitter, the weights should all be 1 or 0 (1 for points inside
    the stellar disk, 0 for points outside). They should *not* be normalized so 
    that the sum of all the weights equals 1; normalizing the weights in this
    manner leads to multiplying the PSF by 1/N_offsets.
    '''
    # Define the stellar diameter parameters
    stellar_diam_and_jitter_keywords = {}
    stellar_diam_mas = 5
    stellar_diam_and_jitter_keywords['stellar_diam_mas'] = stellar_diam_mas
    # Offset array parameters
    stellar_diam_and_jitter_keywords['N_rings_of_offsets'] = 5
    stellar_diam_and_jitter_keywords['r_ring0'] = (stellar_diam_mas/2)/(stellar_diam_and_jitter_keywords['N_rings_of_offsets']+1)
    stellar_diam_and_jitter_keywords['dr_rings'] = stellar_diam_and_jitter_keywords['r_ring0']*np.ones(stellar_diam_and_jitter_keywords['N_rings_of_offsets'])
    stellar_diam_and_jitter_keywords['N_offsets_per_ring'] = np.array([3,5,7,9,11])
    stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'] = np.array([90,0,45,0,45])
    stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'] = stellar_diam_and_jitter_keywords['r_ring0'] + np.sum(stellar_diam_and_jitter_keywords['dr_rings'])
    stellar_diam_and_jitter_keywords['use_finite_stellar_diam'] = 1
    stellar_diam_and_jitter_keywords['add_jitter'] = 0
    stellar_diam_and_jitter_keywords['use_saved_deltaE_and_weights'] = 0
    stellar_diam_and_jitter_keywords['N_offsets_counting_origin'] = np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'])+1
    
    # Calculate the parameters for all of the regions in all of the rings
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict,yl_outer_dict, boundary_coords_dict = \
    jitter.Determine_offsets_and_areas(stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'],
                                       stellar_diam_and_jitter_keywords['N_rings_of_offsets'], 
                                       stellar_diam_and_jitter_keywords['N_offsets_per_ring'], 
                                       stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'],
                                       stellar_diam_and_jitter_keywords['r_ring0'],
                                       stellar_diam_and_jitter_keywords['dr_rings'])
         
    # For easier iterating, reshape the offsets and areas into lists
    x_offsets_list = [];
    y_offsets_list = [];
    A_offsets_list = [];
    for iring in np.arange(stellar_diam_and_jitter_keywords['N_rings_of_offsets']+1):
        # Extract the data for the ring
        y_offsets_iring = y_offsets[iring]
        x_offsets_iring = x_offsets[iring]
        A_offsets_iring = A_offsets[iring]
        # The zeroth ring has only one data point (the onaxis source)
        if iring==0:
            # Append the data for the zeroth ring
            x_offsets_list = np.append(x_offsets_list,x_offsets_iring)
            y_offsets_list = np.append(y_offsets_list,y_offsets_iring)
            A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
        else:
            # The remaining rings have multiple data points
            # Iterate over each region
            Nregions_iring = x_offsets_iring.shape[0]
            for ireg in range(Nregions_iring):
                # Append the data for region ireg in ring iring
                x_offsets_list = np.append(x_offsets_list,x_offsets_iring[ireg])
                y_offsets_list = np.append(y_offsets_list,y_offsets_iring[ireg])
                A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
                
    # Store the offsets and areas in an offset_field_data dictionary
    offset_field_data = {'x_offsets_mas':x_offsets_list,\
                         'y_offsets_mas':y_offsets_list,\
                         'A_offsets':A_offsets_list}
    # and add to stellar_diam_and_jitter_keywords
    stellar_diam_and_jitter_keywords['offset_field_data'] = offset_field_data

    # Calculate the weight for each of these offsets
    stellar_diam_and_jitter_keywords = jitter.calculate_weights_for_jitter_and_finite_stellar_diameter(stellar_diam_and_jitter_keywords)
    
    # The returned offset weights are equal to W_i * Anorm_i.
    # Check that the sum of these weights is close to 1.
    assert(np.allclose(np.sum(stellar_diam_and_jitter_keywords['offset_field_data']['offset_weights']),1.0))

###############################################################################
def test_obs_with_finite_stellar_diam():
    '''
     This function tests running an observation sequence with finite stellar diameter included.
     
    '''
    # Set up keywords and parameters that do not change
    # optics keywords
    Vmag = 8
    sptype = 'G0V'
    stellar_diam_mas = 10 # arbitrary for this test
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}
    
    # Set up the detector
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    
    #--------------------------------------------------------------------------
    # Simulation with finite stellar diameter included
    
    # jitter and finite stellar diameter keywords
    stellar_diam_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(quicktest=True,stellar_diam_mas=stellar_diam_mas)
    
    # Define the scene
    host_star_properties_disk = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag','stellar_diam_mas':stellar_diam_mas}
    base_scene_disk = scene.Scene(host_star_properties_disk)
    
    # Set up the optics
    optics_disk =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, stellar_diam_and_jitter_keywords=stellar_diam_keywords, if_quiet=True)
    
    # Test a single frame 
    n_frames = 1
    simulatedImage_list_disk = observation.generate_observation_sequence(base_scene_disk, optics_disk, detector, exp_time, n_frames)
    host_star_image_disk = simulatedImage_list_disk[0].host_star_image.data
    
    #--------------------------------------------------------------------------
    # Same simulation without finite stellar diameter
    host_star_properties_point = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    base_scene_point = scene.Scene(host_star_properties_point)
    optics_point =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    simulatedImage_list_point = observation.generate_observation_sequence(base_scene_point, optics_point, detector, exp_time, n_frames)
    host_star_image_point = simulatedImage_list_point[0].host_star_image.data
    #--------------------------------------------------------------------------
    # Check that the maximum intensity is lower for the disk than the point
    # (The total intensity is the same for both cases, but for the disk, the
    #  intensity is spread out over the surface of the disk, making the max
    #  lower.)
    Imax_point = np.max(host_star_image_point[:])
    Imax_disk = np.max(host_star_image_disk[:])
    assert Imax_point > Imax_disk
###############################################################################
def test_all_pol_obs_with_finite_stellar_diam():
    '''
    Test that the calculations run for optics.prism = 'POL0',
    optics.prism = 'POL45', and polaxis = -10
    '''
    # Set up keywords
    # optics keywords
    Vmag = 8
    sptype = 'G0V'
    stellar_diam_mas = 10 # Arbitrary for the purposes of this test
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag','stellar_diam_mas':stellar_diam_mas}
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}
    
    # Set up the detectior
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    n_frames = 1
    
    # jitter and finite stellar diameter keywords
    # need a clean set for each polarization
    stellar_diam_keywords_m10 = jitter.load_predefined_jitter_and_stellar_diam_params(mintest=True,stellar_diam_mas=stellar_diam_mas)
    stellar_diam_keywords_pol0 = jitter.load_predefined_jitter_and_stellar_diam_params(mintest=True,stellar_diam_mas=stellar_diam_mas)
    stellar_diam_keywords_pol45 = jitter.load_predefined_jitter_and_stellar_diam_params(mintest=True,stellar_diam_mas=stellar_diam_mas)
    
    # Define the scene
    base_scene = scene.Scene(host_star_properties)
    
    # For polaxis = -10
    # Set up the optics
    optics_keywords_m10 ={'cor_type':cor_type, 'use_errors':1, 'polaxis':-10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_m10 =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords_m10, stellar_diam_and_jitter_keywords=stellar_diam_keywords_m10, if_quiet=True)
    simulatedImage_list_polaxism10 = observation.generate_observation_sequence(base_scene, optics_m10, detector, exp_time, n_frames)
          
    # For pol0
    prism = 'POL0'
    optics_keywords_0_90 ={'cor_type':cor_type, 'use_errors':1, 'polaxis':-10, 'output_dim':201,'prism':prism,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_0_90 =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords_0_90, stellar_diam_and_jitter_keywords=stellar_diam_keywords_pol0, if_quiet=True)
    simulatedImage_list_0_90 = observation.generate_observation_sequence(base_scene, optics_0_90, detector, exp_time, n_frames)
        
    # For pol45
    prism = 'POL45'
    optics_keywords_45_135 ={'cor_type':cor_type, 'use_errors':1, 'polaxis':-10, 'output_dim':201,'prism':prism,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_45_135 =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords_45_135, stellar_diam_and_jitter_keywords=stellar_diam_keywords_pol45, if_quiet=True)
    simulatedImage_list_45_135 = observation.generate_observation_sequence(base_scene, optics_45_135, detector, exp_time, n_frames)
   
###############################################################################
def test_jittered_weights():
    '''
    This function compares the calculated weights against an OS11-based 
    example that includes both jitter and a finite stellar diameter. The
    OS11-based example uses an offset source distribtuion that approximately 
    reproduces the offsets and regions shown in a figure in John Krist's paper.
    '''
    
    # Load the example stellar diameter and jitter parameter dictionary
    stellar_diam_and_jitter_keywords = jitter.load_predefined_jitter_and_stellar_diam_params()
    # Change 'add_jitter' to 1 because the example includes jitter
    stellar_diam_and_jitter_keywords['add_jitter']=1
    # Add the stellar diameter for the example
    stellar_diam_and_jitter_keywords['stellar_diam_mas'] = 0.9
    # Add the number of offsets counting the origin
    stellar_diam_and_jitter_keywords['N_offsets_counting_origin'] = np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'])+1
    
    # Calculate the offset locations and the area of each region
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, \
    yl_outer_dict, boundary_coords_dict = \
    jitter.Determine_offsets_and_areas(stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'],\
                                       stellar_diam_and_jitter_keywords['N_rings_of_offsets'], \
                                       stellar_diam_and_jitter_keywords['N_offsets_per_ring'],\
                                       stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'],\
                                       stellar_diam_and_jitter_keywords['r_ring0'],\
                                       stellar_diam_and_jitter_keywords['dr_rings'])
    # For easier iterating, reshape the offsets and areas into lists
    x_offsets_list = [];
    y_offsets_list = [];
    A_offsets_list = [];
    for iring in np.arange(stellar_diam_and_jitter_keywords['N_rings_of_offsets']+1):
        # Extract the data for the ring
        x_offsets_iring = x_offsets[iring]
        y_offsets_iring = y_offsets[iring]
        A_offsets_iring = A_offsets[iring]
        # The zeroth ring has only one data point (the onaxis source)
        if iring==0:
            # Append the data for the zeroth ring
            x_offsets_list = np.append(x_offsets_list,x_offsets_iring)
            y_offsets_list = np.append(y_offsets_list,y_offsets_iring)
            A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
        else:
            # The remaining rings have multiple data points
            # Iterate over each region
            Nregions_iring = x_offsets_iring.shape[0]
            for ireg in range(Nregions_iring):
                # Append the data for region ireg in ring iring
                x_offsets_list = np.append(x_offsets_list,x_offsets_iring[ireg])
                y_offsets_list = np.append(y_offsets_list,y_offsets_iring[ireg])
                A_offsets_list = np.append(A_offsets_list,A_offsets_iring)
                
    # Store the offsets and areas in an offset_field_data dictionary
    offset_field_data = {'x_offsets_mas':x_offsets_list,\
                         'y_offsets_mas':y_offsets_list,\
                         'A_offsets':A_offsets_list}
    # and add to stellar_diam_and_jitter_keywords
    stellar_diam_and_jitter_keywords['offset_field_data'] = offset_field_data
    
    # Define the expected RMS jitter values 
    this_file_dir = os.path.dirname(__file__) # this file's folder
    filepath = 'test_data/OS11inputs.csv'
    abs_path = os.path.join(this_file_dir,filepath)
    OS11inputs = pd.read_csv(abs_path)
    # use just the entries for t0 for this test
    xRMS = OS11inputs['x RMS jitter'][0]
    yRMS = OS11inputs['y RMS jitter'][0]
    # add these values to the stellar diameter and jitter dictionary
    stellar_diam_and_jitter_keywords['jitter_sigmax'] = xRMS
    stellar_diam_and_jitter_keywords['jitter_sigmay'] = yRMS
    
    # Calculate the weights
    stellar_diam_and_jitter_keywords = jitter.calculate_weights_for_jitter_and_finite_stellar_diameter(stellar_diam_and_jitter_keywords)
    calculated_weights = stellar_diam_and_jitter_keywords['offset_field_data']['offset_weights']
    
    # Load the example weights
    filepath_weights = 'test_data/example_jitter_weights_for_timeseries.txt'
    abs_path_weights = os.path.join(this_file_dir,filepath_weights)
    example_weights_OS11 = np.loadtxt(abs_path_weights,skiprows=1)
    # use only the first row for t0
    example_weights_t0 = example_weights_OS11[0]
    
    # Compare the two sets of weights
    assert(np.allclose(calculated_weights[1:-1],example_weights_t0[1:-1]))
###############################################################################
def test_obs_with_jitter():
    '''
    This function checks that a case with only jitter runs without issue.
    '''
    # Set up keywords and parameters that do not change
    # optics keywords
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords ={'cor_type':cor_type, 'use_errors':1, 'polaxis':10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}
    
    # Set up the detector
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    
    #--------------------------------------------------------------------------
    # Simulation with jitter included
    
    # jitter and finite stellar diameter keywords
    jitter_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(quicktest=True,stellar_diam_mas=0)
    jitter_keywords['add_jitter'] = 1
    jitter_keywords['use_finite_stellar_diam'] = 0
    jitter_keywords['jitter_sigmax'] = 0.3172369069947508
    jitter_keywords['jitter_sigmay'] = 0.3080179283580678
    
    # Define the scene
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    base_scene = scene.Scene(host_star_properties)
    
    # Set up the optics
    optics_jitter =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, stellar_diam_and_jitter_keywords=jitter_keywords, if_quiet=True)
    
    # Test a single frame 
    n_frames = 1
    simulatedImage_list_jitter = observation.generate_observation_sequence(base_scene, optics_jitter, detector, exp_time, n_frames)
    host_star_image_jitter = simulatedImage_list_jitter[0].host_star_image.data
    
    #--------------------------------------------------------------------------
    # Same simulation without jitter
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    host_star_image = simulatedImage_list[0].host_star_image.data
    #--------------------------------------------------------------------------
    # Expect that jitter will reduce the maximum intensity
    assert(np.max(host_star_image)>np.max(host_star_image_jitter))
###############################################################################
def test_pol_obs_with_finite_stellar_diam_and_jitter():
    '''
    Test that the calculations run for optics.prism = 'POL0'
    '''
    # Set up keywords
    # optics keywords
    Vmag = 8
    sptype = 'G0V'
    stellar_diam_mas = 10 # Arbitrary for the purposes of this test
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'
    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag','stellar_diam_mas':stellar_diam_mas}
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}
    
    # Set up the detectior
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    n_frames = 1
    
    # jitter and finite stellar diameter keywords
    # need a clean set for each polarization
    stellar_diam_keywords_pol0 = jitter.load_predefined_jitter_and_stellar_diam_params(quicktest=True,stellar_diam_mas=stellar_diam_mas)
    # modify the dictionary so that jitter will be included
    stellar_diam_keywords_pol0['add_jitter'] = 1
    stellar_diam_keywords_pol0['jitter_sigmax'] = 0.3172369069947508
    stellar_diam_keywords_pol0['jitter_sigmay'] = 0.3080179283580678
    
    # Define the scene
    base_scene = scene.Scene(host_star_properties)
    
    # For pol0
    prism = 'POL0'
    optics_keywords_0_90 ={'cor_type':cor_type, 'use_errors':1, 'polaxis':-10, 'output_dim':201,'prism':prism,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics_0_90 =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords_0_90, stellar_diam_and_jitter_keywords=stellar_diam_keywords_pol0, if_quiet=True)
    simulatedImage_list_0_90 = observation.generate_observation_sequence(base_scene, optics_0_90, detector, exp_time, n_frames) 
    host_star_image_with_diam_and_jitter = simulatedImage_list_0_90[0].host_star_image.data
    
    # Also look at the jitter-free point-star case
    optics_basic = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, oversampling_factor=3, optics_keywords=optics_keywords_0_90, if_quiet=True)
    simulatedImage_list_basic = observation.generate_observation_sequence(base_scene, optics_basic, detector, exp_time, n_frames)
    host_star_image_basic = simulatedImage_list_basic[0].host_star_image.data
    
    # Check that the peak intensity is higher for the jitter-free point-star case
    assert(np.max(host_star_image_basic)>np.max(host_star_image_with_diam_and_jitter))
###############################################################################
if __name__ == '__main__':
    test_offsets_and_areas_against_example()
    test_finite_stellar_diam_weights()
    test_obs_with_finite_stellar_diam()
    test_all_pol_obs_with_finite_stellar_diam()
    test_jittered_weights()
    test_obs_with_jitter()
    test_pol_obs_with_finite_stellar_diam_and_jitter()
