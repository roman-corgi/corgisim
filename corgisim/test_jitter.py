import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
from corgisim import scene, instrument, inputs, observation, jitter
from corgisim.scene import SimulatedImage
from astropy.io import fits
import proper
import roman_preflight_proper
import pytest
from scipy.interpolate import RectBivariateSpline
"""
Jitter and Finite Stellar Diameter Tests:
    
    test_offsets_and_areas_against_example: tests use of jitter.py against an example
    check_offset_weights: compares the calculated weights for the example offset list
                          against a pre-calculated list for the same offsets (TODO)
    test_obs_with_finite_stellar_diam: tests an observation sequence with finite stellar diameter included
    test_weight_calculation: shorter version of test_obs_with_finite_stellar_diam used to test the
                             addition of the lines that calculate the weights for the offset regions
    basic_weight_calculation_test: tests the interpolation method that replaces interp2d when calculating
                                   the weights

"""

def test_offsets_and_areas_against_example(show_plots=False,print_details=False):
    '''
     This function tests the functions in jitter.py that calculate the x and
     y coordinates of the offsets and the normalized area of the region 
     represented by each offset against an example that approximately 
     reproduces the jitter offsets and regions shown in a figure in John
     Krist's paper.
    '''
    ###############################################################################
    # Specify the parameters for each ring of offsets
    
    # Ring 0 (Centered at the origin)
    # Define the radius of the ring
    r_ring0 = 0.075
    # Define the set of x coordinates for the ring outline
    x_ring0 = np.linspace(-r_ring0,r_ring0)
    # Define the sets of y coordinates for the ring outline
    yu_ring0 = jitter.Find_ycircle(r_ring0,x_ring0,0,0,'u')
    yl_ring0 = jitter.Find_ycircle(r_ring0,x_ring0,0,0,'l')
    
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
    # Plot the rings of offsets and their regions if desired
    if show_plots == True:
    
        # Set up the figure window
        fig,ax = plt.subplots()
        
        # Plot the Ring 0 region and jitter offset
        # Define the circle plot for Ring 0
        line_r0u, = ax.plot(x_ring0,yu_ring0, color='black')
        line_r0l, = ax.plot(x_ring0,yl_ring0, color='black')
        # Also add a dot for the offset for Ring 0
        line_50c, = ax.plot(0,0,'o', color = 'k')
        
        # Plot the Ring 1 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring1_centers,y_ring1_centers,\
                                              x_ring1_outer,yu_ring1_outer,yl_ring1_outer,\
                                              boundary_coords1,regnum_ring1,fig,ax)
        
        # Plot the Ring 2 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring2_centers,y_ring2_centers,\
                                              x_ring2_outer,yu_ring2_outer,yl_ring2_outer,\
                                              boundary_coords2,regnum_ring2,fig,ax)
                                              
        # Plot the Ring 3 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring3_centers,y_ring3_centers,\
                                              x_ring3_outer,yu_ring3_outer,yl_ring3_outer,\
                                              boundary_coords3,regnum_ring3,fig,ax)
                                              
        # Plot the Ring 4 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring4_centers,y_ring4_centers,\
                                              x_ring4_outer,yu_ring4_outer,yl_ring4_outer,\
                                              boundary_coords4,regnum_ring4,fig,ax)
                                              
        # Plot the Ring 5 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring5_centers,y_ring5_centers,\
                                              x_ring5_outer,yu_ring5_outer,yl_ring5_outer,\
                                              boundary_coords5,regnum_ring5,fig,ax)
        
        # Plot the Ring 6 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring6_centers,y_ring6_centers,\
                                              x_ring6_outer,yu_ring6_outer,yl_ring6_outer,\
                                              boundary_coords6,regnum_ring6,fig,ax)
        
        # Plot the Ring 7 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring7_centers,y_ring7_centers,\
                                              x_ring7_outer,yu_ring7_outer,yl_ring7_outer,\
                                              boundary_coords7,regnum_ring7,fig,ax)
        
        # Plot the Ring 8 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring8_centers,y_ring8_centers,\
                                              x_ring8_outer,yu_ring8_outer,yl_ring8_outer,\
                                              boundary_coords8,regnum_ring8,fig,ax)
        
        # Plot the Ring 9 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring9_centers,y_ring9_centers,\
                                              x_ring9_outer,yu_ring9_outer,yl_ring9_outer,\
                                              boundary_coords9,regnum_ring9,fig,ax)
        
        # Plot the Ring 10 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring10_centers,y_ring10_centers,\
                                              x_ring10_outer,yu_ring10_outer,yl_ring10_outer,\
                                              boundary_coords10,regnum_ring10,fig,ax)
        
        # Plot the Ring 11 regions and jitter offsets
        jitter.Plot_Offsets_And_Region_Outlines(x_ring11_centers,y_ring11_centers,\
                                              x_ring11_outer,yu_ring11_outer,yl_ring11_outer,\
                                              boundary_coords11,regnum_ring11,fig,ax)
        
        
        # Make the plot window appear
        plt.axis('equal')
        plt.xlim([-6.5,6.5])
        plt.ylim([-6.5,6.5])
        plt.show()
    ###############################################################################
    # Calculate the total number of regions (offsets)
    
    regnum_total = regnum_ring1 + regnum_ring2 + regnum_ring3 + regnum_ring4 + regnum_ring5\
                  +regnum_ring6 + regnum_ring7 + regnum_ring8 + regnum_ring9 + regnum_ring10\
                  +regnum_ring11+ 1 # The final 1 accounts for the central region
    
    if print_details == True:
        # Display the result on the screen
        print( 'Total Number of Offsets: ' + str(regnum_total) )
        ###############################################################################
        # Print the ring centers to the screen
        print('Ring 0 Center: ' + str(0))
        print('Ring 1 Center: ' + str(r_ring1_centers))
        print('Ring 2 Center: ' + str(r_ring2_centers))
        print('Ring 3 Center: ' + str(r_ring3_centers))
        print('Ring 4 Center: ' + str(r_ring4_centers))
        print('Ring 5 Center: ' + str(r_ring5_centers))
        print('Ring 6 Center: ' + str(r_ring6_centers))
        print('Ring 7 Center: ' + str(r_ring7_centers))
        print('Ring 8 Center: ' + str(r_ring8_centers))
        print('Ring 9 Center: ' + str(r_ring9_centers))
        print('Ring 10 Center: ' + str(r_ring10_centers))
        print('Ring 11 Center: ' + str(r_ring11_centers))
    ###############################################################################
    # Calculate the area associated with each offset (the region area)
    reg_area_ring0 = np.pi*(r_ring0**2)/4
    reg_area_ring1 = jitter.Region_Area(r_ring0,r_ring1_outer,360/regnum_ring1)
    reg_area_ring2 = jitter.Region_Area(r_ring1_outer,r_ring2_outer,360/regnum_ring2)
    reg_area_ring3 = jitter.Region_Area(r_ring2_outer,r_ring3_outer,360/regnum_ring3)
    reg_area_ring4 = jitter.Region_Area(r_ring3_outer,r_ring4_outer,360/regnum_ring4)
    reg_area_ring5 = jitter.Region_Area(r_ring4_outer,r_ring5_outer,360/regnum_ring5)
    reg_area_ring6 = jitter.Region_Area(r_ring5_outer,r_ring6_outer,360/regnum_ring6)
    reg_area_ring7 = jitter.Region_Area(r_ring6_outer,r_ring7_outer,360/regnum_ring7)
    reg_area_ring8 = jitter.Region_Area(r_ring7_outer,r_ring8_outer,360/regnum_ring8)
    reg_area_ring9 = jitter.Region_Area(r_ring8_outer,r_ring9_outer,360/regnum_ring9)
    reg_area_ring10 = jitter.Region_Area(r_ring9_outer,r_ring10_outer,360/regnum_ring10)
    reg_area_ring11 = jitter.Region_Area(r_ring10_outer,r_ring11_outer,360/regnum_ring11)
    
    if print_details == True:
        # Display the areas to the screen
        print('Region Areas in Ring 0: ' + str(reg_area_ring0))
        print('Region Areas in Ring 1: ' + str(reg_area_ring1))
        print('Region Areas in Ring 2: ' + str(reg_area_ring2))
        print('Region Areas in Ring 3: ' + str(reg_area_ring3))
        print('Region Areas in Ring 4: ' + str(reg_area_ring4))
        print('Region Areas in Ring 5: ' + str(reg_area_ring5))
        print('Region Areas in Ring 6: ' + str(reg_area_ring6))
        print('Region Areas in Ring 7: ' + str(reg_area_ring7))
        print('Region Areas in Ring 8: ' + str(reg_area_ring8))
        print('Region Areas in Ring 9: ' + str(reg_area_ring9))
        print('Region Areas in Ring 10: ' + str(reg_area_ring10))
        print('Region Areas in Ring 11: ' + str(reg_area_ring11))
    
    # Calculate the total area for normalization
    # The total area of the circle is:
    total_area = np.pi*(r_ring11_outer**2)
    
    # Verify that the areas are correct.
    # The total area of all the rings is:
    total_area_of_rings = reg_area_ring0 + reg_area_ring1*regnum_ring1\
                         + reg_area_ring2*regnum_ring2 + reg_area_ring3*regnum_ring3\
                         + reg_area_ring4*regnum_ring4 + reg_area_ring5*regnum_ring5\
                         + reg_area_ring6*regnum_ring6 + reg_area_ring7*regnum_ring7\
                         + reg_area_ring8*regnum_ring8 + reg_area_ring9*regnum_ring9\
                         + reg_area_ring10*regnum_ring10 + reg_area_ring11*regnum_ring11
    
    if print_details == True:
        # Print the comparison to the screen:
        print('The total area is ' +str(total_area) +'.')
        print('The sum of the ring areas is ' +str(total_area_of_rings) +'.')
    
    # The two totals should be nearly identical, allowing for some rounding error
    assert np.allclose(total_area,total_area_of_rings,atol=0.1) == True
    
    # Calculate the normalized areas
    reg_area_ring0_norm = reg_area_ring0 / total_area
    reg_area_ring1_norm = reg_area_ring1 / total_area
    reg_area_ring2_norm = reg_area_ring2 / total_area
    reg_area_ring3_norm = reg_area_ring3 / total_area
    reg_area_ring4_norm = reg_area_ring4 / total_area
    reg_area_ring5_norm = reg_area_ring5 / total_area
    reg_area_ring6_norm = reg_area_ring6 / total_area
    reg_area_ring7_norm = reg_area_ring7 / total_area
    reg_area_ring8_norm = reg_area_ring8 / total_area
    reg_area_ring9_norm = reg_area_ring9 / total_area
    reg_area_ring10_norm = reg_area_ring10 / total_area
    reg_area_ring11_norm = reg_area_ring11 / total_area
    
    if print_details == True:
        # Display the areas to the screen
        print('Normalized Region Areas in Ring 0: ' + str(reg_area_ring0_norm))
        print('Normalized Region Areas in Ring 1: ' + str(reg_area_ring1_norm))
        print('Normalized Region Areas in Ring 2: ' + str(reg_area_ring2_norm))
        print('Normalized Region Areas in Ring 3: ' + str(reg_area_ring3_norm))
        print('Normalized Region Areas in Ring 4: ' + str(reg_area_ring4_norm))
        print('Normalized Region Areas in Ring 5: ' + str(reg_area_ring5_norm))
        print('Normalized Region Areas in Ring 6: ' + str(reg_area_ring6_norm))
        print('Normalized Region Areas in Ring 7: ' + str(reg_area_ring7_norm))
        print('Normalized Region Areas in Ring 8: ' + str(reg_area_ring8_norm))
        print('Normalized Region Areas in Ring 9: ' + str(reg_area_ring9_norm))
        print('Normalized Region Areas in Ring 10: ' + str(reg_area_ring10_norm))
        print('Normalized Region Areas in Ring 11: ' + str(reg_area_ring11_norm))
    
    # Verify that the areas are correct.
    # The normalized total area of all the rings is:
    total_area_of_rings_norm = reg_area_ring0_norm + reg_area_ring1_norm*regnum_ring1\
                         + reg_area_ring2_norm*regnum_ring2 + reg_area_ring3_norm*regnum_ring3\
                         + reg_area_ring4_norm*regnum_ring4 + reg_area_ring5_norm*regnum_ring5\
                         + reg_area_ring6_norm*regnum_ring6 + reg_area_ring7_norm*regnum_ring7\
                         + reg_area_ring8_norm*regnum_ring8 + reg_area_ring9_norm*regnum_ring9\
                         + reg_area_ring10_norm*regnum_ring10 + reg_area_ring11_norm*regnum_ring11
                             
    if print_details == True:
        # Print the comparison to the screen:
        print('The sum of the normalized ring areas is ' +str(total_area_of_rings_norm) +'.')
        
    # The normalized area should be nearly equal to 1, allowing for some rounding error
    assert np.allclose(total_area_of_rings_norm,1.0,rtol=0.001)
    ###############################################################################
    # Save the offsets and the normalized areas to a text file.
    jitter_data_file = 'Jitter_Data_File.txt'
    # Header Data
    fields = ['x_off','y_off','Anorm']
    # Begin writing the file with the header data
    # Write the file
    with open(jitter_data_file,'w') as csvfile:
        # create the csv writer object
        csvwriter=csv.writer(csvfile)
        # write the fields
        csvwriter.writerow(fields)
    
    # Now, add the data for each row of segments
    
    # Ring 0
    ring0_data =['0.','0.',np.format_float_positional(reg_area_ring0_norm)]
    with open(jitter_data_file,'a') as csvfile:
        # create the csv writer object
        csvwriter=csv.writer(csvfile)
        # write the fields
        csvwriter.writerow(ring0_data)
    
    # Ring 1
    # This ring has regnum_ring1 regions with centers specified by x_ring1_centers and
    # y_ring1_centers. Each of these regions has normalized area reg_area_ring1_norm.
    for i in range(regnum_ring1):
        ring1_data = [np.format_float_positional(x_ring1_centers[i]),\
                      np.format_float_positional(y_ring1_centers[i]),\
                      np.format_float_positional(reg_area_ring1_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring1_data)
    
    # Ring 2
    # This ring has regnum_ring2 regions with centers specified by x_ring2_centers and
    # y_ring2_centers. Each of these regions has normalized area reg_area_ring2_norm.
    for i in range(regnum_ring2):
        ring2_data = [np.format_float_positional(x_ring2_centers[i]),\
                      np.format_float_positional(y_ring2_centers[i]),\
                      np.format_float_positional(reg_area_ring2_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring2_data)
            
    # Ring 3
    # This ring has regnum_ring3 regions with centers specified by x_ring3_centers and
    # y_ring3_centers. Each of these regions has normalized area reg_area_ring3_norm.
    for i in range(regnum_ring3):
        ring3_data = [np.format_float_positional(x_ring3_centers[i]),\
                      np.format_float_positional(y_ring3_centers[i]),\
                      np.format_float_positional(reg_area_ring3_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring3_data)
            
    # Ring 4
    # This ring has segnum_ring4 regions with centers specified by x_ring4_centers and
    # y_ring4_centers. Each of these regions has normalized area reg_area_ring4_norm.
    for i in range(regnum_ring4):
        ring4_data = [np.format_float_positional(x_ring4_centers[i]),\
                      np.format_float_positional(y_ring4_centers[i]),\
                      np.format_float_positional(reg_area_ring4_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring4_data)
            
    # Ring 5
    # This ring has regnum_ring5 regions with centers specified by x_ring5_centers and
    # y_ring5_centers. Each of these regions has normalized area reg_area_ring5_norm.
    for i in range(regnum_ring5):
        ring5_data = [np.format_float_positional(x_ring5_centers[i]),\
                      np.format_float_positional(y_ring5_centers[i]),\
                      np.format_float_positional(reg_area_ring5_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring5_data)
            
    # Ring 6
    # This ring has regnum_ring6 regions with centers specified by x_ring6_centers and
    # y_ring6_centers. Each of these regions has normalized area reg_area_ring6_norm.
    for i in range(regnum_ring6):
        ring6_data = [np.format_float_positional(x_ring6_centers[i]),\
                      np.format_float_positional(y_ring6_centers[i]),\
                      np.format_float_positional(reg_area_ring6_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring6_data) 
    
    #Ring 7
    # This ring has regnum_ring7 regions with centers specified by x_ring7_centers and
    # y_ring7_centers. Each of these regions has normalized area reg_area_ring7_norm.
    for i in range(regnum_ring7):
        ring7_data = [np.format_float_positional(x_ring7_centers[i]),\
                      np.format_float_positional(y_ring7_centers[i]),\
                      np.format_float_positional(reg_area_ring7_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring7_data)
            
    # Ring 8
    # This ring has regnum_ring8 regions with centers specified by x_ring8_centers and
    # y_ring8_centers. Each of these regions has normalized area reg_area_ring8_norm.
    for i in range(regnum_ring8):
        ring8_data = [np.format_float_positional(x_ring8_centers[i]),\
                      np.format_float_positional(y_ring8_centers[i]),\
                      np.format_float_positional(reg_area_ring8_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring8_data)
            
    # Ring 9
    # This ring has regnum_ring9 regions with centers specified by x_ring9_centers and
    # y_ring9_centers. Each of these regions has normalized area reg_area_ring9_norm.
    for i in range(regnum_ring9):
        ring9_data = [np.format_float_positional(x_ring9_centers[i]),\
                      np.format_float_positional(y_ring9_centers[i]),\
                      np.format_float_positional(reg_area_ring9_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring9_data)
            
    # Ring 10
    # This ring has regnum_ring1 regions with centers specified by x_ring10_centers and
    # y_ring10_centers. Each of these regions has normalized area reg_area_ring10_norm.
    for i in range(regnum_ring10):
        ring10_data = [np.format_float_positional(x_ring10_centers[i]),\
                      np.format_float_positional(y_ring10_centers[i]),\
                      np.format_float_positional(reg_area_ring10_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring10_data)
            
    # Ring 11
    # This ring has regnum_ring11 regions with centers specified by x_ring11_centers and
    # y_ring11_centers. Each of these regions has normalized area reg_area_ring11_norm.
    for i in range(regnum_ring11):
        ring11_data = [np.format_float_positional(x_ring11_centers[i]),\
                      np.format_float_positional(y_ring11_centers[i]),\
                      np.format_float_positional(reg_area_ring11_norm)]
        with open(jitter_data_file,'a') as csvfile:
            # create the csv writer object
            csvwriter=csv.writer(csvfile)
            # write the fields
            csvwriter.writerow(ring11_data)
        
    ###############################################################################
    # Test that the manually specified regions match those calculated by
    # jitter.Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, \
    #                                    N_offsets_per_ring,starting_offset_ang_by_ring,\
    #                                    r_ring0=0.075,dr_rings=None)
    
    # Define the function inputs to match the example above
    outer_radius_of_offset_circle = r_ring11_outer
    N_rings_of_offsets = 11
    N_offsets_per_ring = np.array([regnum_ring1,regnum_ring2,regnum_ring3,regnum_ring4,regnum_ring5,regnum_ring6,regnum_ring7,regnum_ring8,regnum_ring9,regnum_ring10,regnum_ring11])
    starting_offset_ang_by_ring = np.array([theta_ring1_centers_start,theta_ring2_centers_start,theta_ring3_centers_start,theta_ring4_centers_start,theta_ring5_centers_start,theta_ring6_centers_start,theta_ring7_centers_start,theta_ring8_centers_start,theta_ring9_centers_start,theta_ring10_centers_start,theta_ring11_centers_start])
    r_ring0=0.075
    dr_rings=np.array([dr_ring1,dr_ring2,dr_ring3,dr_ring4,dr_ring5,dr_ring6,dr_ring7,dr_ring8,dr_ring9,dr_ring10,dr_ring11])
    
    # Use the function to calculate the parameters for all of the regions in all of the rings
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict,yl_outer_dict, boundary_coords_dict = \
         jitter.Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, N_offsets_per_ring, starting_offset_ang_by_ring,r_ring0,dr_rings)
         
    # Check that the offsets and areas match those calculated above
    if print_details == True:
        print('Checking that normalized areas match')
        print('Ring 0 Difference:' +str(reg_area_ring0_norm - A_offsets[0]))
        print('Ring 1 Difference:' +str(reg_area_ring1_norm - A_offsets[1]))
        print('Ring 2 Difference:' +str(reg_area_ring2_norm - A_offsets[2]))
        print('Ring 3 Difference:' +str(reg_area_ring3_norm - A_offsets[3]))
        print('Ring 4 Difference:' +str(reg_area_ring4_norm - A_offsets[4]))
        print('Ring 5 Difference:' +str(reg_area_ring5_norm - A_offsets[5]))
        print('Ring 6 Difference:' +str(reg_area_ring6_norm - A_offsets[6]))
        print('Ring 7 Difference:' +str(reg_area_ring7_norm - A_offsets[7]))
        print('Ring 8 Difference:' +str(reg_area_ring8_norm - A_offsets[8]))
        print('Ring 9 Difference:' +str(reg_area_ring9_norm - A_offsets[9]))
        print('Ring 10 Difference:' +str(reg_area_ring10_norm - A_offsets[10]))
        print('Ring 11 Difference:' +str(reg_area_ring11_norm - A_offsets[11]))
        
        print('Checking the offset coordinates')
        print('Max Ring 0 Difference:' +str(np.max(np.abs(x_offsets[0]-0))) +','\
                                       +str(np.max(np.abs(y_offsets[0]-0))))
        print('Max Ring 1 Difference:' +str(np.max(np.abs(x_offsets[1]-x_ring1_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[1]-y_ring1_centers))))
        print('Max Ring 2 Difference:' +str(np.max(np.abs(x_offsets[2]-x_ring2_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[2]-y_ring2_centers))))
        print('Max Ring 3 Difference:' +str(np.max(np.abs(x_offsets[3]-x_ring3_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[3]-y_ring3_centers))))
        print('Max Ring 4 Difference:' +str(np.max(np.abs(x_offsets[4]-x_ring4_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[4]-y_ring4_centers))))
        print('Max Ring 5 Difference:' +str(np.max(np.abs(x_offsets[5]-x_ring5_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[5]-y_ring5_centers))))
        print('Max Ring 6 Difference:' +str(np.max(np.abs(x_offsets[6]-x_ring6_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[6]-y_ring6_centers))))
        print('Max Ring 7 Difference:' +str(np.max(np.abs(x_offsets[7]-x_ring7_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[7]-y_ring7_centers))))
        print('Max Ring 8 Difference:' +str(np.max(np.abs(x_offsets[8]-x_ring8_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[8]-y_ring8_centers))))
        print('Max Ring 9 Difference:' +str(np.max(np.abs(x_offsets[9]-x_ring9_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[9]-y_ring9_centers))))
        print('Max Ring 10 Difference:' +str(np.max(np.abs(x_offsets[10]-x_ring10_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[10]-y_ring10_centers))))
        print('Max Ring 11 Difference:' +str(np.max(np.abs(x_offsets[11]-x_ring11_centers))) +','\
                                       +str(np.max(np.abs(y_offsets[11]-y_ring11_centers))))
    
    # The normalized ring areas should match for the two calculation methods.
    for iring in range(12):
        s1temp = 'reg_area_ring' + str(iring) +'_norm - A_offsets[' +str(iring) +']'
        assert np.allclose(eval(s1temp),0)
    
    # The offset coordinates should match for the two methods.
    assert np.allclose(np.max(np.abs(x_offsets[0])),0)
    assert np.allclose(np.max(np.abs(y_offsets[0])),0)
    for iring in range(1,12):
        sxtemp = 'np.max(np.abs(x_offsets[' + str(iring) + '] - x_ring' +str(iring) + '_centers))'
        sytemp = 'np.max(np.abs(y_offsets[' + str(iring) + '] - y_ring' +str(iring) + '_centers))'
        assert np.allclose(eval(sxtemp),0)
        assert np.allclose(eval(sytemp),0)
    ###############################################################################
    if show_plots == True:
        # Test the automated plotting script
        jitter.Plot_ALL_Offsets_And_Region_Outlines(x_offsets, y_offsets, x_outer_dict, yu_outer_dict, yl_outer_dict, boundary_coords_dict, N_rings_of_offsets, N_offsets_per_ring)
###############################################################################
def check_offset_weights():
    
    '''
    This function compares the calculated weights for the example offset list
    against a pre-calculated list for the same offsets.
    '''
    
    # jitter and finite stellar diameter keywords
    stellar_diam_and_jitter_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(starID=None)
    
    # build the list of offsets
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, yl_outer_dict, boundary_coords_dict \
        = jitter.Determine_offsets_and_areas(stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'], \
                                      stellar_diam_and_jitter_keywords['N_rings_of_offsets'], \
                                      stellar_diam_and_jitter_keywords['N_offsets_per_ring'], \
                                      stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'], \
                                      stellar_diam_and_jitter_keywords['r_ring0'], \
                                      stellar_diam_and_jitter_keywords['dr_rings'])
    print('Offsets calculated!')
    
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
    
    # look at the weight calculation
    if 'r_stellar_disc_mas' in stellar_diam_and_jitter_keywords:
        if stellar_diam_and_jitter_keywords['stellar_diam_mas'] != 2*stellar_diam_and_jitter_keywords['r_stellar_disc_mas']:
            raise KeyError("ERROR: The provided stellar radius and stellar diameter are inconsistent.")
        r_stellar_disc_mas = stellar_diam_and_jitter_keywords['r_stellar_disc_mas']
    else:
        # Calculate the radius
        r_stellar_disc_mas = 0.5*stellar_diam_and_jitter_keywords['stellar_diam_mas']
        # Add to the dictionary
        stellar_diam_and_jitter_keywords['r_stellar_disc_mas'] = r_stellar_disc_mas
        
    # If an outer radius of the offset circle has not been specified, set
    # it to the radius of the stellar disc.
    if 'outer_radius_of_offset_circle' not in stellar_diam_and_jitter_keywords.keys():
        stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'] = r_stellar_disc_mas
    
    # Set up a uniform grid of offsets (X,Y).
    
    outer_radius_of_offset_circle = stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle']
        
    # If the resolution of the offset grid has not been specified, default to 260.
    if 'N_offsetgrid' not in stellar_diam_and_jitter_keywords.keys():
        N_offsetgrid = 260
        stellar_diam_and_jitter_keywords['N_offsetgrid'] = N_offsetgrid
    else:
        N_offsetgrid = stellar_diam_and_jitter_keywords['N_offsetgrid']
            
    x = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid*10)
    y = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid*10)
    X,Y = np.meshgrid(x,y)

    # Define the top-hat function
    disc_indices = (X**2 + Y**2 <= r_stellar_disc_mas**2)
    
    # Work on the interpolation
    x_predetermined = x_offsets_list
    y_predetermined = y_offsets_list
    
    #interp = RegularGridInterpolator([x,y], disc_indices)    
    #W = interp(np.array([x_offsets_list,y_offsets_list]).T,'quintic')
    
    #ax = plt.axes(projection='3d')
    #ax.plot3D(x_offsets_list,y_offsets_list,W,'ro')
    #plt.show()
    
    #Xr = X.ravel()
    #Yr = Y.ravel()
    #Zr = disc_indices.ravel()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #for iplt in range(len(Xr)):
    #    ax.scatter(Xr[iplt],Yr[iplt],Zr[iplt],marker='.',color='b')
        
    # TODO: Fill in this test function once jitter is implemented too.
###############################################################################
def test_obs_with_finite_stellar_diam():
    '''
     This function tests running an observation sequence with finite stellar diameter included.
    '''
    
    # Set up keywords
    # optics keywords
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
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}

    # jitter and finite stellar diameter keywords
    stellar_diam_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(starID=None)
    
    # Define the scene
    base_scene = scene.Scene(host_star_properties)
    
    # Set up the optics
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, stellar_diam_and_jitter_keywords=stellar_diam_keywords, if_quiet=True)
    
    # Set up the detectior
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    
    # Test a single frame 
    n_frames = 1
    simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    
    return simulatedImage_list
   
    # Try getting out just the onaxis electric field
    #assert isinstance(simulatedImage_list, list)
    #assert len(simulatedImage_list) == n_frames
    #assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    #assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

    # Test a single full frame 
    #simulatedImage_list_fullframe = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames,full_frame=True, loc_x=300, loc_y=300)

    #assert isinstance(simulatedImage_list_fullframe, list)
    #assert len(simulatedImage_list_fullframe) == n_frames
    #assert isinstance(simulatedImage_list_fullframe[n_frames-1], SimulatedImage)
    #assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector, fits.hdu.hdulist.HDUList)

    #assert len(simulatedImage_list_fullframe[n_frames-1].image_on_detector) == 2 # Primary and Image HDU
    #assert isinstance(simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].data, np.ndarray)

    #assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EXPTIME'] == exp_time
    #assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[1].header['EMGAIN_C'] == gain

    #assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['PHTCNT'] == detector.photon_counting
    #assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['OPGAIN'] == gain
    #assert simulatedImage_list_fullframe[n_frames-1].image_on_detector[0].header['FRAMET'] == exp_time
    
    # Test several frames

    #n_frames = 100
    #simulatedImage_list = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
    #assert isinstance(simulatedImage_list, list)
    #assert len(simulatedImage_list) == n_frames
    #assert isinstance(simulatedImage_list[n_frames-1], SimulatedImage)
    #assert isinstance(simulatedImage_list[n_frames-1].image_on_detector, fits.hdu.image.PrimaryHDU)

###############################################################################
def test_weight_calculation():
    
    # This function tests the lines that calculate the weights for each offset
    # region.
    
    # Set up keywords
    # optics keywords
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
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}

    # jitter and finite stellar diameter keywords
    stellar_diam_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(starID=None)
    
    # Define the scene
    base_scene = scene.Scene(host_star_properties)
    
    # Set up the optics
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, stellar_diam_and_jitter_keywords=stellar_diam_keywords, if_quiet=True)
    
    # Test
    sim_scene = optics.get_host_star_psf(base_scene)
    
###############################################################################
def basic_weight_calculation_test():
    '''
    Testing replacing interp2d as the interpolation method when calculating the weights
    '''

    # jitter and finite stellar diameter keywords
    stellar_diam_and_jitter_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(starID=None)
    
    # offsets and areas
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, yl_outer_dict, boundary_coords_dict \
        = jitter.Determine_offsets_and_areas(stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'], \
                                      stellar_diam_and_jitter_keywords['N_rings_of_offsets'], \
                                      stellar_diam_and_jitter_keywords['N_offsets_per_ring'], \
                                      stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'], \
                                      stellar_diam_and_jitter_keywords['r_ring0'], \
                                      stellar_diam_and_jitter_keywords['dr_rings'])
    print('Offsets calculated!')
    
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
    
    # Determine the total number of offsets
    N_offsets = np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'])+1
    stellar_diam_and_jitter_keywords['N_offsets_counting_origin'] = N_offsets
    
    # setup
    if 'r_stellar_disc_mas' in stellar_diam_and_jitter_keywords:
        if stellar_diam_and_jitter_keywords['stellar_diam_mas'] != 2*stellar_diam_and_jitter_keywords['r_stellar_disc_mas']:
            raise KeyError("ERROR: The provided stellar radius and stellar diameter are inconsistent.")
        r_stellar_disc_mas = stellar_diam_and_jitter_keywords['r_stellar_disc_mas']
    else:
        # Calculate the radius
        r_stellar_disc_mas = 0.5*stellar_diam_and_jitter_keywords['stellar_diam_mas']
        # Add to the dictionary
        stellar_diam_and_jitter_keywords['r_stellar_disc_mas'] = r_stellar_disc_mas
        
    # If an outer radius of the offset circle has not been specified, set
    # it to the radius of the stellar disc.
    if 'outer_radius_of_offset_circle' not in stellar_diam_and_jitter_keywords.keys():
        stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'] = r_stellar_disc_mas
        
    outer_radius_of_offset_circle = stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle']
         
    # If the resolution of the offset grid has not been specified, default to 260.
    if 'N_offsetgrid' not in stellar_diam_and_jitter_keywords.keys():
        N_offsetgrid = 260
        stellar_diam_and_jitter_keywords['N_offsetgrid'] = N_offsetgrid
    else:
        N_offsetgrid = stellar_diam_and_jitter_keywords['N_offsetgrid']
             
    x = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid)
    y = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid)
    X,Y = np.meshgrid(x,y)
     
    # Define the top-hat function for the stellar disc if appropriate
    disc_indices = (X**2 + Y**2 <= r_stellar_disc_mas**2)
    
    # Specify the weights on the uniform grid
    Wjitdisc = disc_indices
    
    # Interpolate Wjitdisc to find the values at the predetermined offsetes
    f_interp = RectBivariateSpline(x,y,Wjitdisc.T)
    x_predetermined = stellar_diam_and_jitter_keywords['offset_field_data']['x_offsets_mas']
    y_predetermined = stellar_diam_and_jitter_keywords['offset_field_data']['y_offsets_mas']
    W = np.zeros(stellar_diam_and_jitter_keywords['N_offsets_counting_origin'],)
    for i_offset in range(stellar_diam_and_jitter_keywords['N_offsets_counting_origin']):
        W[i_offset] = f_interp(x_predetermined[i_offset],y_predetermined[i_offset])
    
    # Test plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for iplt in range(stellar_diam_and_jitter_keywords['N_offsets_counting_origin']):
        ax.scatter(x_predetermined[iplt],y_predetermined[iplt],W[iplt],marker='.',color='b')
    plt.show()
###############################################################################
def test_all_pol_obs_with_finite_stellar_diam():
    '''
    Test that the calculations run for no polarization, optics.prism = 'POL0',
    optics.prism = 'POL45', and polaxis = -10
    '''
    # TODO: Add pol0 and pol45 options
    
    # Set up keywords
    # optics keywords
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
    
    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':-10, 'output_dim':201,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    
    # emccd keywords
    gain =1000
    emccd_keywords ={'em_gain':gain}
    
    # jitter and finite stellar diameter keywords
    stellar_diam_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(starID=None)
    
    # Define the scene
    base_scene = scene.Scene(host_star_properties)
    
    # Set up the optics
    optics =  instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, stellar_diam_and_jitter_keywords=stellar_diam_keywords, if_quiet=True)
    
    # Set up the detectior
    detector = instrument.CorgiDetector( emccd_keywords)
    
    # Define the exposure time
    exp_time = 2000
    
    # Start with polaxis = 10
    # Test a single frame 
    n_frames = 1
    simulatedImage_list_polaxism10 = observation.generate_observation_sequence(base_scene, optics, detector, exp_time, n_frames)
      
    
###############################################################################
if __name__ == '__main__':
    #test_offsets_and_areas_against_example()
    test_obs_with_finite_stellar_diam()
    #test_weight_calculation()
    #basic_weight_calculation_test()