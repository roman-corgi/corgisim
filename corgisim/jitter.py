# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
import timeit
import proper
import roman_preflight_proper

"""
This script contains a set of functions for calculating jitter or jitter-like
effects (such as the stellar diameter).

One method for estimating the jitter is to represent the jitter distribution
by a set of source offsets, approximate and weight the intensity for each
offset, then combine the weighted intensities to form the final intensity.

The set of source offsets and the normalized area of the region represented by 
each offset can be determined in advance.

One set of functions calculate geometric quantities:
    
    Find_ycircle(r,x,xc,yc,half): calculates the y coordinate that corresponds 
                                  to the specified x coordinate, with half 
                                  determining whether the "upper" or "lower" 
                                  half of the circle is being considered.
                                  
    Polar_in_rad_to_Cartesian_coords(r,theta): converts from polar to Cartesian
                                               coordinates
        
One set of functions determines the offsets and the areas of their 
corresponding regions:
    
    region_centers(theta_start_deg,theta_num): determines the center of each
                                               region along a ring
                                               
    radial_boundaries(theta_start,theta_num): defines the radial boundaries of
                                              the regions along a ring
                                                                                    
    Determine_ring_params(r_ring_inner,dr_ring,regnum_ring,
                          theta_ring_centers_start): 
                           returns all of the desired parameters for a ring
                           of regions. These parameters are the locations of
                           the region centers (the offsets) and the
                           coordinates needed to plot the region outlines
                           (outer ring plus radial boundaries)
                           
    
    Region_Area(r_inner,r_outer,theta): calculates the area of the region
    
    Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, \
                                N_offsets_per_ring,starting_offset_ang_by_ring,' \
                                r_ring0,dr_rings):
                                determines the set of offsets and the area of
                                the region associated with each offset.
                                The area of each region is normalized to the
                                area of the full circle.

One set of functions plots the offsets and the outlines of their corresponding
regions:
    
    Plot_Offsets_And_Region_Outlines(x_ring_centers,y_ring_centers,
                                      x_ring_outer,yu_ring_outer,yl_ring_outer,
                                      boundary_coords,regnum_ring):
                                      Plots a ring of offsets and their
                                      region outlines
                                      
    Plot_ALL_Offsets_And_Region_Outlines(x_offsets,y_offsets,x_outer_dict,\
                                             yu_outer_dict,yl_outer_dict,\
                                             boundary_coords_dict,N_rings_of_offsets,\
                                             N_offsets_per_ring)
                                         Automates plotting all of the offsets
                                         and their associated regions by 
                                         iteratively calling
                                         Plot_Offsets_And_Region_Outlines

One set of functions save the offsets and the areas of their corresponding
regions to a csv file:
    
    save_offsets_and_areas(x_offsets,y_offsets,A_offsets,N_rings_of_offsets):
                     Saves the coordinates of the offset sources and the areas
                     of the regions they represent.
        
There is also a function to define a set of predefined parameters for the 
jitter and finite stellar diameter calculations:
    
    load_predefined_jitter_and_stellar_diam_params(starID=None)
    
One set of functions constructs the library of delta electric fields and weights:
    
    build_delta_e_field_library(stellar_diam_and_jitter_keywords,optics):
        Builds a list of offset sources, the area of the region 
        represented by each offset, and the weight for each offset;
        calculates the electric field for each offset; and saves a library of 
        delta electric fields for use in calculating the effects of jitter or 
        finite stellar diameter.
        
    calculate_weights_for_jitter_and_finite_stellar_diameter(stellar_diam_and_jitter_keywords):
        Calculates the set of weights to use when combining the
        intensities of the offset sources to add the effects of jitter and the
        finite stellar diameter
    
"""
###############################################################################
# Functions that calculate geometric quantities

def Find_ycircle(r,x,xc,yc,half):
    '''
     This function calculates the y coordinate that corresponds to the
     specified x coordinate, with half determining whether the "upper" or
     "lower" half of the circle is being considered.
     This function is capable of handling vector inputs, and it returns 0
     in situations where x does not correspond to a point on the circle.
    
     Inputs:
            r: radius of the circle
            x: x coordinate(s) of the desired point(s)
            xc: x coordinate of the circle center
            yc: y coordinate of the circle center
            half: string specifying which half of the circle to use
                  Can be 'u', 'U', 't', or 'T' for the upper/top half or
                         'l', 'L', 'b,' or 'B' for the lower/bottom half.
     Output:
            y: the y coordinate for each point
    '''
    
    # For the upper/top half of a circle
    if half.startswith('u') or half.startswith('U') or half.startswith('t') \
        or half.startswith('T'):
            y = np.sqrt(r**2 - (x-xc)**2) + yc
                # For the lower/bottom half of a circle
    elif half.startswith('l') or half.startswith('L') or half.startswith('b') \
        or half.startswith('B'):
            y = -np.sqrt(r**2 - (x-xc)**2) +yc
    return y
        
def Polar_in_rad_to_Cartesian_coords(r,theta):
    '''
     This function converts from polar to Cartesian coordinates.
     Inputs:
            r: radius
            theta: angle
     Outputs:
            x: Cartesian coordinate along the horizontal axis
            y: Cartesian coordinate along the vertical axis
    '''
    
    # Define the Cartesian coordinates
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    # Return the answer
    return x,y

###############################################################################
# Functions for determining the offsets and their associated regions

def region_centers(theta_start_deg,theta_num):
    '''
     This function determines the centers of the regions in an annular ring.
     Inputs: 
             theta_start: starting angle in degrees
             theta_num:   number of regions in the ring
     Outputs:
             theta:       an array of angles locating the centers
    '''
    
    # Determine the spacing in theta in degrees
    delta_theta = 360/theta_num
    
    # Construct an array of angles
    theta = theta_start_deg + np.arange(theta_num)*delta_theta
    return theta

def radial_boundaries(theta_start,theta_num):
    '''
     This function determines the radial boundaries of the regions along a 
     ring.
     Inputs:
            theta_start: starting angle in degrees
            theta_num:   number of regions in the ring
     Outputs:
            theta_boundaries: the angles defining the region boundaries
    '''
    
    # Determine the spacing in theta in degrees
    delta_theta = 360/theta_num
    # The first region center has theta = theta_start.
    # Determine the angles for the region boundaries.
    theta_boundaries = theta_start + delta_theta/2 + delta_theta*np.arange(theta_num)
    return theta_boundaries

def Determine_ring_params(r_ring_inner,dr_ring,regnum_ring,theta_ring_centers_start):
    '''
     This function returns all of the desired parameters for a ring of regions.
     These are: the locations of the region centers (the jitter offsets)
                the coordinates needed to plot the region outlines (outer ring
                plus radial boundaries)
     Inputs:
            r_ring_inner: radius of the inner ring boundary
            dr_ring:      with of the ring
            regnum_ring:  number of regions within the ring
            theta_ring_centers_start: angle of the center of the first region
                                      IN DEGREES
     Outputs:
            x_ring_centers: x coordinates of the region centers
            y_ring_centers: y coordinates of the region centers
            r_ring_centers: radius of the region centers
            r_ring_outer:   radius of the outer ring defining the regions
            x_ring_outer:   x coordinates for points along the outer circle
                            defining the regions
            yu_ring_outer:  y coordinates for points along the upper half of
                            the circle defining the outer boundary of the 
                            regions
            yl_ring_outer:  y coordinates for points along the lower half of
                            the circle defining the outer boundar of the 
                            regions
            boundary_coords: coordinates defining the boundaries between
                             adjacent regions
    '''
    
    # Determining the locations of the region centers in polar and Cartesian
    # coordinates
    
    # Step 1: Determine the radius for the region centers
    r_ring_centers = r_ring_inner + dr_ring/2
    
    # Step 2: Determine the angle for each of the region centers
    theta_ring_centers = region_centers(theta_ring_centers_start, regnum_ring)*np.pi/180
    
    # Step 3: Convert to Cartesian coordinates
    x_ring_centers,y_ring_centers = Polar_in_rad_to_Cartesian_coords(r_ring_centers, theta_ring_centers)
    
    # Determining the coordinates for the outer circle defining the regions
    
    # Step 1: Determine the outer radius of the regions
    r_ring_outer = r_ring_centers + dr_ring/2
    
    # Step 2: Define the sets of coordinates that describe the outer circle
    x_ring_outer = np.linspace(-r_ring_outer, r_ring_outer)
    yu_ring_outer = Find_ycircle(r_ring_outer,x_ring_outer,0,0,'u')
    yl_ring_outer = Find_ycircle(r_ring_outer,x_ring_outer,0,0,'l')
    
    # Determining the coordinates for the radial region boundaries
    
    # Step 1: Determine the angles for the radial region boundaries
    theta_ring_radial_boundaries = radial_boundaries(theta_ring_centers_start,regnum_ring)*np.pi/180
    
    # Step 2: Specify the pair of coordinates for each boundary
    boundary_coords = np.zeros((regnum_ring,2,2),dtype=float)
    for ii in range(regnum_ring): # Iterate over each region
        # Determine the Cartesian coordinates for the inner and outer points
        # of the boundary
        boundary_xi,boundary_yi = Polar_in_rad_to_Cartesian_coords(r_ring_inner, theta_ring_radial_boundaries[ii])
        boundary_xo,boundary_yo = Polar_in_rad_to_Cartesian_coords(r_ring_outer, theta_ring_radial_boundaries[ii])
        # Store these points in the array for plotting purposes
        boundary_coords[ii,0,0] = boundary_xi
        boundary_coords[ii,0,1] = boundary_yi
        boundary_coords[ii,1,0] = boundary_xo
        boundary_coords[ii,1,1] = boundary_yo
        
    # Return the answers
    return x_ring_centers, y_ring_centers, r_ring_centers, r_ring_outer, \
           x_ring_outer, yu_ring_outer, yl_ring_outer, boundary_coords
                   
def Region_Area(r_inner,r_outer,theta):
    '''
     This function calculates the area of a region specified by its inner
     and outer radii and its wedge angle (in DEGREES).
    '''
    area = np.pi*(r_outer**2 - r_inner**2)*theta/360
    return area

def Determine_offsets_and_areas(outer_radius_of_offset_circle, N_rings_of_offsets, \
                                N_offsets_per_ring,starting_offset_ang_by_ring,\
                                r_ring0=0.075,dr_rings=0):
    '''
     This function determines the set of offsets and the area of the region 
     associated with each offset. The region area is normalized to the area
     of the full circle.
     Inputs:
            outer_radius_of_offset_circle: The outer radius of the circle
                                           containing all of the offsets and
                                           their corresponding regions. This
                                           radius is equal to the outer radii
                                           of the regions in the outermost
                                           ring.
            N_rings_of_offsets: The number of rings of offsets, not counting
                                the zeroth ring that consists of the offset
                                at (0,0) and its region.
            N_offsets_per_ring: The number of offsets in each ring.
                                To use different number of offsets in different
                                rings, define N_offsets_per_ring as an array
                                with one entry per ring.
            starting_offset_ang_by_ring: The angle (in DEGREES) of the first
                                         offset in the ring. To use a different
                                         angle for different rings, define
                                         starting_offset_ang_by_ring as an 
                                         array with one entry per ring.
            r_ring0: The radius of the region corresponding to the offset
                     at the center of the circle (0,0). Defaults to 0.075.
            dr_rings: The width for each ring of regions, not including the
                      central region (whose width is specified by r_ring0).
                      To assign specific widths, define dr_rings as an array
                      with one entry per ring.
                      If left unspecified, the same width will be applied to
                      each ring beyond the 0th.
                      Note that the sum of all the entries in dr_rings plus
                      r_ring0 must equal outer_radius_of_offset_circle.
     Outputs:
            x_offsets: A dictionary containing the x coordinates for each
                       offset in each ring
            y_offsets: A dictionary containing the y coordinates for each 
                       offset in each ring
            A_offsets: A dictionary containing the area of each offset in
                       each ring
            x_outer_dict: A dictionary containing a set of x coordinates for
                          the outer circle defining each ring (useful for plotting)
            yu_outer_dict: A dictionary containing a set of y coordinates for
                           the upper half of the outer circle defining each
                           ring (useful for plotting)
            yl_outer_dict: A dictionary containing a set of y coordinates for
                           the lower half of the outer circle defining each
                           ring (useful for plotting)
            boundary coords_dict: A dictionary containing the coordinates that
                                  specify the boundaries between regions in
                                  the same ring
    '''
    
    # If dr_rings is unspecified, define it as 
    # (outer_radius_of_offset_circle - r_ring0) / N_rings_of_offsets
    if dr_rings.all == 0:
        dr_rings = (outer_radius_of_offset_circle - r_ring0) / N_rings_of_offsets
        
    # If dr_rings is a scalar, construct an array of values for each ring
    if np.isscalar(dr_rings) == True:
        dr_rings = dr_rings * np.ones(N_rings_of_offsets)
        
    # Verify that the ring widths are specified correctly
    # All of the widths should add up to the total radius of the circle
    rounding_tol=3 # Number of decimal places to which we expect the radii to match
    if np.round(np.sum(dr_rings) + r_ring0,rounding_tol) != np.round(outer_radius_of_offset_circle,rounding_tol):
        raise ValueError('The sum of all the ring widths plus r_ring0 must equal outer_radius_of_offset_circle.')
    
    # If the number of offsets per ring is a scalar, construct an array
    if np.isscalar(N_offsets_per_ring) == True:
        N_offsets_per_ring = N_offsets_per_ring * np.ones(N_rings_of_offsets)
        
    # If starting angle is a scalar, construct an array
    if np.isscalar(starting_offset_ang_by_ring) == True:
        starting_offset_ang_by_ring = starting_offset_ang_by_ring * np.ones(N_rings_of_offsets)
    
    # Initialize the lists that will contain the x and y coordinates of the
    # offsets and the areas of their corresponding regions. Also initialize
    # dictionaries to store the coordinates outlining the regions, which can be
    # used for plotting purposes.
    
    x_offsets = {} # x coordinates of the offsets
    y_offsets = {} # y coordinates of the offsets
    A_offsets = {} # areas of the offsets
    x_outer_dict = {} # x coordinates for points along the outer circle that
                      # defines a ring of regions
    yu_outer_dict = {} # y coordinates for points along the upper half of the
                       # outer circle that defines a ring of regions
    yl_outer_dict = {} # y coordinates for points along the lower half of the
                       # outer circle that defines a ring of regions
    boundary_coords_dict = {} # coordinates that describe the boundaries
                              # between regions in the same ring
    r_outer_dict = {} # outer radii of the circles defining the rings of regions
    
    # Calculate the area of the entire circle (for normalization)
    total_area = np.pi*(outer_radius_of_offset_circle**2)
    
    # Iteratively fill in the values for each ring of offsets
    for iring in np.arange(N_rings_of_offsets+1):
        if iring==0: # The region containing the origin is different from the rest
            # Define the coordinates for the offset at the origin
            x_ring_centers = 0
            y_ring_centers = 0
            # Also specify that thte radius of the outer circle is r_ring0
            r_ring_outer = r_ring0
            # Define the set of x coordinates for the ring outline
            x_ring_outer = np.linspace(-r_ring0,r_ring0)
            # Define the sets of y coordinates for the ring outline
            yu_ring_outer = Find_ycircle(r_ring0,x_ring_outer,0,0,'u')
            yl_ring_outer = Find_ycircle(r_ring0,x_ring_outer,0,0,'l')
            # Specify that there are no boundaries because only one region
            boundary_coords = []
            # Calculate the area of the region around the origin
            reg_area_ringi = np.pi*(r_ring0**2)/4
            # Normalize the area
            reg_area_ringi = reg_area_ringi / total_area
        else: # use Determine_ring_params and Region_Area for the other rings 
              # of regions
            
            # Identify the appropriate parameters for the ring
            # Outer radius of the previous ring:
            r_outer_prev_ring = r_outer_dict[iring-1]
            # Width of current ring
            dr_iring = dr_rings[iring-1]
            # Number of regions in the current ring
            regnum_iring = N_offsets_per_ring[iring-1]
            # Angle of the first offset in the current ring
            theta_iring_centers_start = starting_offset_ang_by_ring[iring-1]
            # Determine the ring parameters
            x_ring_centers, y_ring_centers,r_ring_centers, r_ring_outer,\
            x_ring_outer, yu_ring_outer,yl_ring_outer, boundary_coords \
            = Determine_ring_params(r_outer_prev_ring,dr_iring,regnum_iring,theta_iring_centers_start)
            # Calculate the areas of the regions
            reg_area_ringi = Region_Area(r_outer_prev_ring,r_ring_outer,360/regnum_iring)
            # Normalize the area
            reg_area_ringi = reg_area_ringi / total_area
            
        # Store the results in the appropriate dictionaries
        x_offsets[iring] = x_ring_centers
        y_offsets[iring] = y_ring_centers
        A_offsets[iring] = reg_area_ringi
        x_outer_dict[iring] = x_ring_outer
        yu_outer_dict[iring] = yu_ring_outer
        yl_outer_dict[iring] = yl_ring_outer
        boundary_coords_dict[iring] = boundary_coords
        r_outer_dict[iring] = r_ring_outer
        
    # Return the results
    return x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, \
           yl_outer_dict, boundary_coords_dict

###############################################################################
# Functions for plotting the offsets and the outlines of their associated
# regions

def Plot_Offsets_And_Region_Outlines(x_ring_centers,y_ring_centers,\
                                      x_ring_outer,yu_ring_outer,yl_ring_outer,\
                                      boundary_coords,regnum_ring,fig,ax):
    '''
     This function plots a ring of offsets and their region outlines
     Inputs:
            x_ring_centers: x coordinates of the centers of the regions
            y_ring_centers: y coordinates of the centers of the regions
            x_ring_outer:   x coordinates for points along the outer circle
                            that defines the regions in the ring
            yu_ring_outer:  y coordinates for points along the upper half of
                            the circle that defines the regions in the ring
            yl_ring_outer:  y coordinates for points along the lower half of
                            the circle that defines the regions in the ring
            boundary_coords: coordinates that define the boundaries between
                             adjacent regions in the ring
            regnum_ring:    number of regions in the ring
            fig:            the figure that will contain the result
            ax:             the figure axes
     Output:
            a plot of the ring of offsets and outlines of their regions
     NOTE: This function does not initialize the plot window or show the plot.
           It is expected that the plot window will be set up before this
           function is called (because this will allow all of the rings to be
           drawn on the same plot).
    '''
    
    # Plot the region centers
    line_rc, = ax.plot(x_ring_centers,y_ring_centers,'k.')
    
    # Plot the outer circular boundary
    line_rou, = ax.plot(x_ring_outer,yu_ring_outer,'k')
    line_rol, = ax.plot(x_ring_outer,yl_ring_outer,'k')
    
    # Plot the radial region boundaries
    for ii in range(regnum_ring):
        line_bound, = ax.plot(boundary_coords[ii,:,0],boundary_coords[ii,:,1],'b')
        
def Plot_ALL_Offsets_And_Region_Outlines(x_offsets,y_offsets,x_outer_dict,\
                                         yu_outer_dict,yl_outer_dict,\
                                         boundary_coords_dict,N_rings_of_offsets,\
                                         N_offsets_per_ring):
    '''
     This function iteratively uses Plot_Offsets_And_Region_Outlines to plot
     all of the offsets and their corresponding regions.
     Inputs:
            x_offsets: A dictionary containing the x coordinates for each
                       offset in each ring
            y_offsets: A dictionary containing the y coordinates for each
                       offset in each ring
            x_outer_dict: A dictionary containing a set of x coordinates for
                          the outer circle defining each ring of regions
            yu_outer_dict: A dictionary containing a set of y coordinates for
                           the upper half of the outer circle defining each 
                           ring of regions
            yl_outer_dict: A dictionary containing a set of y coordinates for
                           the lower half of the outer circle defining each
                           ring of regions
            boundary_coords_dict: A dictionary containing the coordinates that
                                  specify the boundaries between regions in
                                  the same ring
            N_rings_of_offsets: The number of rings of offsets, not counting
                                the zeroth ring that consists of the offset
                                at (0,0) and its region.
            N_offsets_per_ring: The number of offsets in each ring.
     Output:
            A plot of all of the offsets and their corresponding regions
    '''
    
    # Set up the figure window
    fig,ax = plt.subplots()
    
    # Iteratively draw each ring of offsets and regions
    # The zeroth ring is a special case
    # Plot the central dot
    line_rc, = ax.plot(0,0,'k.')
    # Plot the outer circular boundary
    line_rou, = ax.plot(x_outer_dict[0],yu_outer_dict[0],'k')
    line_rol, = ax.plot(x_outer_dict[0],yl_outer_dict[0],'k')
    
    # For the remaining rings, use Plot_Offsets_And_Region_Outlines                                 
    for iring in np.arange(1,N_rings_of_offsets+1):
        Plot_Offsets_And_Region_Outlines(x_offsets[iring],y_offsets[iring],\
                                         x_outer_dict[iring],yu_outer_dict[iring],\
                                         yl_outer_dict[iring],boundary_coords_dict[iring],\
                                         N_offsets_per_ring[iring-1],fig,ax)
        # Note: N_offsets_per_ring doesn't include the zeroth ring.
    
    
    # Make the plot window appear
    plt.axis('equal')
    plt.show()

###############################################################################
# Functions for saving the offsets and the areas of their corresponding
# regions to a fits file

def save_offsets_and_areas(x_offsets,y_offsets,A_offsets,N_rings_of_offsets):
    '''
     This function saves the coordinates of the offset sources and the areas
     of the regions they represent.
     Inputs:
            x_offsets: A dictionary containing the x coordinates for each
                       offset in each ring
            y_offsets: A dictionary containing the y coordinates for each
                       offset in each ring
            A_offsets: A dictionary containing the area of each region for
                       each offset in each ring
            N_rings_of_offsets: Number of rings of offset sources not
                                counting the onaxis source (the zeroth ring)
     Outputs:
            a text file containing the offsets and areas
    '''
    
    # Save the offsets and the normalized areas to a text file.
    offset_data_file = 'offsets_and_areas.txt'
    # Header Data
    fields = ['x_off','y_off','Anorm']
    # Begin writing the file with the header data
    # Write the file
    with open(offset_data_file,'w') as csvfile:
        # create the csv writer object
        csvwriter=csv.writer(csvfile)
        # write the fields
        csvwriter.writerow(fields)

    # Now, iteratively add the data for each ring of offsets
    for iring in np.arange(N_rings_of_offsets+1):
        # Extract the data for the ring
        x_offsets_iring = x_offsets[iring]
        y_offsets_iring = y_offsets[iring]
        A_offsets_iring = A_offsets[iring]
        # The zeroth ring has only one data point (the onaxis source)
        if iring==0:
            # Write the data for the zeroth ring
            ring_data = [np.format_float_positional(x_offsets_iring),\
                         np.format_float_positional(y_offsets_iring),\
                         np.format_float_positional(A_offsets_iring)]
            with open(offset_data_file,'a') as csvfile:
                # create the csv writer object
                csvwriter=csv.writer(csvfile)
                # write the fields
                csvwriter.writerow(ring_data)
        else:
            # The remaining rings have multiple data points
            # Iterate over each region
            Nregions_iring = x_offsets_iring.shape[0]
            for ireg in range(Nregions_iring):
                # Write the data for region ireg in ring iring
                ring_data = [np.format_float_positional(x_offsets_iring[ireg]),\
                             np.format_float_positional(y_offsets_iring[ireg]),\
                             np.format_float_positional(A_offsets_iring)]
                with open(offset_data_file,'a') as csvfile:
                    # create the csv writer object
                    csvwriter=csv.writer(csvfile)
                    # write the fields
                    csvwriter.writerow(ring_data)
                    
###############################################################################
# Function for loading a predefined set of parameters for the jitter and finite
# stellar diameter calculations

def load_predefined_jitter_and_stellar_diam_params(starID=None):
    '''
     This function loads a predefined set of parameters for running 
     Determine_offsets_and_areas. If a star ID is given, a specific example
     will be loaded for that star. Otherwise, the example will match the one
     given in John Krist's paper.
    
     Inputs:
            starID: Optional parameter to specify a particular star
    
     Outputs:
            stellar_diam_and_jitter_keywords: A dictionary containing all of the
                                              keywords necessary for running the jitter
                                              and finite stellar diameter calculations
    '''
    
    # The example stars that have been defined:
    stars_defined = {'47 UMa c'}
    
    # Check that starID specifies one of these example stars if provided:
    if starID != None:
        if starID not in stars_defined:
            raise KeyError("ERROR: Specified star is not in the list of examples for the jitter and finite stellar diameter calculations.")
            
    # Initialize the dictionary
    stellar_diam_and_jitter_keywords = {}
    
    # Load the appropriate example
    if (starID == '47 UMa c') or (starID == None):
        # Use parameters that approximately reproduce John Krist's example
        
        # Stellar diameter (used for top hat function)
        stellar_diam_and_jitter_keywords['r_stellar_disc_mas'] = 0.45
        stellar_diam_and_jitter_keywords['stellar_diam_mas'] = 2*stellar_diam_and_jitter_keywords['r_stellar_disc_mas']
        
        # Offset array parameters
        stellar_diam_and_jitter_keywords['N_rings_of_offsets'] = 11
        stellar_diam_and_jitter_keywords['N_offsets_per_ring'] = np.array([6,8,12,14,12,10,14,10,14,10,14])
        stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'] = np.array([90,0,45,0,45,0,90,0,90,0,90])
        stellar_diam_and_jitter_keywords['r_ring0'] = 0.075
        stellar_diam_and_jitter_keywords['dr_rings'] = np.array([0.15,0.15,0.15,0.15,0.2,0.4,0.4,0.8,0.8,1.6,1.6])
        stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'] = 6.475
        stellar_diam_and_jitter_keywords['use_finite_stellar_diam'] = 1
        stellar_diam_and_jitter_keywords['add_jitter'] = 0
        #stellar_diam_and_jitter_keywords['use_saved_deltaE_and_weights'] = 0
        
    # Return the dictionary
    return stellar_diam_and_jitter_keywords
###############################################################################
# Functions for building the library of delta electric fields and weights

def build_delta_e_field_library(stellar_diam_and_jitter_keywords,optics):
    '''
    This function builds a list of offset sources, the area of the region 
    represented by each offset, and the weight for each offset;
    calculates the electric field for each offset; and saves a library of 
    delta electric fields for use in calculating the effects of jitter or 
    finite stellar diameter.

    Inputs:
           stellar_diam_and_jitter_keywords: A dictionary containing the necessary information
                                             for the jitter and finite stellar diameter calculations
           optics: a CorgiOptics object that defines the configuration of the CGI optics
    Outputs:
           stellar_diam_and_jitter_keywords, updated to include the offsets, the areas of the
                                             regions represented by the offsets, and the
                                             delta electric fields in offset_field_data

    '''
    # Step 0: Verify that the required information is all contained in the
    #         keyword dictionary.
    required_keys = {'outer_radius_of_offset_circle','N_rings_of_offsets',\
                     'N_offsets_per_ring','starting_offset_ang_by_ring',\
                     'r_ring0','dr_rings'}
    missing_keys = required_keys - stellar_diam_and_jitter_keywords.keys()
    if missing_keys:
        raise KeyError(f"ERROR: Missing required keywords: {missing_keys}")
        
    # Step 1: Generate the list of offset sources and the area of the
    #         region represented by each offset.
    
    x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, yl_outer_dict, boundary_coords_dict \
        = Determine_offsets_and_areas(stellar_diam_and_jitter_keywords['outer_radius_of_offset_circle'], \
                                      stellar_diam_and_jitter_keywords['N_rings_of_offsets'], \
                                      stellar_diam_and_jitter_keywords['N_offsets_per_ring'], \
                                      stellar_diam_and_jitter_keywords['starting_offset_ang_by_ring'], \
                                      stellar_diam_and_jitter_keywords['r_ring0'], \
                                      stellar_diam_and_jitter_keywords['dr_rings'])
    #print('Offsets calculated!')
    
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
    
    # Step 2: Calculate the weight for each offset
    stellar_diam_and_jitter_keywords = calculate_weights_for_jitter_and_finite_stellar_diameter(stellar_diam_and_jitter_keywords)
            
    # Step 3: Calculate the onaxis electric field 
    # The specific electric field components calculated will vary depending on
    # the polarization case.
    if optics.prism == 'POL0':
        # 0/90 deg polarization case
        # models the polarization aberration of the speckle field
        # polaxis=-1 and 1 gives -45->X and 45->X aberrations, incoherently
        # averaging the two gives the x polarized intensity data.
        # polaxis=-2 and 2 gives -45->Y and 45->Y aberrations, incoherently
        # averaging the two gives the y polarized intensity data. 
        polaxis_params = [-1, 1, -2, 2]
        E0_components = [] # There will be four calculated fields
        optics_keywords_pol_xy = optics.optics_keywords.copy()
        for polaxis in polaxis_params:
            optics_keywords_pol_xy['polaxis'] = polaxis
            (E0, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_pol_xy,QUIET=optics.quiet)
            E0_components.append(E0)
    elif optics.prism == 'POL45':
        # 45/135 deg polarization case
        # models the polarization aberration of the speckle field
        # polaxis=-3 and 3 gives -45->45 and 45->45 aberrations, incoherently
        # averaging the two gives the 45 degree polarized intensity data.
        # polaxis=-2 and 2 gives -45->-45 and 45->-45 aberrations, incoherently
        # averaging the two gives the -45 degree polarized intensity data. 
        polaxis_params = [-3, 3, -4, 4]
        E0_components = [] # There will be four calculated fields
        optics_keywords_pol_45 = optics.optics_keywords.copy()
        for polaxis in polaxis_params:
            optics_keywords_pol_45['polaxis'] = polaxis
            (E0, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_pol_45,QUIET=optics.quiet)
            E0_components.append(E0)
    elif optics.optics_keywords['polaxis'] == -10:
        # if polaxis is set to -10, obtain full aberration model by individually summing intensities obtained from polaxis=-2, -1, 1, 2
        optics_keywords_m10 = optics.optics_keywords.copy()
        polaxis_params = [-2, -1, 1, 2]
        E0_components = [] # There will be four calculated fields
        for polaxis in polaxis_params:
            optics_keywords_m10['polaxis'] = polaxis
            (E0, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_m10,QUIET=optics.quiet)
            E0_components.append(E0)
    else: 
        # use built in polaxis settings to obtain specific/averaged aberration 
        E0_components = []
        (E0, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics.optics_keywords,QUIET=optics.quiet)
        E0_components.append(E0)
        
    print('Onaxis field calculated')
        
    # Step 4: Build the library of delta electric fields
    # The specific library calculations will vary depending on the polarization
    # case.
    if optics.prism == 'POL0':
        #0/90 case
        # models the polarization aberration of the speckle field
        # Four electric field components were calculated.
        # polaxis=-1 and 1 gives -45->X and 45->X aberrations, incoherently
        # averaging the two gives the x polarized intensity data.
        # polaxis=-2 and 2 gives -45->Y and 45->Y aberrations, incoherently
        # averaging the two gives the y polarized intensity data. 
     
        # Initialize the dictionary that will store the delta electric fields
        field_shape = E0_components[0].shape
        delta_e_fields = {'delta_E_m45in_xout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                          'delta_E_45in_xout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                          'delta_E_m45in_yout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                          'delta_E_45in_yout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex)}
     
        # Iterate over each offset
        for i_offset in np.arange(N_offsets):
         
            # Change the x and y offset sent to the proper model
            optics_keywords_pol_xy['source_x_offset_mas'] = x_offsets_list[i_offset]
            optics_keywords_pol_xy['source_y_offset_mas'] = y_offsets_list[i_offset]
         
            # Calculate the offset electric field for each polarization component
            offset_E_components = []
            for polaxis in polaxis_params:
                optics_keywords_pol_xy['polaxis'] = polaxis
                (Eoff, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_pol_xy,QUIET=optics.quiet)
                offset_E_components.append(Eoff)
             
            # Calculate the delta electric fields and save
            delta_e_fields['delta_E_m45in_xout'][i_offset,:,:,:] = offset_E_components[0] - E0_components[0]
            delta_e_fields['delta_E_45in_xout'][i_offset,:,:,:] = offset_E_components[1] - E0_components[1]
            delta_e_fields['delta_E_m45in_yout'][i_offset,:,:,:] = offset_E_components[2] - E0_components[2]
            delta_e_fields['delta_E_45in_yout'][i_offset,:,:,:] = offset_E_components[3] - E0_components[3]
         
    elif optics.prism == 'POL45':
         #45/135 case
         # models the polarization aberration of the speckle field
         # Four electric field components were calculated.
         # polaxis=-3 and 3 gives -45->45 and 45->45 aberrations, incoherently
         # averaging the two gives the 45 degree polarized intensity data.
         # polaxis=-2 and 2 gives -45->-45 and 45->-45 aberrations, incoherently
         # averaging the two gives the -45 degree polarized intensity data. 
         
         # Initialize the dictionary that will store the electric fields
         field_shape = E0_components[0].shape
         delta_e_fields = {'delta_E_m45in_45out':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_45in_45out':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_m45in_135out':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_45in_135out':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex)}
         
         # Iterate over each offset
         for i_offset in np.arange(N_offsets):
             
             # Change the x and y offset sent to the proper model
             optics_keywords_pol_45['source_x_offset_mas'] = x_offsets_list[i_offset]
             optics_keywords_pol_45['source_y_offset_mas'] = y_offsets_list[i_offset]
             
             # Calculate the offset electric field for each polarization component
             offset_E_components = []
             for polaxis in polaxis_params:
                 optics_keywords_pol_45['polaxis'] = polaxis
                 (Eoff, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_pol_45,QUIET=optics.quiet)
                 offset_E_components.append(Eoff)
                 
             # Calculate the delta electric fields and save
             delta_e_fields['delta_E_m45in_45out'][i_offset,:,:,:] = offset_E_components[0] - E0_components[0]
             delta_e_fields['delta_E_45in_45out'][i_offset,:,:,:] = offset_E_components[1] - E0_components[1]
             delta_e_fields['delta_E_m45in_135out'][i_offset,:,:,:] = offset_E_components[2] - E0_components[2]
             delta_e_fields['delta_E_45in_135out'][i_offset,:,:,:] = offset_E_components[3] - E0_components[3]
            
    elif optics.optics_keywords['polaxis'] == -10:
         # if polaxis is set to -10, obtain full aberration model by individually summing intensities obtained from polaxis=-2, -1, 1, 2
         # Four electric field components were calculated.
         
         # Initialize the dictionary that will store the electric fields
         field_shape = E0_components[0].shape
         delta_e_fields = {'delta_E_m45in_yout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_m45in_xout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_45in_xout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex),
                           'delta_E_45in_yout':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex)}
         
         # Iterate over each offset
         for i_offset in np.arange(N_offsets):
             
            # Chanage the x and y offset sent to the proper model
            optics_keywords_m10['source_x_offset_mas'] = x_offsets_list[i_offset]
            optics_keywords_m10['source_y_offset_mas'] = y_offsets_list[i_offset]
             
            # Calculate the offset electric field for each polarization component
            offset_E_components = []
            for polaxis in polaxis_params:
                optics_keywords_m10['polaxis'] = polaxis
                (Eoff, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics_keywords_m10,QUIET=optics.quiet)
                offset_E_components.append(Eoff)
                 
            # Calculate the delta electric fields and save
            delta_e_fields['delta_E_m45in_yout'][i_offset,:,:,:] = offset_E_components[0] - E0_components[0]
            delta_e_fields['delta_E_m45in_xout'][i_offset,:,:,:] = offset_E_components[1] - E0_components[1]
            delta_e_fields['delta_E_45in_xout'][i_offset,:,:,:] = offset_E_components[2] - E0_components[2]
            delta_e_fields['delta_E_45in_yout'][i_offset,:,:,:] = offset_E_components[3] - E0_components[3]

    else: 
        # use built in polaxis settings to obtain specific/averaged aberration 
        
        # Initialize the dictionary that will store the electric field
        field_shape = E0_components[0].shape
        delta_e_fields = {'delta_E':np.zeros((N_offsets,field_shape[0],field_shape[1],field_shape[2]),dtype=complex)}
        
        # Iterate over each offset
        for i_offset in np.arange(N_offsets):
            
            # Change the x and y offset sent to the proper model
            optics.optics_keywords['source_x_offset_mas'] = x_offsets_list[i_offset]
            optics.optics_keywords['source_y_offset_mas'] = y_offsets_list[i_offset]
            
            # Calculate the offset electric field
            (Eoff, sampling) = proper.prop_run_multi('roman_preflight',  optics.lam_um, 1024,PASSVALUE=optics.optics_keywords,QUIET=optics.quiet)
            
        # Calculate the delta electric fields and save
        delta_e_fields['delta_E'][i_offset,:,:,:] = Eoff - E0_components[0]
            
            
    # Add the delta_e_fields to the offset_field_data dictionary
    stellar_diam_and_jitter_keywords['offset_field_data']['delta_e_fields'] = delta_e_fields    
    
    # To prevent recalculating the library when it already exists, update
    # stellar_diam_and_jitter_keywords['use_saved_deltaE_and_weights'] to 2
    stellar_diam_and_jitter_keywords['use_saved_deltaE_and_weights'] = 2
    
    # Return the updated dictionary
    return stellar_diam_and_jitter_keywords

def calculate_weights_for_jitter_and_finite_stellar_diameter(stellar_diam_and_jitter_keywords):
    '''
    This function calculates the set of weights to use when combining the
    intensities of the offset sources to add the effects of jitter and the
    finite stellar diameter
    
    Inputs:
           stellar_diam_and_jitter_keywords: A dictionary containing the necessary information
                                  for the jitter and finite stellar diameter calculations
    Outputs:
           stellar_diam_and_jitter_keywords, updated to include the weights in offset_field_data
    '''
    
    # Is the finite stellar diameter being considered?
    if ('use_finite_stellar_diam' in stellar_diam_and_jitter_keywords.keys() ) and \
        stellar_diam_and_jitter_keywords['use_finite_stellar_diam'] == 1:
            stellar_diam_flag = 1
    else:
        stellar_diam_flag = 0
        
    # Is jitter being considered?
    if ('add_jitter' in stellar_diam_and_jitter_keywords.keys() ) and \
        stellar_diam_and_jitter_keywords['add_jitter'] == 1:
            jitter_flag = 1
    else:
        jitter_flag = 0
    
    # Case 1: Finite stellar diameter only
    if (stellar_diam_flag == 1) and (jitter_flag == 0):
        # The weighting function is a top-hat function whose radius matches the
        # radius of the stellar disc.
        
        # If the radius of the stellar disc has already been calculated, verify
        # that it is equal to half of the stellar diameter. If needed, calculate
        # the radius
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
            
    x = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid)
    y = np.linspace(-outer_radius_of_offset_circle,outer_radius_of_offset_circle,N_offsetgrid)
    X,Y = np.meshgrid(x,y)
    
    # Define the top-hat function for the stellar disc if appropriate
    if stellar_diam_flag == 1:
        disc_indices = (X**2 + Y**2 <= r_stellar_disc_mas**2)
        
    # Define the Gaussia if appropriate
    # TODO: Add the Gaussian for jitter
    
    # Specify the weights, Wjitdisc
    if (stellar_diam_flag == 1) and (jitter_flag == 0):
        # Stellar diameter only
        # Weights are set by the top-hat function
        Wjitdisc = disc_indices
        
    # Resample Wjitdisc to match the predetermined offsets
    # Step 1: Interpolate
    #f_interp = interpolate.interp2d(x,y,Wjitdisc,kind='quintic')
    f_interp = RectBivariateSpline(x,y,Wjitdisc.T)
    # Step 2: Specify what offsets to use for the grid points after interpolation
    # The predetermined offsets:
    x_predetermined = stellar_diam_and_jitter_keywords['offset_field_data']['x_offsets_mas']
    y_predetermined = stellar_diam_and_jitter_keywords['offset_field_data']['y_offsets_mas']
    # Step 3: Obtain the result.
    W = np.zeros(stellar_diam_and_jitter_keywords['N_offsets_counting_origin'])
    for i_offset in range(stellar_diam_and_jitter_keywords['N_offsets_counting_origin']):
        W[i_offset] = f_interp(x_predetermined[i_offset],y_predetermined[i_offset])
        
    # Since the interpolation is no longer symmetric about the x axis and about the y axis,
    # add that symmetry back by averaging the weights for points in the same ring that 
    # share the same x or y coordinate, and assigning those points the average weight.
    
    # There are stellar_diam_and_jitter_keywords['N_rings_of_offsets'] rings,
    # and the number of regions in each ring is contained in
    # stellar_diam_and_jitter_keywords['N_offsts_per_ring].
    
    # Average the weights for points in the same ring that are symmetric about the y axis.
    for i in range(stellar_diam_and_jitter_keywords['N_rings_of_offsets']): # Iterate over each ring
        # Extract the number of regions in that ring
        nreg_ringi = stellar_diam_and_jitter_keywords['N_offsets_per_ring'][i]
        #print('Starting to apply y axis symmetry for Ring %d.' % i)
        ring_end_index = 1+np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'][0:i+1])
        #print('Ring ends at index %d' % ring_end_index)
        for j in range(nreg_ringi): # Iterate over each region within the ring
            offset_index = 1+np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'][0:i])+j # Which element in the coordinate array to use
            # Extract the x coordinate for the jth region of ring i
            x_test = x_predetermined[offset_index]
            y_test = y_predetermined[offset_index]
            #print('Offset_index: %d, x coordinate: %.8f' % (offset_index,x_test))
            # Check if any of the remaining regions in the ring share the same x coord
            # Note that we don't need to check a point against itself or repeat a check
            # more than once.
            if offset_index < ring_end_index-2: # 
                for k in range(offset_index+1,ring_end_index):
                    if np.round(x_test,decimals=6) == -np.round(x_predetermined[k],decimals=6) and np.round(y_test,decimals=6) == np.round(y_predetermined[k],decimals=6):
                        #print('Comparing offset %d against offset %d: Match' % (offset_index,k))
                        # Average the weights, and assign the result to both points
                        Wavg = (W[offset_index]+W[k])/2
                        W[offset_index] = Wavg
                        W[k] = Wavg
                    #else:
                        #print('Comparing offset %d against offset %d: Not a match' % (offset_index,k))
            elif offset_index == ring_end_index-2: # Only one last point to check
                if np.round(x_test,decimals=6) == -np.round(x_predetermined[offset_index+1],decimals=6) and np.round(y_test,decimals=6) == np.round(y_predetermined[offset_index+1],decimals=6):
                    #print('Comparing offset %d against offset %d: Match' % (offset_index,ring_end_index-1))
                    # Average the weights, and assign the result to both points
                    Wavg = (W[offset_index]+W[offset_index+1])/2
                    W[offset_index] = Wavg
                    W[offset_index+1] = Wavg
                #else:
                    #print('Comparing offset %d against offset %d: Not a match' % (offset_index,ring_end_index-1))
                    
    # Average the weights for points in the same ring that are symmetric about the x axis.
    for i in range(stellar_diam_and_jitter_keywords['N_rings_of_offsets']): # Iterate over each ring
        # Extract the number of regions in that ring
        nreg_ringi = stellar_diam_and_jitter_keywords['N_offsets_per_ring'][i]
        #print('Starting to apply x axis symmetry for Ring %d.' % i)
        ring_end_index = 1+np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'][0:i+1])
        #print('Ring ends at index %d' % ring_end_index)
        for j in range(nreg_ringi): # Iterate over each region within the ring
            offset_index = 1+np.sum(stellar_diam_and_jitter_keywords['N_offsets_per_ring'][0:i])+j # Which element in the coordinate array to use
            # Extract the x coordinate for the jth region of ring i
            x_test = x_predetermined[offset_index]
            y_test = y_predetermined[offset_index]
            #print('Offset_index: %d, x coordinate: %.8f' % (offset_index,x_test))
            # Check if any of the remaining regions in the ring share the same x coord
            # Note that we don't need to check a point against itself or repeat a check
            # more than once.
            if offset_index < ring_end_index-2: # 
                for k in range(offset_index+1,ring_end_index):
                    if np.round(x_test,decimals=6) == np.round(x_predetermined[k],decimals=6) and np.round(y_test,decimals=6) == -np.round(y_predetermined[k],decimals=6):
                        #print('Comparing offset %d against offset %d: Match' % (offset_index,k))
                        # Average the weights, and assign the result to both points
                        Wavg = (W[offset_index]+W[k])/2
                        W[offset_index] = Wavg
                        W[k] = Wavg
                    #else:
                        #print('Comparing offset %d against offset %d: Not a match' % (offset_index,k))
            elif offset_index == ring_end_index-2: # Only one last point to check
                if np.round(x_test,decimals=6) == np.round(x_predetermined[offset_index+1],decimals=6) and np.round(y_test,decimals=6) == -np.round(y_predetermined[offset_index+1],decimals=6):
                    #print('Comparing offset %d against offset %d: Match' % (offset_index,ring_end_index-1))
                    # Average the weights, and assign the result to both points
                    Wavg = (W[offset_index]+W[offset_index+1])/2
                    W[offset_index] = Wavg
                    W[offset_index+1] = Wavg
                #else:
                    #print('Comparing offset %d against offset %d: Not a match' % (offset_index,ring_end_index-1))
        
    # Normalize W to a total of 1.0
    Wtot = np.sum(W)
    #print(Wtot)
    Wnorm = W/Wtot
    # Finally, multiply Wnorm by the normalized area for each predetermined offset
    Anorm = stellar_diam_and_jitter_keywords['offset_field_data']['A_offsets'] # The normalized areas
    offset_weights = np.zeros(stellar_diam_and_jitter_keywords['N_offsets_counting_origin']) # Array to store the final weights
    for i in range(stellar_diam_and_jitter_keywords['N_offsets_counting_origin']):
        offset_weights[i] = Wnorm[i]*Anorm[i]
        
    # Add the offset weights to the offset field data dictionary
    stellar_diam_and_jitter_keywords['offset_field_data']['offset_weights'] = offset_weights
    
    # Return the updated dictionary
    return stellar_diam_and_jitter_keywords

###############################################################################

        
        