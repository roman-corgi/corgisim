#   Copyright 2020 California Institute of Technology
# ------------------------------------------------------------------

# Version 0.9, 15 June 2020, John Krist; beta-test version for mask evaluation
# Version 0.9.1, 29 June 2020, John Krist; rotated SPC pupils & Lyot stops 
# Version 0.9.2, 7 July 2020, John Krist; updated HLC occulters
# Version 1.0, 1.0.1, 15 Sept 2020, John Krist:
#   updated to designated flight masks; replaced some error maps with measured ones; added thick lenses
# Version 1.0.2, 28 Oct 2020, John Krist: added optional mask array parameters (for JPL internal uses)
# Version 1.2, 19 Feb 2021, John Krist: added code to subtract quadratic phase term in final image due
#     to conjugate pupil not at front focus of imaging lens (only applied when NOT using PIL or defocus
#     lenses); switched to MFT-based propagation to/from HLC FPM at high sampling to avoid interpolation
#     errors; added spc-mswc mode (provided by A.J. Riggs) for multi-star wavefront control modeling;
#     updated error maps; removed user-specfied lam0 and lambda0_m parameters (use only hard-coded values); 
#     changed default grid sizes, including getting rid of intermediate grid size changes; added secondary
#     mirror despace parameter; read in list of available HLC FPM files; removed 
#     propagation to fold 4 (not significant); added TO_PLANE option to thick lenses; added options
#     for using Band 1 with the spc-wide and spc-mswc; removed the output_field_rootname option;
#     added Phase C polarization aberrations from Jim McGuire; added HLCs for band 2-4 and rotated SPC
# Version 1.2.1, 14 May 2021, John Krist: added option to use field_stop_array with hlc
# Version 1.2.2, 20 May 2021, John Krist: updated aperture sizes; use 2Kx2K grids for HLC when phase
#     retrieval lenses used; use OAP6 aperture when pinhole is specified
# Version 1.2.3, 15 July 2021, John Krist: added dm_sampling_m as optional parameter
# Version 1.2.4, 3 Aug 2021, John Krist: added HLC FPM files for band 3g 
# Version 1.2.5, 20 Sept 2021, John Krist: added use_lens_errors parameter, 
#                 updated defocus and PIL error maps; added facility for separate error maps for each 
#                 lens in to_from_doublet 
# Version 1.2.6, 7 Oct 2021, John Krist: added save_ref_radius to optional parameters, used for phase 
#                 retrieval studies in conjuction with CGISim (Python only); this value is only valid 
#                 if propagation ends at the detector
# Version 1.2.7, 21 Oct 2021, John Krist: if SPC and use_pupil_mask eq 0, then use fold mirror 
#                 errors rather than SPC substrate errors
# Version 1.2.8, 11 Nov 2021, John Krist: Added option for spc-spec_band2_rotated; fixed rotated SPC
#                 mask orientations 
# Version 1.2.9, 15 March 2022, John Krist: Added option for beam displacement at the SPAM, allow HLC
#                 to use pupil_mask_array, and add image shift at the detector 
# Version 1.2.9a, 11 April 2022, John Krist: fixed contributed HLC bands 2-4 FPM orientations; set PIL
#                     separation to 0.5 mm
# Version 1.2.9b, 31 May 2022, John Krist: changed spam_*_shift_* to work only if SPC modes used, 
#                     otherwise those values are ignored
# Version 1.3, June 2022, John Krist: added option to use CVS instead of telescope (use_cvs); 
#                     additional CVS-related parameters: cvs_source_z_offset_m, cvs_jitter_mirror_*_offset_*,
#                     cvs_stop_*; now incorporates sampling changes at FPM and pupil masks (may be caused by
#                     motions of secondary, CVS source or FCM); added ability to rotate pupil masks (CVS OTA, SPC, Lyot);
#                     added measured prescription values for PIL & defocus lenses (from David Marx);
#                     changed order of SPAM beam and SPC mask shifts
# Version 1.4, Sept 2022, A.J. Riggs: added option to use Zernike WFS; J. Krist: adjusted orientation of rotated SPC;
#                   fixed orientation of CVS stop
# Version 1.5, March 2023: D. Marx: added excam_despace_m optional parameter; updated properties, including error maps,
#                   for defocus and pupil imaging lenses:
#                   corrected values for radius of curvatures calculated from measured Fringes for plano surfaces for the 
#                   defocus lenses and PIL. Was: calculated using 10mm clear aperture; Now: calculated using 8mm clear aperture
#                   Defocus 1 r2: was -146.3, now -93.62
#                   Defocus 2 r2: was -153.7, now -98.35
#                   Defocus 3 r1: was 0.085371, now 0.085518
#                             r2: was -154.9, now -99.12
#                   Defocus 4 r2: was -110.0, now -70.41
#                   PIL R1_b: was 858.2, now 190.0 ('was' value was also based on wrong FR value)
# Version 2.0, May 2023: J. Krist: renamed from roman_phasec to roman_preflight; now use cgi_dm for flight DM model, with 
#           DM voltages rather than strokes as inputs; removed use_lens_errors option (now just uses value of use_errors); 
#           added CFAM baffle (only propagate to there if use_aperture = 1); now rotate images to CGI image orientation; source and FSM 
#           positional offsets modified to match CGI image orientations; source and fsm positional offsets modified to match
#           new output orientation;FOCM changed to FCM, including offsets; added other miscellaneous baffles, used only when
#           use_aperture = 1; changed axis of DM tilts; baseline SPCs (spc-spec, spec-wide; not contributed modes) can now use smaller 
#           pupil grids (500 pix across pupil); added "small_spc_grid" option to compute baseline SPCs at half default pupil resolution
#           (500 rather than 1000 pix across pupil) resolution. Changed axis of DM tilt. DM2 is offset from DM1 by 0.1 actuator in X. 


# Experience has shown that with the Intel math libraries the program
# can hang if run in a multithreaded condition with more than one 
# thread allocated, so set to 1. This must be done before importing numpy

import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

import proper
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from roman_preflight_proper import trim, mft2, ffts, polmap, shift_image, transform_image 
import roman_preflight_proper as rp

# Written by John Krist


###############################################################################33
def radius( n ):
    x = np.arange( n ) - int(n)//2
    r = np.sqrt( x*x + x[:,np.newaxis]**2 )
    return r

###############################################################################33
def angle( n ):
    x = np.arange( n ) - int(n)//2
    return np.arctan2( x[:,np.newaxis], x ) * (180 / np.pi)

###############################################################################33
def glass_index( glass, lambda_m, data_dir ):
    a = np.loadtxt( data_dir + '/glass/' + glass+'_index.txt' )  # lambda_um index pairs
    f = interp1d( a[:,0], a[:,1], kind='cubic' )
    return f( lambda_m*1e6 )

###############################################################################33
def to_from_singlet( wavefront, dz_to_lens, dz_from_lens, r1, r2, thickness, nglass, 
    surface_name, next_surface_name, ERROR_MAP_FILE=' ', TO_PLANE=0):

    # single thick lens
    # r1, r2 = front & rear radii of curvature

    f = 1 / ( (nglass - 1) * ( 1.0/r1 - 1.0/r2 + (nglass - 1)*thickness / (nglass*r1*r2) ) )      # effective focal length
    h1 = -f*(nglass - 1)*thickness / (nglass*r2)    # front principal plane
    h2 = -f*(nglass - 1)*thickness / (nglass*r1)    # rear principal plane

    proper.prop_propagate( wavefront, dz_to_lens+h1, surface_name )
    proper.prop_lens( wavefront, f, surface_name )
    if ERROR_MAP_FILE != ' ': 
        proper.prop_errormap( wavefront, ERROR_MAP_FILE, WAVEFRONT=True )
    proper.prop_propagate( wavefront, -h2+dz_from_lens, next_surface_name, TO_PLANE=TO_PLANE )

    return

###############################################################################33
def to_from_doublet( wavefront, dz_to_lens, dz_from_lens, r1_a, r2_a, thickness_a, nglass_a, separation_m, 
    r1_b, r2_b, thickness_b, nglass_b, surface_name, next_surface_name, ERROR_MAP_FILES=' ', TO_PLANE=0):

    # two air-gapped thick lenses (lenses a & b)

    f_a = 1 / ( (nglass_a - 1) * ( 1.0/r1_a - 1.0/r2_a + (nglass_a - 1)*thickness_a / (nglass_a*r1_a*r2_a) ) )
    h1_a = -f_a*(nglass_a - 1)*thickness_a / (nglass_a*r2_a)
    h2_a = -f_a*(nglass_a - 1)*thickness_a / (nglass_a*r1_a)

    f_b = 1 / ( (nglass_b - 1) * ( 1.0/r1_b - 1.0/r2_b + (nglass_b - 1)*thickness_b / (nglass_b*r1_b*r2_b) ) )
    h1_b = -f_b*(nglass_b - 1)*thickness_b / (nglass_b*r2_b)
    h2_b = -f_b*(nglass_b - 1)*thickness_b / (nglass_b*r1_b)

    proper.prop_propagate( wavefront, dz_to_lens+h1_a, surface_name )
    proper.prop_lens( wavefront, f_a, surface_name+' lens #1' )
    if ERROR_MAP_FILES != ' ':
        proper.prop_errormap( wavefront, ERROR_MAP_FILES[0], WAVEFRONT=True )
    proper.prop_propagate( wavefront, -h2_a+separation_m+h1_b )
    proper.prop_lens( wavefront, f_b, surface_name+' lens #2' )
    if ERROR_MAP_FILES != ' ':
        if ERROR_MAP_FILES[1] != ' ':
            proper.prop_errormap( wavefront, ERROR_MAP_FILES[1], WAVEFRONT=True )
    proper.prop_propagate( wavefront, -h2_b+dz_from_lens, next_surface_name, TO_PLANE=TO_PLANE )

    return

###############################################################################33
def roman_preflight( lambda_m, output_dim0, PASSVALUE={'dummy':0} ):

    # "output_dim" is used to specify the output dimension in pixels at the final image plane.
    # Computational grid sizes are hardcoded for each coronagraph.

    use_cvs = 0                 # use CVS instead of telescope? (1=yes, 0=no)
    cvs_stop_x_shift_m = 0      # shift of CVS entrance pupil mask in meters
    cvs_stop_y_shift_m = 0
    cvs_stop_z_shift_m = 0      # shift of CVS entrance pupil mask along optical axis (+ is downstream)
    cvs_stop_rotation_deg = 0   # rotation of CVS entrance pupil mask in degrees
    small_spc_grid = 0          # set to 1 to use 500 pix across pupil, else 1000 (baseline SPCs only)
    pupil_array = 0             # 2D array containing pupil pattern (overrides default)
    pupil_mask_array = 0        # 2D array containing SPC pupil mask pattern (overrides default)
    fpm_array = 0               # 2D array containing FPM mask pattern (overrides default)
    fpm_mask = 0                # 2D array where 1=FPM pattern defined, 0=substrate
    lyot_stop_array = 0         # 2D array containing Lyot stop mask pattern (overrides default)
    field_stop_array = 0        # 2D array containing field stop mask pattern (overrides default)

    cor_type = 'hlc'            # coronagraph type ('hlc', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'none')
    source_x_offset_mas = 0     # source offset in mas (tilt applied at primary)
    source_y_offset_mas = 0                 
    source_x_offset = 0         # source offset in lambda0_m/D radians (tilt applied at primary)
    source_y_offset = 0                 
    cvs_source_z_offset_m = 0               # additional distance between CVS source and next optic, in meters
    cvs_jitter_mirror_x_offset_mas = 0      # source offset in milliarcsec (tilt applied at CVS jitter mirror)
    cvs_jitter_mirror_y_offset_mas = 0      # 
    cvs_jitter_mirror_x_offset = 0          # source offset in lambda0_m/D radians (tilt applied at CVS jitter mirror) 
    cvs_jitter_mirror_y_offset = 0
    polaxis = 0                 # polarization axis aberrations: 
                                #    -2 = -45d in, Y out 
                                #    -1 = -45d in, X out 
                                #     1 = +45d in, X out 
                                #     2 = +45d in, Y out 
                                #     5 = mean of modes -1 & +1 (X channel polarizer)
                                #     6 = mean of modes -2 & +2 (Y channel polarizer)
                                #    10 = mean of all modes (no polarization filtering)
    use_errors = 1              # use optical surface phase errors? 1 or 0 
    zindex = np.array([0,0])    # array of Zernike polynomial indices
    zval_m = np.array([0,0])    # array of Zernike coefficients (meters RMS WFE)
    sm_despace_m = 0            # secondary mirror despace (meters) 
    use_pupil_defocus = 1       # include pupil defocus
    use_aperture = 0            # use apertures on all optics? 1 or 0
    cgi_x_shift_pupdiam = 0     # X,Y shear of wavefront at FSM (bulk displacement of CGI); normalized relative to pupil diameter
    cgi_y_shift_pupdiam = 0          
    cgi_x_shift_m = 0           # X,Y shear of wavefront at FSM (bulk displacement of CGI) in meters
    cgi_y_shift_m = 0          
    end_at_fsm = 0              # end propagation after propagating to FSM (no FSM errors)
    fsm_x_offset_mas = 0        # offset in focal plane caused by tilt of FSM in mas
    fsm_y_offset_mas = 0         
    fsm_x_offset = 0            # offset in focal plane caused by tilt of FSM in lambda0/D
    fsm_y_offset = 0            
    fcm_z_shift_m = 0          # offset (meters) of focus correction mirror (+ increases path length)
    use_dm1 = 0                 # use DM1? 1 or 0
    use_dm2 = 0                 # use DM2? 1 or 0
    dm_version = rp.dm_version  # string, DM version
    dm_v_quant = 110.0 / 2.**16 # DM DAC voltage quantization resolution 
    dm_sampling_m = 0.9906e-3   # actuator spacing in meters
    dm_temp_c = 26.0
    dm1_v = np.zeros((48,48))
    dm1_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm1_yc_act = 23.5              
    dm1_xtilt_deg = 9.65        # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm1_ytilt_deg = 0 
    dm1_ztilt_deg = 0 
    dm2_v = np.zeros((48,48))
    dm2_xc_act = 23.5 - 0.1     # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm2_yc_act = 23.5               
    dm2_xtilt_deg = 9.65 
    dm2_ytilt_deg = 0
    dm2_ztilt_deg = 0
    spam_x_shift_pupdiam = 0    # X,Y shift of wavefront at SPAM; normalized relative to pupil diameter
    spam_y_shift_pupdiam = 0
    spam_x_shift_m = 0          # X,Y shift of wavefront at SPAM in meters
    spam_y_shift_m = 0
    use_pupil_mask = 1          # SPC only: use SPC pupil mask (0 or 1)
    mask_x_shift_pupdiam = 0    # X,Y shear of shaped pupil mask; normalized relative to pupil diameter
    mask_y_shift_pupdiam = 0          
    mask_x_shift_m = 0          # X,Y shear of shaped pupil mask in meters
    mask_y_shift_m = 0          
    mask_rotation_deg = 0
    use_fpm = 1                 # use occulter? 1 or 0
    fpm_x_offset = 0            # FPM x,y offset in lambda0/D
    fpm_y_offset = 0
    fpm_x_offset_m = 0          # FPM x,y offset in meters
    fpm_y_offset_m = 0
    fpm_z_shift_m = 0           # occulter offset in meters along optical axis (+ = away from prior optics)
    pinhole_diam_m = 0          # FPM pinhole diameter in meters
    end_at_fpm_exit_pupil = 0   # return field at FPM exit pupil?
    use_lyot_stop = 1           # use Lyot stop? 1 or 0
    lyot_x_shift_pupdiam = 0    # X,Y shear of Lyot stop mask; normalized relative to pupil diameter
    lyot_y_shift_pupdiam = 0  
    lyot_x_shift_m = 0          # X,Y shear of Lyot stop mask in meters
    lyot_y_shift_m = 0  
    lyot_rotation_deg = 0
    use_field_stop = 1          # use field stop (HLC)? 1 or 0
    field_stop_radius_lam0 = 0  # field stop radius in lambda0/D
    field_stop_x_offset = 0     # field stop offset in lambda0/D
    field_stop_y_offset = 0
    field_stop_x_offset_m = 0   # field stop offset in meters
    field_stop_y_offset_m = 0
    use_pupil_lens = 0          # use pupil imaging lens? 0 or 1
    use_defocus_lens = 0        # use defocusing lens? Options are 1, 2, 3, 4
    end_at_exit_pupil = 0       # return exit pupil corresponding to final image plane
    excam_despace_m = 0         # increase in spacing between final optic and detector
    final_sampling_m = 0        # final sampling in meters (overrides final_sampling_lam0)
    final_sampling_lam0 = 0     # final sampling in lambda0/D
    output_dim = output_dim0    # dimension of output in pixels (overrides output_dim0)
    image_x_offset_m = 0        # shift of image at detector plane in meters
    image_y_offset_m = 0

    save_ref_radius = 0

    data_dir = rp.data_dir
    map_dir = rp.map_dir
    polfile = rp.polfile
    dm_dir = rp.dm_files_dir

    if 'PASSVALUE' in locals():
        if 'data_dir' in PASSVALUE: 
            data_dir = PASSVALUE['data_dir']
            map_dir = data_dir + '/maps/'
            polfile = data_dir + '/pol/preflight_pol'
            dm_dir = data_dir + '/dm/'
        if 'cor_type' in PASSVALUE: cor_type = PASSVALUE['cor_type']
        if 'use_cvs' in PASSVALUE: use_cvs = PASSVALUE['use_cvs']
        if 'use_fpm' in PASSVALUE: use_fpm = PASSVALUE['use_fpm']
        if 'small_spc_grid' in PASSVALUE: small_spc_grid = PASSVALUE['small_spc_grid']
        if 'pupil_array' in PASSVALUE: pupil_array = PASSVALUE['pupil_array']
        if 'pupil_mask_array' in PASSVALUE: pupil_mask_array = PASSVALUE['pupil_mask_array']
        if 'fpm_array' in PASSVALUE: 
            fpm_array = PASSVALUE['fpm_array']
            fpm_array_sampling_m = PASSVALUE['fpm_array_sampling_m']
        if 'lyot_stop_array' in PASSVALUE: lyot_stop_array = PASSVALUE['lyot_stop_array']
        if 'field_stop_array' in PASSVALUE: 
            field_stop_array = PASSVALUE['field_stop_array']
            field_stop_array_sampling_m = PASSVALUE['field_stop_array_sampling_m']
        if 'use_pupil_lens' in PASSVALUE: use_pupil_lens = PASSVALUE['use_pupil_lens']
        if 'use_defocus_lens' in PASSVALUE: use_defocus_lens = PASSVALUE['use_defocus_lens']

    is_spc = False
    is_hlc = False
    is_zwfs = False

    if cor_type.find('hlc') != -1:
        is_hlc = True
        if cor_type == 'hlc' or cor_type == 'hlc_band1':
            file_directory = data_dir + '/hlc_20190210b/'         # must have trailing "/"
            lambda0_m = 0.575e-6
        elif cor_type == 'hlc_band2':
            file_directory = data_dir + '/hlc_20200617c_band2/'   # must have trailing "/"
            lambda0_m = 0.660e-6
        elif cor_type == 'hlc_band3':
            file_directory = data_dir + '/hlc_20200614b_band3/'   # must have trailing "/"
            lambda0_m = 0.730e-6
        elif cor_type == 'hlc_band4':
            file_directory = data_dir + '/hlc_20200609b_band4/'   # must have trailing "/"
            lambda0_m = 0.825e-6
        else:
            print("Unsupported HLC mode")
            raise Exception(' ')
        pupil_diam_pix = 309.0
        if use_cvs != 0:
            pupil_file = file_directory + 'pupil_rotated.fits'
        else:
            pupil_file = file_directory + 'pupil.fits'
        if use_fpm != 0:
            # find nearest available FPM wavelength that matches specified wavelength
            lam_um = lambda_m * 1e6
            f = open( file_directory + 'fpm_files.txt' )
            fpm_nlam = int(f.readline())
            fpm_lam_um = np.zeros((fpm_nlam),dtype=float)
            for j in range(0,fpm_nlam):
                fpm_lam_um[j] = float(f.readline())
            fpm_lams = [j.strip() for j in f.readlines()] 
            f.close()
            diff = np.abs(fpm_lam_um - lam_um)
            w = np.argmin( diff )
            if diff[w] > 0.0001:
                print("Error in roman_preflight: requested wavelength not within 0.1 nm of nearest available FPM wavelength.")
                print("  requested (um) = " + str(lam_um) + "  closest available (um) = " + str(fpm_lam_um[w]) ) 
                raise Exception(' ')
            fpm_rootname = file_directory + fpm_lams[w]
            (r, header) = proper.prop_fits_read( fpm_rootname+'real.fits', header=True ) 
            i = proper.prop_fits_read( fpm_rootname+'imag.fits' )
            fpm_array = r + i * 1j
            fpm_mask = (r != r[0,0]).astype(int)
            fpm_lam0_m = header['FPMLAM0M']             # FPM reference wavelength
            fpm_sampling_lam0divD = header['FPMDX']     # lam/D sampling @ fpm_lam0_m
        lyot_stop_file = file_directory + 'lyot_rotated.fits'
        field_stop_radius_lam0 = 9.7
        if use_defocus_lens != 0 or use_pupil_lens != 0:
            n = 2048
        else:
            n = 1024
        n_mft = 1400
    elif cor_type == 'zwfs':
        is_zwfs = True 
        file_directory = data_dir + '/zwfs/'         # must have trailing "/"
        lambda0_m = 0.575e-6
        pupil_diam_pix = 309.0
        if use_cvs != 0:
            pupil_file = file_directory + 'pupil_rotated.fits'
        else:
            pupil_file = file_directory + 'pupil.fits'
        if use_fpm != 0:
            # find nearest available FPM wavelength that matches specified wavelength
            lam_um = lambda_m * 1e6
            f = open(file_directory + 'fpm_files.txt')
            fpm_nlam = int(f.readline())
            fpm_lam_um = np.zeros((fpm_nlam), dtype=float)
            for j in range(0, fpm_nlam):
                fpm_lam_um[j] = float(f.readline())
            fpm_lams = [j.strip() for j in f.readlines()]
            f.close()
            diff = np.abs(fpm_lam_um - lam_um)
            w = np.argmin(diff)
            if diff[w] > 0.0001:
                print("Error in roman_preflight: requested wavelength not within 0.1 nm of nearest available ZWFS FPM wavelength.")
                print("  requested (um) = " + str(lam_um) + "  closest available (um) = " + str(fpm_lam_um[w]) ) 
                raise Exception(' ')
            fpm_rootname = file_directory + fpm_lams[w]
            (r, header) = proper.prop_fits_read( fpm_rootname+'real.fits', header=True ) 
            i = proper.prop_fits_read( fpm_rootname+'imag.fits' )
            fpm_array = r + i * 1j
            fpm_mask = (r != r[0, 0]).astype(int)
            fpm_lam0_m = header['FPMLAM0M']  # FPM reference wavelength
            fpm_sampling_lam0divD = header['FPMDX']  # lam/D sampling @ fpm_lam0_m
        if use_defocus_lens != 0 or use_pupil_lens != 0:
            n = 2048
        else:
            n = 1024
        n_mft = 1400
        use_field_stop = False
        use_pupil_mask = False
        use_lyot_stop = False
    elif cor_type == 'spc-spec' or cor_type == 'spc-spec_band2' or cor_type == 'spc-spec_band3':
        is_spc = True
        file_dir = data_dir + '/spc_20200617_spec/' # must have trailing "/"
        if small_spc_grid != 0:
                pupil_diam_pix = 500.0          # Y axis pupil diameter in pixels
                if use_pupil_lens != 0 or use_defocus_lens != 0:
                    n = 2048 
                else:
                    n = 1024
        else: 
                pupil_diam_pix = 1000.0         # Y axis pupil diameter in pixels
                if use_pupil_lens != 0 or use_defocus_lens != 0:
                    n = 4096 
                else:
                    n = 2048
        sgrid = str(int(pupil_diam_pix))
        if use_cvs != 0:
            pupil_file = file_dir + 'pupil_SPC-20200617_' + sgrid + '_rotated.fits'
        else:
            pupil_file = file_dir + 'pupil_SPC-20200617_' + sgrid + '.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200617_' + sgrid + '_rounded9_rotated.fits'
        fpm_sampling_lam0divD = 0.05         # sampling in fpm_lam0_m/D of FPM mask 
        fpm_file = file_dir + 'fpm_0.05lamD.fits'
        if cor_type == 'spc-spec_band2':
            fpm_lam0_m = 0.66e-6
        else:
            fpm_lam0_m = 0.73e-6
        lambda0_m = fpm_lam0_m     # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200617_' + sgrid + '.fits'
        n_mft = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-spec_rotated' or cor_type == 'spc-spec_band2_rotated' or cor_type == 'spc-spec_band3_rotated':
        is_spc = True
        file_dir = data_dir + '/spc_20200628_specrot/'     # must have trailing "/"
        pupil_diam_pix = 1000.0     # Y axis pupil diameter in pixels
        if use_cvs != 0:
            pupil_file = file_dir + 'new_pupil_SPC-20200628_1000_rotated.fits'
        else:
            pupil_file = file_dir + 'new_pupil_SPC-20200628_1000.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200628_1000_rotated.fits'
        fpm_sampling_lam0divD = 0.05         # sampling in fpm_lam0_m/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200628_res20_flip.fits'
        if cor_type == 'spc-spec_band2_rotated':
            fpm_lam0_m = 0.66e-6
        else:
            fpm_lam0_m = 0.73e-6
        lambda0_m = fpm_lam0_m     # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200628_1000_flip.fits'
        if use_defocus_lens != 0 or use_pupil_lens != 0:
            n = 4096
        else:
            n = 2048
        n_mft = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-wide' or cor_type == 'spc-wide_band4' or cor_type == 'spc-wide_band1':
        is_spc = True
        file_dir = data_dir + '/spc_20200610_wfov/' # must have trailing "/"
        if small_spc_grid != 0:
                pupil_diam_pix = 500.0          # Y axis pupil diameter in pixels
                if use_pupil_lens != 0 or use_defocus_lens != 0:
                    n = 2048
                else:
                    n = 1024
        else:
                pupil_diam_pix = 1000.0         # Y axis pupil diameter in pixels
                if use_pupil_lens != 0 or use_defocus_lens != 0:
                    n = 4096
                else:
                    n = 2048
        sgrid = str(int(pupil_diam_pix))
        if use_cvs != 0:
            pupil_file = file_dir + 'pupil_SPC-20200610_' + sgrid + '_rotated.fits'
        else:
            pupil_file = file_dir + 'pupil_SPC-20200610_' + sgrid + '.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200610_' + sgrid + '_rounded9_gray_rotated.fits'
        fpm_sampling_lam0divD = 0.1    # sampling in lambda0/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200610_0.1_lamc_div_D.fits'
        if cor_type == 'spc-wide_band1':
            fpm_lam0_m = 0.575e-6
        else:
            fpm_lam0_m = 0.825e-6
        lambda0_m = fpm_lam0_m        # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200610_' + sgrid + '.fits'
        n_mft = 1400
    elif cor_type == 'spc-mswc':
        is_spc = True
        file_dir = data_dir + '/spc_20200623_mswc/' # must have trailing "/"
        pupil_diam_pix = 982.0  # Y axis pupil diameter in pixels
        if use_cvs != 0:
            pupil_file = file_dir + 'pupil_SPC-20200623_982.fits'
        else:
            pupil_file = file_dir + 'pupil_SPC-20200623_982_rotated.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200623_982_rounded9_gray_rotated.fits'
        fpm_sampling_lam0divD = 0.1    # sampling in lambda0/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200623_0.1_lamc_div_D.fits'
        if cor_type == 'spc-mswc_band1':
            fpm_lam0_m = 0.575e-6
        else:
            fpm_lam0_m = 0.825e-6
        lambda0_m = fpm_lam0_m        # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200623_982_rotated.fits'
        if use_defocus_lens != 0 or use_pupil_lens != 0:
            n = 4096
        else:
            n = 2048
        n_mft = 1400
    elif cor_type == 'none': 
        pupil_diam_pix = 500.0
        lambda0_m = 0.575e-6
        use_fpm = 0
        use_lyot_stop = 0
        use_field_stop = 0
        n = 1024
    else:
        raise Exception('ERROR: Unsupported cor_type: '+cor_type)

    mas_per_lamD = lambda0_m * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000    # mas per lambda0/D

    if 'PASSVALUE' in locals():
        if 'sm_despace_m' in PASSVALUE: sm_despace_m = PASSVALUE['sm_despace_m']
        if 'source_x_offset' in PASSVALUE and 'source_x_offset_mas' in PASSVALUE:
            print("ERROR: can only specify source_x_offset or source_x_offset_mas, not both")
            raise Exception(' ')
        if 'source_y_offset' in PASSVALUE and 'source_y_offset_mas' in PASSVALUE:
            print("ERROR: can only specify source_y_offset or source_y_offset_mas, not both")
            raise Exception(' ')
        if 'source_x_offset' in PASSVALUE: source_x_offset = PASSVALUE['source_x_offset']
        if 'source_y_offset' in PASSVALUE: source_y_offset = PASSVALUE['source_y_offset']
        if 'source_x_offset_mas' in PASSVALUE: source_x_offset = PASSVALUE['source_x_offset_mas'] / mas_per_lamD
        if 'source_y_offset_mas' in PASSVALUE: source_y_offset = PASSVALUE['source_y_offset_mas'] / mas_per_lamD
        if 'cvs_source_z_offset_m' in PASSVALUE: cvs_source_z_offset_m = PASSVALUE['cvs_source_z_offset_m']
        if 'cvs_stop_x_shift_m' in PASSVALUE: cvs_stop_x_shift_m = PASSVALUE['cvs_stop_x_shift_m']
        if 'cvs_stop_y_shift_m' in PASSVALUE: cvs_stop_y_shift_m = PASSVALUE['cvs_stop_y_shift_m']
        if 'cvs_stop_z_shift_m' in PASSVALUE: cvs_stop_z_shift_m = PASSVALUE['cvs_stop_z_shift_m']
        if 'cvs_stop_rotation_deg' in PASSVALUE: cvs_stop_rotation_deg = PASSVALUE['cvs_stop_rotation_deg']
        if 'cvs_jitter_mirror_x_offset' in PASSVALUE and 'cvs_jitter_mirror_x_offset_mas' in PASSVALUE:
            print("ERROR: can only specify cvs_jitter_mirror_x_offset or cvs_jitter_mirror_x_offset_mas, not both")
            raise Exception(' ')
        if 'cvs_jitter_mirror_y_offset' in PASSVALUE and 'cvs_jitter_mirror_y_offset_mas' in PASSVALUE:
            print("ERROR: can only specify cvs_jitter_mirror_y_offset or cvs_jitter_mirror_y_offset_mas, not both")
            raise Exception(' ')
        if 'cvs_jitter_mirror_x_offset' in PASSVALUE: cvs_jitter_mirror_x_offset = PASSVALUE['cvs_jitter_mirror_x_offset']
        if 'cvs_jitter_mirror_y_offset' in PASSVALUE: cvs_jitter_mirror_y_offset = PASSVALUE['cvs_jitter_mirror_y_offset']
        if 'cvs_jitter_mirror_x_offset_mas' in PASSVALUE: cvs_jitter_mirror_x_offset_mas = PASSVALUE['cvs_jitter_mirror_x_offset_mas'] / mas_per_lamD
        if 'cvs_jitter_mirror_y_offset_mas' in PASSVALUE: cvs_jitter_mirror_y_offset_mas = PASSVALUE['cvs_jitter_mirror_y_offset_mas'] / mas_per_lamD
        if 'use_aperture' in PASSVALUE: use_aperture = PASSVALUE['use_aperture']
        if 'use_errors' in PASSVALUE: use_errors = PASSVALUE['use_errors']
        if 'use_pupil_defocus' in PASSVALUE: use_pupil_defocus = PASSVALUE['use_pupil_defocus']
        if 'polaxis' in PASSVALUE: polaxis = PASSVALUE['polaxis'] 
        if 'zindex' in PASSVALUE: zindex = np.array( PASSVALUE['zindex'] )
        if 'zval_m' in PASSVALUE: zval_m = np.array( PASSVALUE['zval_m'] )
        if 'end_at_fsm' in PASSVALUE: end_at_fsm = PASSVALUE['end_at_fsm']
        if 'cgi_x_shift_pupdiam' in PASSVALUE and 'cgi_x_shift_m' in PASSVALUE:
            print("ERROR: can only specify cgi_x_shift_pupdiam or cgi_x_shift_m, not both")
            raise Exception(' ')
        if 'cgi_y_shift_pupdiam' in PASSVALUE and 'cgi_y_shift_m' in PASSVALUE:
            print("ERROR: can only specify cgi_y_shift_pupdiam or cgi_y_shift_m, not both")
            raise Exception(' ')
        if 'cgi_x_shift_pupdiam' in PASSVALUE: cgi_x_shift_pupdiam = PASSVALUE['cgi_x_shift_pupdiam']
        if 'cgi_y_shift_pupdiam' in PASSVALUE: cgi_y_shift_pupdiam = PASSVALUE['cgi_y_shift_pupdiam']
        if 'cgi_x_shift_m' in PASSVALUE: cgi_x_shift_m = PASSVALUE['cgi_x_shift_m']
        if 'cgi_y_shift_m' in PASSVALUE: cgi_y_shift_m = PASSVALUE['cgi_y_shift_m']
        if 'fsm_x_offset' in PASSVALUE and 'fsm_x_offset_mas' in PASSVALUE:
            print("ERROR: can only specify fsm_x_offset or fsm_x_offset_mas, not both")
            raise Exception(' ')
        if 'fsm_y_offset' in PASSVALUE and 'fsm_y_offset_mas' in PASSVALUE:
            print("ERROR: can only specify fsm_y_offset or fsm_y_offset_mas, not both")
            raise Exception(' ')
        if 'fsm_x_offset' in PASSVALUE: fsm_x_offset = PASSVALUE['fsm_x_offset']
        if 'fsm_y_offset' in PASSVALUE: fsm_y_offset = PASSVALUE['fsm_y_offset']
        if 'fsm_x_offset_mas' in PASSVALUE: fsm_x_offset = PASSVALUE['fsm_x_offset_mas'] / mas_per_lamD
        if 'fsm_y_offset_mas' in PASSVALUE: fsm_y_offset = PASSVALUE['fsm_y_offset_mas'] / mas_per_lamD
        if 'fcm_z_shift_m' in PASSVALUE: fcm_z_shift_m = PASSVALUE['fcm_z_shift_m']
        if 'dm_sampling_m' in PASSVALUE: dm_sampling_m = PASSVALUE['dm_sampling_m']
        if 'dm_temp_c' in PASSVALUE: dm_temp_c = PASSVALUE['dm_temp_c']
        if 'dm_version' in PASSVALUE: dm_version = PASSVALUE['dm_version']
        if 'dm_v_quant' in PASSVALUE: dm_v_quant = PASSVALUE['dm_v_quant']
        if 'use_dm1' in PASSVALUE: use_dm1 = PASSVALUE['use_dm1'] 
        if 'dm1_v' in PASSVALUE: dm1_v = PASSVALUE['dm1_v']
        if 'dm1_xc_act' in PASSVALUE: dm1_xc_act = PASSVALUE['dm1_xc_act']
        if 'dm1_yc_act' in PASSVALUE: dm1_yc_act = PASSVALUE['dm1_yc_act']
        if 'dm1_xtilt_deg' in PASSVALUE: dm1_xtilt_deg = PASSVALUE['dm1_xtilt_deg']
        if 'dm1_ytilt_deg' in PASSVALUE: dm1_ytilt_deg = PASSVALUE['dm1_ytilt_deg']
        if 'dm1_ztilt_deg' in PASSVALUE: dm1_ztilt_deg = PASSVALUE['dm1_ztilt_deg']
        if 'use_dm2' in PASSVALUE: use_dm2 = PASSVALUE['use_dm2']
        if 'dm2_v' in PASSVALUE: dm2_v = PASSVALUE['dm2_v']
        if 'dm2_xc_act' in PASSVALUE: dm2_xc_act = PASSVALUE['dm2_xc_act']
        if 'dm2_yc_act' in PASSVALUE: dm2_yc_act = PASSVALUE['dm2_yc_act']
        if 'dm2_xtilt_deg' in PASSVALUE: dm2_xtilt_deg = PASSVALUE['dm2_xtilt_deg']
        if 'dm2_ytilt_deg' in PASSVALUE: dm2_ytilt_deg = PASSVALUE['dm2_ytilt_deg']
        if 'dm2_ztilt_deg' in PASSVALUE: dm2_ztilt_deg = PASSVALUE['dm2_ztilt_deg']
        if 'spam_x_shift_pupdiam' in PASSVALUE and 'spam_x_shift_m' in PASSVALUE:
            print("ERROR: can only specify spam_x_shift_pupdiam or spam_x_shift_m, not both")
            raise Exception(' ')
        if 'spam_y_shift_pupdiam' in PASSVALUE and 'spam_y_shift_m' in PASSVALUE:
            print("ERROR: can only specify spam_y_shift_pupdiam or spam_y_shift_m, not both")
            raise Exception(' ')
        if 'spam_x_shift_pupdiam' in PASSVALUE: spam_x_shift_pupdiam = PASSVALUE['spam_x_shift_pupdiam']
        if 'spam_y_shift_pupdiam' in PASSVALUE: spam_y_shift_pupdiam = PASSVALUE['spam_y_shift_pupdiam']
        if 'spam_x_shift_m' in PASSVALUE: spam_x_shift_m = PASSVALUE['spam_x_shift_m']
        if 'spam_y_shift_m' in PASSVALUE: spam_y_shift_m = PASSVALUE['spam_y_shift_m']
        if 'use_pupil_mask' in PASSVALUE: use_pupil_mask = PASSVALUE['use_pupil_mask']
        if 'mask_x_shift_pupdiam' in PASSVALUE and 'mask_x_shift_m' in PASSVALUE:
            print("ERROR: can only specify mask_x_shift_pupdiam or mask_x_shift_m, not both")
            raise Exception(' ')
        if 'mask_y_shift_pupdiam' in PASSVALUE and 'mask_y_shift_m' in PASSVALUE:
            print("ERROR: can only specify mask_y_shift_pupdiam or mask_y_shift_m, not both")
            raise Exception(' ')
        if 'mask_x_shift_pupdiam' in PASSVALUE: mask_x_shift_pupdiam = PASSVALUE['mask_x_shift_pupdiam']
        if 'mask_y_shift_pupdiam' in PASSVALUE: mask_y_shift_pupdiam = PASSVALUE['mask_y_shift_pupdiam']
        if 'mask_x_shift_m' in PASSVALUE: mask_x_shift_m = PASSVALUE['mask_x_shift_m']
        if 'mask_y_shift_m' in PASSVALUE: mask_y_shift_m = PASSVALUE['mask_y_shift_m']
        if 'mask_rotation_deg' in PASSVALUE: mask_rotation_deg = PASSVALUE['mask_rotation_deg']
        if 'fpm_x_offset' in PASSVALUE and 'fpm_x_offset_m' in PASSVALUE:
            print("ERROR: can only specify fpm_x_offset or fpm_x_offset_m, not both")
            raise Exception(' ')
        if 'fpm_y_offset' in PASSVALUE and 'fpm_y_offset_m' in PASSVALUE:
            print("ERROR: can only specify fpm_y_offset or fpm_y_offset_m, not both")
            raise Exception(' ')
        if 'fpm_x_offset' in PASSVALUE: fpm_x_offset = PASSVALUE['fpm_x_offset']
        if 'fpm_y_offset' in PASSVALUE: fpm_y_offset = PASSVALUE['fpm_y_offset']
        if 'fpm_x_offset_m' in PASSVALUE: fpm_x_offset_m = PASSVALUE['fpm_x_offset_m']
        if 'fpm_y_offset_m' in PASSVALUE: fpm_y_offset_m = PASSVALUE['fpm_y_offset_m']
        if 'fpm_z_shift_m' in PASSVALUE: fpm_z_shift_m = PASSVALUE['fpm_z_shift_m']
        if 'pinhole_diam_m' in PASSVALUE: pinhole_diam_m = PASSVALUE['pinhole_diam_m']
        if 'end_at_fpm_exit_pupil' in PASSVALUE: end_at_fpm_exit_pupil = PASSVALUE['end_at_fpm_exit_pupil']
        if 'use_lyot_stop' in PASSVALUE: use_lyot_stop = PASSVALUE['use_lyot_stop']
        if 'lyot_x_shift_pupdiam' in PASSVALUE and 'lyot_x_shift_m' in PASSVALUE:
            print("ERROR: can only specify lyot_x_shift_pupdiam or lyot_x_shift_m, not both")
            raise Exception(' ')
        if 'lyot_y_shift_pupdiam' in PASSVALUE and 'lyot_y_shift_m' in PASSVALUE:
            print("ERROR: can only specify lyot_y_shift_pupdiam or lyot_y_shift_m, not both")
            raise Exception(' ')
        if 'lyot_x_shift_pupdiam' in PASSVALUE: lyot_x_shift_pupdiam = PASSVALUE['lyot_x_shift_pupdiam']
        if 'lyot_y_shift_pupdiam' in PASSVALUE: lyot_y_shift_pupdiam = PASSVALUE['lyot_y_shift_pupdiam']
        if 'lyot_x_shift_m' in PASSVALUE: lyot_x_shift_m = PASSVALUE['lyot_x_shift_m']
        if 'lyot_y_shift_m' in PASSVALUE: lyot_y_shift_m = PASSVALUE['lyot_y_shift_m']
        if 'lyot_rotation_deg' in PASSVALUE: lyot_rotation_deg = PASSVALUE['lyot_rotation_deg']
        if 'use_field_stop' in PASSVALUE: use_field_stop = PASSVALUE['use_field_stop']
        if 'field_stop_radius_lam0' in PASSVALUE: field_stop_radius_lam0 = PASSVALUE['field_stop_radius_lam0']
        if 'field_stop_x_offset' in PASSVALUE and 'field_stop_x_offset_m' in PASSVALUE:
            print("ERROR: can only specify field_stop_x_offset or field_stop_x_offset_m, not both")
            raise Exception(' ')
        if 'field_stop_y_offset' in PASSVALUE and 'field_stop_y_offset_m' in PASSVALUE:
            print("ERROR: can only specify field_stop_y_offset or field_stop_y_offset_m, not both")
            raise Exception(' ')
        if 'field_stop_x_offset' in PASSVALUE: field_stop_x_offset = PASSVALUE['field_stop_x_offset']
        if 'field_stop_y_offset' in PASSVALUE: field_stop_y_offset = PASSVALUE['field_stop_y_offset']
        if 'field_stop_x_offset_m' in PASSVALUE: field_stop_x_offset_m = PASSVALUE['field_stop_x_offset_m']
        if 'field_stop_y_offset_m' in PASSVALUE: field_stop_y_offset_m = PASSVALUE['field_stop_y_offset_m']
        if 'end_at_exit_pupil' in PASSVALUE: end_at_exit_pupil = PASSVALUE['end_at_exit_pupil']
        if 'final_sampling_lam0' in PASSVALUE and 'final_sampling_m' in PASSVALUE:
            print("ERROR: can only specify final_sampling_lam0 or final_sampling_m, not both")
            raise Exception(' ')
        if 'final_sampling_lam0' in PASSVALUE: final_sampling_lam0 = PASSVALUE['final_sampling_lam0']
        if 'final_sampling_m' in PASSVALUE: final_sampling_m = PASSVALUE['final_sampling_m']
        if 'output_dim' in PASSVALUE: output_dim = PASSVALUE['output_dim']
        if 'excam_despace_m' in PASSVALUE: excam_despace_m = PASSVALUE['excam_despace_m']
        if 'image_x_offset_m' in PASSVALUE: image_x_offset_m = PASSVALUE['image_x_offset_m']
        if 'image_y_offset_m' in PASSVALUE: image_y_offset_m = PASSVALUE['image_y_offset_m']
        if 'save_ref_radius' in PASSVALUE: save_ref_radius = PASSVALUE['save_ref_radius']

    if use_cvs == 0:
        diam = 2.363114
        fl_pri = 2.838279206904720
        d_pri_sec = 2.285150508110035 + sm_despace_m
        fl_sec = -0.654200796568004
        diam_sec = 0.58166
        d_sec_pomafold = 2.993753469304728 + sm_despace_m
        diam_pomafold = 0.09
        d_pomafold_m3 = 1.680935841598811
        fl_m3 = 0.430216463069001
        diam_m3 = 0.2
        d_m3_pupil = 0.469156807765176
        d_m3_m4 = 0.943514749358944
        fl_m4 = 0.116239114833590
        diam_m4 = 0.07
        d_m4_m5 = 0.429145636743193
        fl_m5 = 0.198821518772608
        d_m5_pupil = 0.716529242927776
        diam_m5 = 0.07
        d_m5_ttfold = 0.351125431220770
        diam_ttfold = 0.06
        d_ttfold_fsm = d_m5_pupil - d_m5_ttfold 
        if use_pupil_defocus:
            d_ttfold_fsm = d_ttfold_fsm + 0.033609  # 33.6 mm to put pupil 6 mm from SPC mask
    else:
        diam = 0.040234882
        d_cvs_source_fold1 = 0.750
        d_cvs_fold1_oap1 = 0.789886718750000
        fl_cvs_oap1 = d_cvs_source_fold1 + d_cvs_fold1_oap1
        d_cvs_oap1_stop = 0.344113281250000 + cvs_stop_z_shift_m
        d_cvs_stop_oap2 = 0.295980957031250 - cvs_stop_z_shift_m
        d_cvs_oap2_focus = 0.30881902887
        fl_cvs_oap2 = d_cvs_oap2_focus
        d_cvs_focus_oap3 = 0.15440951444
        fl_cvs_oap3 = d_cvs_focus_oap3
        d_cvs_oap3_jitterfold = 0.157618895173073
        d_cvs_jitterfold_oap4 = 0.085733268737793
        d_cvs_oap4_focus = 0.15666673228
        fl_cvs_oap4 = d_cvs_oap4_focus
        d_cvs_focus_oap5 = 0.31333346457
        fl_cvs_oap5 = d_cvs_focus_oap5
        d_cvs_oap5_fold2 = 0.136466537475586
        d_cvs_fold2_fold3 = 0.105
        d_cvs_fold3_fsm = 0.35560079956054691

    diam_fsm = 0.0508
    d_fsm_oap1 = 0.354826767220001
    fl_oap1 = 0.503331895563883
    diam_oap1 = 0.060
    d_oap1_fcm = 0.768029932093727 + fcm_z_shift_m
    diam_fcm = 0.035
    d_fcm_oap2 = 0.314507535543064 + fcm_z_shift_m
    fl_oap2 = 0.579205571254990
    diam_oap2 = 0.060
    d_oap2_dm1 = 0.775857408587825
    d_dm1_dm2 = 1.0
    diam_dm_aperture = 0.054
    d_dm2_oap3 = 0.394833855161549
    fl_oap3 = 1.217276467668519
    diam_oap3 = 0.06
    d_oap3_fold3 = 0.505329955078121
    diam_fold3 = 0.06
    d_fold3_oap4 = 1.158897671642761
    fl_oap4 = 0.446951159052363
    diam_oap4 = 0.06
    d_oap4_pupilmask = 0.423013568764728
    diam_beam_at_pupil_mask = 0.017000141
    d_pupilmask_oap5 = 0.408810704327559
    fl_oap5 = 0.548189354706822
    diam_oap5 = 0.06
    d_oap5_fpm = fl_oap5                    # to front of FPM 
    fpm_thickness = 0.006363747896388863    # account for FPM thickness (inclination included)
    fpm_index = glass_index('SILICA',lambda_m,data_dir)
    m_per_lamdivD_575nm_at_fpm = 1.8541536e-05    # nominal sampling per lam/D at lam=575 nm
    d_fpm_oap6 = fpm_thickness / fpm_index + 0.543766629917668     # from front of FPM
    fl_oap6 = d_fpm_oap6
    diam_oap6 = 0.054
    d_oap6_lyotstop = 0.687476361491529
    diam_beam_at_lyot_stop = 0.016998272
    d_oap6_exitpupil = d_oap6_lyotstop - 6e-3
    d_lyotstop_oap7 = 0.401748561745987
    fl_oap7 = 0.708251420923810
    diam_oap7 = 0.054
    d_oap7_fieldstop = fl_oap7 
    m_per_lamdivD_575nm_at_field_stop = 2.3957964e-05     # nominal sampling per lam/D at lam=575 nm
    d_fieldstop_oap8 = 0.210985170345932 * 0.997651
    fl_oap8 = d_fieldstop_oap8
    diam_oap8 = 0.026
    d_oap8_pupil = 0.237561587674008
    d_pupil_filter = 0.130
    d_oap8_filter = d_oap8_pupil + d_pupil_filter   # to front of filter
    d_cfambaffle_filter = 0.0145
    d_oap8_cfambaffle = d_oap8_filter - (1e-3 + 4e-3) - d_cfambaffle_filter
    diam_cfambaffle = 0.0071
    diam_filter = 0.009
    filter_thickness = 0.004016105782012525      # account for filter thickness (inclination included)
    filter_index = glass_index('SILICA',lambda_m,data_dir)
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095  # from front of filter
    diam_lens = 0.0104
    d_lens_fold4 = 0.202226
    if use_defocus_lens != 0:
        d_lens_fold4 = d_lens_fold4 + 0.001 
    diam_fold4 = 0.036
    d_fold4_image = 0.050206330646919 + excam_despace_m

    if use_dm1 != 0 or use_dm2 != 0:
        dm_struct = rp.load_cgi_dm_files( dm_files_dir=dm_dir, version=dm_version, temp_c=dm_temp_c )

    wavefront = proper.prop_begin( diam, lambda_m, n, float(pupil_diam_pix)/n )
    if source_x_offset != 0 or source_y_offset != 0:
        # compute tilted wavefront to offset source by xoffset,yoffset lambda0_m/D
        xtilt_lam = source_y_offset * lambda0_m / lambda_m
        ytilt_lam = source_x_offset * lambda0_m / lambda_m
        x = np.tile( (np.arange(n)-n//2)/(pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
        x = 0
        y = 0

    if use_cvs == 1:
        # create diverging beam by focusing from a dummy lens and going through focus (where a pinhole would
        # be in reality), and then onto the OTA pupil mask
        proper.prop_circular_aperture( wavefront, 1.3, NORM=True )
        proper.prop_propagate( wavefront, fl_cvs_oap1, ' ' )
        proper.prop_lens( wavefront, fl_cvs_oap1 )
        #proper.prop_propagate( wavefront, fl_cvs_oap1, 'Source' )  # pinhole would go here
        proper.prop_propagate( wavefront, cvs_source_z_offset_m+fl_cvs_oap1+d_cvs_source_fold1+d_cvs_fold1_oap1, 'CVS OAP1' )
        proper.prop_lens( wavefront, fl_cvs_oap1 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'cvs_oap1_measured_phase_error_v1.0.fits', WAVEFRONT=True )
        proper.prop_propagate( wavefront, d_cvs_oap1_stop, 'CVS STOP' )

    if cor_type != 'none':
        if np.isscalar(pupil_array):
            pupil = proper.prop_fits_read( pupil_file )
        else:
            pupil = pupil_array
        pupil = trim(pupil, n)
        if use_cvs == 1 and (cvs_stop_x_shift_m != 0 or cvs_stop_y_shift_m != 0 or cvs_stop_rotation_deg != 0 or cvs_source_z_offset_m != 0):
            # rotate and/or shift OTA pupil mask; if source Z offset is non-zero, the change match new beam sampling
            xshift_pix = -cvs_stop_x_shift_m / proper.prop_get_sampling(wavefront)
            yshift_pix = -cvs_stop_y_shift_m / proper.prop_get_sampling(wavefront)
            if cvs_source_z_offset_m != 0:
                mag = 2 * proper.prop_get_beamradius(wavefront) / diam_beam_at_pupil_mask
            else: 
                mag = 1
            pupil = transform_image( pupil, xshift_pix, yshift_pix, cvs_stop_rotation_deg, mag )
        proper.prop_multiply( wavefront, pupil )
    else:
        proper.prop_circular_aperture( wavefront, 1.0, NORM=True )
    proper.prop_define_entrance( wavefront )
    if zindex[0] != 0: proper.prop_zernikes( wavefront, zindex, zval_m )

    if use_cvs == 0:
        # Roman OTA, starting at primary and ending at FSM
        proper.prop_lens( wavefront, fl_pri )
        if polaxis != 0: polmap( wavefront, polfile, pupil_diam_pix, polaxis )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True )

        proper.prop_propagate( wavefront, d_pri_sec, 'secondary' )
        proper.prop_lens( wavefront, fl_sec )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_sec/2.0 )  

        proper.prop_propagate( wavefront, d_sec_pomafold, 'POMA FOLD' )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_POMAFOLD_measured_phase_error_V2.0.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_pomafold/2.0 ) 

        proper.prop_propagate( wavefront, d_pomafold_m3, 'M3' )
        proper.prop_lens( wavefront, fl_m3 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_M3_measured_phase_error_V2.0.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m3/2.0 ) 

        proper.prop_propagate( wavefront, d_m3_m4, 'M4' )
        proper.prop_lens( wavefront, fl_m4 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_M4_measured_phase_error_V2.0.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m4/2.0 ) 
 
        proper.prop_propagate( wavefront, d_m4_m5, 'M5' )
        proper.prop_lens( wavefront, fl_m5 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_M5_measured_phase_error_V2.0.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m5/2.0 ) 
 
        proper.prop_propagate( wavefront, d_m5_ttfold, 'TT FOLD' )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits', WAVEFRONT=True )
        if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_ttfold/2.0 )

        proper.prop_propagate( wavefront, d_ttfold_fsm, 'FSM' )
        # if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_LOWORDER_phase_error_V2.0.fits', WAVEFRONT=True )
    else:
        # CGI CVS, starting at OTA pupil mask and ending at FSM
        proper.prop_propagate( wavefront, d_cvs_stop_oap2, 'CVS OAP2' )
        proper.prop_lens( wavefront, fl_cvs_oap2 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'cvs_oap2_measured_phase_error_v1.0.fits', WAVEFRONT=True )

        proper.prop_propagate( wavefront, d_cvs_oap2_focus+d_cvs_focus_oap3, 'CVS OAP3' )
        proper.prop_lens( wavefront, fl_cvs_oap3 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'cvs_oap3_measured_phase_error_v1.0.fits', WAVEFRONT=True )

        proper.prop_propagate( wavefront, d_cvs_oap3_jitterfold, 'CVS JITTER_FOLD' )
        if cvs_jitter_mirror_x_offset != 0 or cvs_jitter_mirror_y_offset != 0:
            # compute tilted wavefront to offset source by cvs_jitter_mirror_x_offset,cvs_jitter_mirror_y_offset lambda0_m/D
            xtilt_lam = cvs_jitter_mirror_y_offset * lambda0_m / lambda_m
            ytilt_lam = cvs_jitter_mirror_x_offset * lambda0_m / lambda_m
            x = np.tile( (np.arange(n)-n//2)/(pupil_diam_pix/2.0), (n,1) )
            y = np.transpose(x)
            proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
            x = 0
            y = 0
 
        proper.prop_propagate( wavefront, d_cvs_jitterfold_oap4, 'CVS OAP4' )
        proper.prop_lens( wavefront, fl_cvs_oap4 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'cvs_oap4_measured_phase_error_v1.0.fits', WAVEFRONT=True )

        proper.prop_propagate( wavefront, d_cvs_oap4_focus+d_cvs_focus_oap5, 'CVS OAP5' )
        proper.prop_lens( wavefront, fl_cvs_oap5 )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'cvs_oap5_measured_phase_error_v1.0.fits', WAVEFRONT=True )

        #prop_propagate, wavefront, d_cvs_oap5_fold2, 'FOLD2'
        #prop_propagate, wavefront, d_cvs_fold2_fold3, 'FOLD3'
        #prop_propagate, wavefront, d_cvs_fold3_fsm, 'FSM'

        proper.prop_propagate( wavefront, d_cvs_oap5_fold2+d_cvs_fold2_fold3+d_cvs_fold3_fsm, 'FSM' )

    if end_at_fsm == 1:
        (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )
        wavefront = trim(wavefront, n)
        return wavefront, sampling_m
    if cgi_x_shift_pupdiam != 0 or cgi_y_shift_pupdiam != 0 or cgi_x_shift_m != 0 or cgi_y_shift_m != 0:    # bulk coronagraph pupil shear
        # FFT the field, apply a tilt, FFT back
        if cgi_x_shift_pupdiam != 0 or cgi_y_shift_pupdiam != 0:
            # offsets are normalized to pupil diameter
            xshift_pix = cgi_x_shift_pupdiam * pupil_diam_pix
            yshift_pix = cgi_y_shift_pupdiam * pupil_diam_pix
        else:
            # offsets are meters
            xshift_pix = cgi_x_shift_m / proper.prop_get_sampling(wavefront)
            yshift_pix = cgi_y_shift_m / proper.prop_get_sampling(wavefront)
        wavefront0 = proper.prop_get_wavefront(wavefront)
        wavefront0 = shift_image( wavefront0, xshift_pix, yshift_pix )
        wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
        wavefront0 = 0
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_FSM_FLIGHT_coated_measured_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fsm/2.0 )
    if ( fsm_x_offset != 0.0 or fsm_y_offset != 0.0 ):
        # compute tilted wavefront to offset source by fsm_x_offset,fsm_y_offset lambda0_m/D
        xtilt_lam = -fsm_y_offset * lambda0_m / lambda_m
        ytilt_lam = -fsm_x_offset * lambda0_m / lambda_m
        x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
        x = 0
        y = 0

    proper.prop_propagate( wavefront, d_fsm_oap1, 'OAP1' )
    proper.prop_lens( wavefront, fl_oap1 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP1_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap1/2.0 )  

    proper.prop_propagate( wavefront, d_oap1_fcm, 'FCM' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fcm/2.0 )

    proper.prop_propagate( wavefront, d_fcm_oap2, 'OAP2' )
    proper.prop_lens( wavefront, fl_oap2 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP2_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap2/2.0 )  

    proper.prop_propagate( wavefront, d_oap2_dm1, 'DM1' )
    if use_dm1 != 0: 
        rp.cgi_dm( wavefront, dm_struct, 1, dm1_v, dm_sampling_m=dm_sampling_m, dm_v_quant=dm_v_quant,
                      dm_xc_act=dm1_xc_act, dm_yc_act=dm1_yc_act, 
                      dm_xtilt_deg=dm1_xtilt_deg, dm_ytilt_deg=dm1_ytilt_deg, dm_ztilt_deg=dm1_ztilt_deg )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_dm_aperture/2.0 )  

    proper.prop_propagate( wavefront, d_dm1_dm2, 'DM2' )
    if use_dm2 == 1: 
        rp.cgi_dm( wavefront, dm_struct, 2, dm2_v, dm_sampling_m=dm_sampling_m, dm_v_quant=dm_v_quant,
                      dm_xc_act=dm2_xc_act, dm_yc_act=dm2_yc_act, 
                      dm_xtilt_deg=dm2_xtilt_deg, dm_ytilt_deg=dm2_ytilt_deg, dm_ztilt_deg=dm2_ztilt_deg )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_dm_aperture/2.0 )  

    proper.prop_propagate( wavefront, d_dm2_oap3, 'OAP3' )
    proper.prop_lens( wavefront, fl_oap3 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP3_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap3/2.0 ) 

    proper.prop_propagate( wavefront, d_oap3_fold3, 'FOLD_3' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fold3/2.0 )

    proper.prop_propagate( wavefront, d_fold3_oap4, 'OAP4' )
    proper.prop_lens( wavefront, fl_oap4 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP4_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap4/2.0 )

    proper.prop_propagate( wavefront, d_oap4_pupilmask, 'PUPIL_MASK' )    # flat/reflective shaped pupil 
    if (is_spc == True and use_pupil_mask != 0) or np.isscalar(pupil_mask_array) == False:
        if np.isscalar(pupil_mask_array): 
            pupil_mask = proper.prop_fits_read( pupil_mask_file )
            pupil_mask = trim( pupil_mask, n )
        else:
            pupil_mask = trim( pupil_mask_array, n )

        if sm_despace_m != 0 or fcm_z_shift_m != 0 or (use_cvs == 1 and cvs_source_z_offset_m != 0):
            # resize SPC pupil mask if something moved that would change wavefront sampling 
            mag = 2 * proper.prop_get_beamradius(wavefront) / diam_beam_at_pupil_mask
        else:
            mag = 1

        # shift/rotate/magnify SPC pupil mask
        if mask_x_shift_pupdiam != 0 or mask_y_shift_pupdiam != 0 or mask_x_shift_m != 0 or mask_y_shift_m != 0 or mask_rotation_deg != 0 or mag != 1:
            if mask_x_shift_pupdiam != 0 or mask_y_shift_pupdiam != 0:
                # offsets are normalized to pupil diameter
                xshift_pix = mask_x_shift_pupdiam * diam_beam_at_pupil_mask / proper.prop_get_sampling(wavefront)
                yshift_pix = mask_y_shift_pupdiam * diam_beam_at_pupil_mask / proper.prop_get_sampling(wavefront)
            else:
                xshift_pix = mask_x_shift_m / proper.prop_get_sampling(wavefront)
                yshift_pix = mask_y_shift_m / proper.prop_get_sampling(wavefront)
            pupil_mask = transform_image( pupil_mask, xshift_pix, yshift_pix, mask_rotation_deg, mag )
        proper.prop_multiply( wavefront, pupil_mask )
        pupil_mask = 0
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_PUPILMASK_phase_error_V1.0.fits', WAVEFRONT=True )
        # bulk beam shear at SPAM
        if spam_x_shift_pupdiam != 0 or spam_y_shift_pupdiam != 0 or spam_x_shift_m != 0 or spam_y_shift_m != 0:
            if spam_x_shift_pupdiam != 0 or spam_y_shift_pupdiam != 0:
                # offsets are normalized to pupil diameter
                xshift_pix = spam_x_shift_pupdiam * pupil_diam_pix
                yshift_pix = spam_y_shift_pupdiam * pupil_diam_pix
            else:
                xshift_pix = spam_x_shift_m / proper.prop_get_sampling(wavefront)
                yshift_pix = spam_y_shift_m / proper.prop_get_sampling(wavefront)
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = shift_image( wavefront0, xshift_pix, yshift_pix )
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
    else:
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_PUPILFOLD_phase_error_V1.0.fits', WAVEFRONT=True )

    proper.prop_propagate( wavefront, d_pupilmask_oap5, 'OAP5' )
    proper.prop_lens( wavefront, fl_oap5 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP5_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap5/2.0 )  

    proper.prop_propagate( wavefront, d_oap5_fpm+fpm_z_shift_m, 'FPM', TO_PLANE=True )
    if use_fpm == 1:
        if fpm_x_offset != 0 or fpm_y_offset != 0 or fpm_x_offset_m != 0 or fpm_y_offset_m != 0:
            # To shift FPM, apply FPM, shift field back
            m_per_lam0divD = m_per_lamdivD_575nm_at_fpm * lambda0_m / 575e-9        # meters per lam/D at reference wavelength for nominal design 
            if fpm_x_offset != 0 or fpm_y_offset != 0:
                # shifts are specified in lambda0/D
                xshift_pix = -fpm_x_offset * m_per_lam0divD / proper.prop_get_sampling(wavefront)
                yshift_pix = -fpm_y_offset * m_per_lam0divD / proper.prop_get_sampling(wavefront)
            else:
                xshift_pix = -fpm_x_offset_m / proper.prop_get_sampling(wavefront)
                yshift_pix = -fpm_y_offset_m / proper.prop_get_sampling(wavefront)
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = shift_image( wavefront0, xshift_pix, yshift_pix )
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0

        m_per_lamdivD = m_per_lamdivD_575nm_at_fpm * lambda_m / 575e-9  # meters per lam/D at current wavelength for nominal design 
        if is_hlc == True or is_zwfs == True:
            # use MFTs to use super-sampled FPM
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, 1 )              # to virtual pupil
            wavefront0 *= fpm_array[0,0]                    # apply amplitude & phase from FPM clear area
            nfpm = fpm_array.shape[0]
            fpm_sampling_lamdivD = fpm_sampling_lam0divD * fpm_lam0_m / lambda_m    # FPM sampling at current wavelength in lambda_m/D
            fpm_sampling_m = fpm_sampling_lamdivD * m_per_lamdivD
            sampling = fpm_sampling_m * (pupil_diam_pix / n) / proper.prop_get_sampling(wavefront)
            wavefront_fpm = mft2(wavefront0, sampling, pupil_diam_pix, nfpm, -1)   # MFT to highly-sampled focal plane
            wavefront_fpm *= fpm_mask * (fpm_array - 1)      # subtract field inside FPM region, add in FPM-multiplied region
            wavefront_fpm = mft2(wavefront_fpm, sampling, pupil_diam_pix, n, +1)        # MFT back to virtual pupil
            wavefront0 += wavefront_fpm
            wavefront_fpm = 0
            wavefront0 = ffts( wavefront0, -1 )     # back to normally-sampled focal plane to continue propagation
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
        elif is_spc == True:
            # super-sample FPM
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, 1 )                # to virtual pupil
            wavefront0 = trim(wavefront0, n_mft)
            if np.isscalar(fpm_array):
                fpm_array = proper.prop_fits_read( fpm_file )
                fpm_sampling_lamdivD = fpm_sampling_lam0divD * fpm_lam0_m / lambda_m     # FPM sampling at current wavelength in lam/D
            else:
                m_per_lamdivD = proper.prop_get_sampling(wavefront) / (pupil_diam_pix / n)
                fpm_sampling_lamdivD = fpm_array_sampling_m / m_per_lamdivD
            nfpm = fpm_array.shape[1]
            fpm_sampling_m = fpm_sampling_lamdivD * m_per_lamdivD
            sampling = fpm_sampling_m * (pupil_diam_pix / n) / proper.prop_get_sampling(wavefront)
            wavefront0 = mft2(wavefront0, sampling, pupil_diam_pix, nfpm, -1)   # MFT to highly-sampled focal plane
            wavefront0 *= fpm_array
            wavefront0 = mft2(wavefront0, sampling, pupil_diam_pix, n, +1)  # MFT to virtual pupil 
            wavefront0 = ffts( wavefront0, -1 )    # back to normally-sampled focal plane
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
 
        if fpm_x_offset != 0 or fpm_y_offset != 0 or fpm_x_offset_m != 0 or fpm_y_offset_m != 0:
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = shift_image( wavefront0, -xshift_pix, -yshift_pix )
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0

    if pinhole_diam_m != 0:
        # "pinhole_diam_m" is pinhole diameter in meters
        dx_m = proper.prop_get_sampling(wavefront)
        dx_pinhole_diam_m = pinhole_diam_m / 101.0        # 101 samples across pinhole
        n_out = 105
        m_per_lamD = dx_m * n / float(pupil_diam_pix)        # current focal plane sampling in lambda_m/D
        dx_pinhole_lamD = dx_pinhole_diam_m / m_per_lamD    # pinhole sampling in lambda_m/D
        n_in = int(round(pupil_diam_pix * 1.2))
        wavefront0 = proper.prop_get_wavefront(wavefront)
        wavefront0 = ffts( wavefront0, +1 )            # to virtual pupil
        wavefront0 = trim(wavefront0, n_in)
        m = dx_pinhole_lamD * n_in * float(n_out) / pupil_diam_pix
        wavefront0 = mft2( wavefront0, dx_pinhole_lamD, pupil_diam_pix, n_out, -1)        # MFT to highly-sampled focal plane
        p = (radius(n_out)*dx_pinhole_diam_m) <= (pinhole_diam_m/2.0)
        p = p.astype(int)
        wavefront0 *= p
        p = 0
        wavefront0 = mft2( wavefront0, dx_pinhole_lamD, pupil_diam_pix, n, +1)            # MFT back to virtual pupil
        wavefront0 = ffts( wavefront0, -1 )            # back to normally-sampled focal plane
        wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
        wavefront0 = 0

    proper.prop_propagate( wavefront, d_fpm_oap6-fpm_z_shift_m, 'OAP6' )
    proper.prop_lens( wavefront, fl_oap6 )
    if use_errors != 0 and end_at_fpm_exit_pupil == 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP6_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0 or pinhole_diam_m != 0: proper.prop_circular_aperture( wavefront, diam_oap6/2.0 )  

    if end_at_fpm_exit_pupil == 1:
        proper.prop_propagate( wavefront, d_oap6_exitpupil, 'FPM EXIT PUPIL' )
        (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )
        wavefront = trim(wavefront, n)
        return wavefront, sampling_m

    proper.prop_propagate( wavefront, d_oap6_lyotstop, 'LYOT STOP' )
    if use_lyot_stop != 0:
        if np.isscalar(lyot_stop_array):
            lyot = proper.prop_fits_read( lyot_stop_file )
            lyot = trim( lyot, n )
        else:
            lyot = trim( lyot_stop_array, n )
        if sm_despace_m != 0 or fcm_z_shift_m != 0 or (use_cvs == 1 and cvs_source_z_offset_m != 0):
            #-- resize Lyot stop if something moved that would change sampling at pupil
            mag = 2 * proper.prop_get_beamradius(wavefront) / diam_beam_at_lyot_stop
        else:
            mag = 1
        if lyot_x_shift_pupdiam != 0 or lyot_y_shift_pupdiam != 0 or lyot_x_shift_m != 0 or lyot_y_shift_m != 0 or lyot_rotation_deg != 0 or mag != 1:
            if lyot_x_shift_pupdiam != 0 or lyot_y_shift_pupdiam != 0:
                # offsets are normalized to pupil diameter
                xshift_pix = lyot_x_shift_pupdiam * diam_beam_at_lyot_stop / proper.prop_get_sampling(wavefront)
                yshift_pix = lyot_y_shift_pupdiam * diam_beam_at_lyot_stop / proper.prop_get_sampling(wavefront)
            else:
                xshift_pix = lyot_x_shift_m / proper.prop_get_sampling(wavefront)
                yshift_pix = lyot_y_shift_m / proper.prop_get_sampling(wavefront)
            lyot = transform_image( lyot, xshift_pix, yshift_pix, lyot_rotation_deg, mag )
        proper.prop_multiply( wavefront, lyot )
        lyot = 0
    if use_pupil_lens != 0 or pinhole_diam_m != 0: proper.prop_circular_aperture( wavefront, 1.079, NORM=True )

    proper.prop_propagate( wavefront, d_lyotstop_oap7, 'OAP7' )
    proper.prop_lens( wavefront, fl_oap7 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP7_phase_error_V4.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap7/2.0 )  

    proper.prop_propagate( wavefront, d_oap7_fieldstop, 'FIELD_STOP', TO_PLANE=1 )
    if use_field_stop != 0:
        m_per_lam0divD = m_per_lamdivD_575nm_at_field_stop * lambda0_m / 575e-9   # meters per lam/D at reference wavelength for nominal design
        if np.isscalar(field_stop_array) and is_hlc:     # is HLC, no field stop array  
            stop_radius_m = field_stop_radius_lam0 * m_per_lam0divD
            if field_stop_x_offset != 0 or field_stop_y_offset != 0:
                # convert offsets in lambda0/D to meters
                field_stop_x_offset_m = field_stop_x_offset * m_per_lam0divD
                field_stop_y_offset_m = field_stop_y_offset * m_per_lam0divD
            proper.prop_circular_aperture( wavefront, stop_radius_m, -field_stop_x_offset_m, -field_stop_y_offset_m )
        elif not np.isscalar(field_stop_array):
            # use MFTs to use super-sampled field stop
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, 1 )                # to virtual pupil
            wavefront0 = trim(wavefront0, n_mft)
            m_per_lamdivD = proper.prop_get_sampling(wavefront) / (pupil_diam_pix / n)
            field_stop_sampling_lamdivD = field_stop_array_sampling_m / m_per_lamdivD
            nstop = field_stop_array.shape[1]
            wavefront0 = mft2(wavefront0, field_stop_sampling_lamdivD, pupil_diam_pix, nstop, -1)   # MFT to highly-sampled focal plane
            wavefront0 *= field_stop_array
            wavefront0 = mft2(wavefront0, field_stop_sampling_lamdivD, pupil_diam_pix, n, +1)  # MFT to virtual pupil 
            wavefront0 = ffts( wavefront0, -1 )    # back to normally-sampled focal plane
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)

    proper.prop_propagate( wavefront, d_fieldstop_oap8, 'OAP8' )
    proper.prop_lens( wavefront, fl_oap8 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP8_phase_error_V3.0.fits', WAVEFRONT=True )
    if use_aperture != 0: 
        proper.prop_circular_aperture( wavefront, diam_oap8/2.0 )  
        # if use_aperture=1, then propagate from OAP8 to the CFAM baffle; otherwise, propagate directly to CFAM (filter)
        proper.prop_propagate( wavefront, d_oap8_cfambaffle, 'CFAM BAFFLE' )
        # baffle is oval design, but one side of beam is blocked by filter itself, so represent baffle as circle
        proper.prop_circular_aperture( wavefront, diam_cfambaffle/2 )
        proper.prop_propagate( wavefront, d_cfambaffle_filter, 'filter' )
    else:
        proper.prop_propagate( wavefront, d_oap8_filter, 'filter' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'roman_phasec_FILTER_phase_error_V1.0.fits', WAVEFRONT=True )

    # propagate to lens, apply lens, then propagate to detector

    if use_pupil_lens == 0 and use_defocus_lens == 0:
        # use imaging lens to create normal focus
        if use_errors != 0:
            imaging_lens_error_files = [map_dir+'roman_phasec_LENS_phase_error_V1.0.fits', ' ']
        else:
            imaging_lens_error_files = ' '
        to_from_doublet( wavefront, d_filter_lens, d_lens_fold4+d_fold4_image, 
                0.10792660718579995, -0.10792660718579995, 0.003, glass_index('S-BSL7R', lambda_m, data_dir), 0.0005, 
                1e10, 0.10608379812011390, 0.0025, glass_index('PBM2R', lambda_m, data_dir), 
                'IMAGING LENS', 'IMAGE', ERROR_MAP_FILES=imaging_lens_error_files, TO_PLANE=1 )
    elif use_pupil_lens != 0:
        # use pupil imaging lens
        if use_errors != 0:
            pupil_lens_error_files = [ map_dir+'roman_phasec_PUPIL_IMAGE_LENS1_measured_phase_error_V3.0.fits', map_dir+'roman_phasec_PUPIL_IMAGE_LENS2_measured_phase_error_V3.0.fits']
        else:
            pupil_lens_error_files = ' '
        to_from_doublet( wavefront, d_filter_lens, d_lens_fold4+d_fold4_image+1e-3, 
                0.0345, -0.2079, 0.002979, glass_index('S-BSL7R', lambda_m, data_dir), 0.000441, 
                190.0, 0.05462, 0.002453, glass_index('PBM2R', lambda_m, data_dir), 
                'PUPIL IMAGING LENS', 'IMAGE', ERROR_MAP_FILES=pupil_lens_error_files, TO_PLANE=0 )
    elif use_defocus_lens != 0:     
        # use one of 4 defocusing lenses
        r1_lens = [0.074908, 0.07965, 0.085518, 0.174872]
        r2_lens = [-93.62, -98.35, -99.12, -70.41]
        ct_lens = [0.004995, 0.00499, 0.004978, 0.004967]
        if use_errors != 0:
            defocus_lens_error_file = map_dir+'roman_phasec_DEFOCUSLENS' + str(use_defocus_lens) + '_measured_phase_error_V3.0.fits'
        else:
            defocus_lens_error_file = ' '
        to_from_singlet( wavefront, d_filter_lens, d_lens_fold4+d_fold4_image+1e-3, 
            r1_lens[use_defocus_lens-1], r2_lens[use_defocus_lens-1], ct_lens[use_defocus_lens-1],
            glass_index('SILICA', lambda_m, data_dir), 'DEFOCUS LENS', 'IMAGE', 
            ERROR_MAP_FILE=defocus_lens_error_file, TO_PLANE=(use_defocus_lens == 3 or use_defocus_lens == 4) )

    if end_at_exit_pupil != 0:
        # create a sharp pupil image
        proper.prop_propagate( wavefront, 0.1 )
        proper.prop_lens( wavefront, 0.1 )
        proper.prop_propagate( wavefront, 0.088125 )
    elif use_defocus_lens == 0 and use_pupil_lens == 0:
        rsqr = proper.prop_radius( wavefront )**2 

    if end_at_exit_pupil == 0:
        if 'PASSVALUE' in locals():
            if 'save_ref_radius' in PASSVALUE:
                # save_ref_radius is an integer >0; if defined, it will create a file
                # 'ref_radius_#.dat', where # is replaced with the value of save_ref_radius.
                # It will write out the wavefront reference radius in binary form. This is
                # used during phase retrieval studies using CGISim 
                fname = 'ref_radius_' + str(save_ref_radius) + '.dat'
                with open( fname, "wb" ) as file1: 
                    np.array( proper.prop_get_refradius(wavefront), dtype=float ).tofile(file1)

    (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )

    # remove phase term due to pupil not at front focus of lens (only in direct imaging mode) 

    if use_defocus_lens == 0 and use_pupil_lens == 0 and end_at_exit_pupil == 0:
        lam_m = np.array([500,525,550,575,600,625,650,700,730,775,825,880]) * 1e-9
        cc = np.array([1.1797,1.1840,1.1865,1.1881,1.1890,1.1891,1.1887,1.1868,1.1852,1.1820,1.1778,1.1726])
        f = interp1d(lam_m, cc, kind='linear', fill_value='extrapolate')
        c = f( lambda_m )
        wavefront *= np.exp((complex(0,1) * np.pi / lambda_m * c) * rsqr) * np.exp(-complex(0,1)*0.1)

    # rotate/flip wavefront to match CGI detector frame coordinates

    wavefront = np.fliplr(wavefront).copy() 
    wavefront = np.rot90( wavefront, 3 ).copy() 
    wavefront = np.roll( wavefront, (1,1), axis=(0,1) )   # put center back in original pixel (n/2,n/2) since n is even

    # shift image, if requested

    if image_x_offset_m != 0 or image_y_offset_m != 0:
        xoff_pix = image_x_offset_m / sampling_m
        yoff_pix = image_y_offset_m / sampling_m
        wavefront = shift_image( wavefront, xoff_pix, yoff_pix )

    # resample as needed

    if final_sampling_m != 0:
        mag = sampling_m / final_sampling_m
        sampling_m = final_sampling_m 
        wavefront = proper.prop_magnify( wavefront, mag, output_dim, AMP_CONSERVE=True )
    elif final_sampling_lam0 != 0:
        if use_pupil_lens != 0 or use_defocus_lens != 0 or end_at_exit_pupil != 0:
            raise Exception('ERROR: Cannot specify final_sampling_lam0 when using pupil or defocus lens or ending at exit pupil.')
        else:
            mag = (float(pupil_diam_pix)/n) / final_sampling_lam0 * (lambda_m/lambda0_m)
            sampling_m = sampling_m / mag
            wavefront = proper.prop_magnify( wavefront, mag, output_dim, AMP_CONSERVE=True )
    else:
        wavefront = trim(wavefront, output_dim)

    return wavefront, sampling_m

