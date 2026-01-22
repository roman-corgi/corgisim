from astropy.io import fits
from corgidrp import mocks
import os
from datetime import datetime, timezone, timedelta
#import warnings

def create_hdu_list(data, header_info, sim_info=None):
    """
    Create an Astropy HDUList for a simulated L1 FITS product.

    - The primary HDU contains a global header with no image data.
    - The image HDU contains the 2D simulated image and its own header.

    Default L1 headers from `corgidrp.mocks` are applied. Optional metadata can be added as comments to the primary header.

    Parameters
    ----------
    data : numpy.ndarray
        2D array representing the simulated image.
    header_info : dict
        Header keywords (e.g., 'EXPTIME', 'EMGAIN_C') to override defaults in the image HDU header.
    sim_info : dict, optional
        Key-value metadata describing the simulation setup, added as comments in the primary header.
    
    Returns
    -------
    hdul : astropy.io.fits.HDUList
        FITS HDUList with:
        [0] Primary HDU (no data, global header)
        [1] Image HDU (image data + header)
    """
    primary_hdu = fits.PrimaryHDU()
    image_hdu = fits.ImageHDU(data)
    hdul = fits.HDUList([primary_hdu, image_hdu])

    # Apply default L1 headers
    # populate L1 headers from input values or default from cgisim
    prihdr, exthdr = mocks.create_default_L1_headers()
    prihdr['SIMPLE'] = True
    prihdr['ORIGIN'] = 'CorgiSim'
    prihdr['mock'] = 1
    prihdr['TELESCOP'] = 'ROMAN'
    prihdr['PSFREF'] = header_info['PSFREF']
    prihdr['OPGAIN'] = header_info['EMGAIN_C']
    if header_info['PHTCNT'] == True:
        prihdr['PHTCNT'] =int(1)
    else:
        prihdr['PHTCNT'] =int(0)

    ### currently we don't have sequence smulation, so the time per frame == exposure time
    ### it needs to be updated later
    prihdr['FRAMET'] = header_info['EXPTIME']
    prihdr['SATSPOTS'] = int(header_info['SATSPOTS'])
    prihdr['ROLL'] = header_info['ROLL']

    ### wait this for tachi to add sattlite spots function
    #prihdr['SATSPOTS'] = header_info['SATSPOTS'] 
    
    time_in_name = isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_"
    prihdr['FILENAME'] =  f"{filename}.fits"

    

    exthdr['NAXIS'] = data.ndim
    exthdr['NAXIS1'] = data.shape[0]
    exthdr['NAXIS2'] = data.shape[1]
    exthdr['EXPTIME'] = header_info['EXPTIME']
    exthdr['EMGAIN_C'] = header_info['EMGAIN_C']
    exthdr['EMGAIN_A'] = header_info['EMGAIN_C']  
    exthdr['KGAINPAR'] =  header_info['KGAINPAR']
    exthdr['EACQ_ROW'] =  header_info['EACQ_ROW']
    exthdr['EACQ_COL'] =  header_info['EACQ_COL']

    exthdr['FSMLOS'] = 1

    # TODO: Figure out which one to use for spec 
    if 'hlc' in cor_type:
        exthdr['FSMPRFL'] = 'NFOV'
    elif 'wide' in cor_type:
        exthdr['FSMPRFL'] = 'WFOV'
    elif cor_type == 'spc-spec_band2':
        exthdr['FSMPRFL'] = 'SPEC660'
    elif cor_type == 'spc-spec_band3':
        exthdr['FSMPRFL'] = 'SPEC730'
    else :
        exthdr['FSMPRFL'] = 'FSM_PROFILE_UNKNOWN'
        
    if header_info['PHTCNT'] == True:
        exthdr['ISPC']= int(1)
    else:
        exthdr['ISPC']= int(0)

    for key in ['FSMX', 'FSMY']:
        exthdr[key] = header_info[key] if key in header_info else 0  # set the header from header_info or default in cgisim

    exthdr['SPAM_H'], exthdr['SPAM_V'], exthdr['SPAMNAME'], exthdr['SPAMSP_H'],exthdr['SPAMSP_V'] = write_headers_SPAM(header_info['cor_type'])
    exthdr['LSAM_H'], exthdr['LSAM_V'], exthdr['LSAMNAME'], exthdr['LSAMSP_H'],exthdr['LSAMSP_V'] = write_headers_LSAM(header_info['cor_type'], header_info['use_lyot_stop'] )
    exthdr['CFAM_H'], exthdr['CFAM_V'], exthdr['CFAMNAME'], exthdr['CFAMSP_H'],exthdr['CFAMSP_V'] = write_headers_CFAM(header_info['bandpass'])
    exthdr['DPAM_H'], exthdr['DPAM_V'], exthdr['DPAMNAME'], exthdr['DPAMSP_H'],exthdr['DPAMSP_V'] = write_headers_DPAM(header_info['cgi_mode'], header_info['polarization_basis'], header_info['prism'], header_info['use_pupil_lens'])
    
    ##### need to update later
    exthdr['FPAM_H'], exthdr['FPAM_V'], exthdr['FPAMNAME'], exthdr['FPAMSP_H'],exthdr['FPAMSP_V'] = write_headers_FPAM(header_info['cor_type'], header_info['bandpass'], header_info['use_fpm'], header_info['nd_filter'])
    exthdr['FSAM_H'], exthdr['FSAM_V'], exthdr['FSAMNAME'], exthdr['FSAMSP_H'],exthdr['FSAMSP_V'] = write_headers_FSAM(header_info['cor_type'], header_info['bandpass'], header_info['slit'], header_info['polaxis'], header_info['use_field_stop'])



    hdul[0].header = prihdr
    hdul[1].header = exthdr

    if sim_info:
        hdul[0].header['COMMENT'] = "Simulation-specific metadata below:"
        for key, value in sim_info.items():
            hdul[0].header.add_comment(f"{key} : {value}")

    return hdul


def create_hdu(data, sim_info=None):
    """
    Create a simple FITS HDU for intermediate image products (e.g., PSFs).

    Parameters
    ----------
    data : numpy.ndarray
        2D array representing the simulated image or PSF.
    sim_info : dict, optional
        Key-value metadata added as header comments.

    Returns
    -------
    hdu : astropy.io.fits.PrimaryHDU
        FITS PrimaryHDU with data and metadata in the header.
    """
    hdu = fits.PrimaryHDU(data)

    if sim_info:
        hdu.header['COMMENT'] = "Simulated data with metadata below:"
        for key, value in sim_info.items():
            hdu.header.add_comment(f"{key} : {value}")

    return hdu


def save_hdu_to_fits( hdul, outdir=None, overwrite=False, write_as_L1=False, filename=None):
        """
        Save an Astropy HDUList to a FITS file.

        Parameters:
            - hdul (astropy.io.fits.HDUList): The HDUList object to be saved.
            - outdir (str, optional): Output directory. Defaults to the current working directory.
            - overwrite (bool): If True, overwrite the file if it already exists. Default is True.
            - write_as_L1 (bool): If True, the file will be named according to the L1 naming convention.
            - filename (str, optional): Name of the output FITS file (without ".fits" extension). 
                                    Required if write_as_L1 is False.
                                    
        """
        if outdir is None:
            outdir = os.getcwd()

        os.makedirs(outdir, exist_ok=True)

        # Handle naming logic
        if write_as_L1:
            #following the name convention for L1 product
            if (hdul[1].data.shape[0] != 1200) or (hdul[1].data.shape[1] != 2200):
                raise ValueError("Only full frame image can be save as L1 products")
                
            if filename is not None:
                raise Warning("The provided filename is overridden for L1 products.")
            prihdr = hdul[0].header
            exthdr = hdul[1].header

            #time_in_name = isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
            #filename = f"CGI_{prihdr['VISITID']}_{time_in_name}_L1_"
            filename = prihdr['FILENAME']
        else:
            if filename is None:
                raise ValueError("Filename must be provided when write_as_L1 is False.")

        # Construct full file path
        filepath = os.path.join(outdir, filename)
        
        # Write the HDUList to file
        hdul.writeto(filepath, overwrite=overwrite)
        print(f"Saved FITS file to: {filepath}")


def isotime_to_yyyymmddThhmmsss(timestr):
    """
    Format an ISO time string into a custom format yyyymmddThhmmsss

    Parameters
    ----------
    timestr : str
        ISO time (e.g., '2025-04-25T23:18:04.786184+00:00').

    Returns
    -------
        str
            Time formatted as 'yyyymmddThhmmsss' (e.g., '20250425T2318048').
    """
    # Parse the input ISO format time
    t = datetime.fromisoformat(timestr)

    # Round to nearest 0.1 second
    microsecond = t.microsecond
    tenth_sec = round(microsecond / 1e5)  # how many tenths
    t = t.replace(microsecond=0)  # reset microseconds

    if tenth_sec == 10:
        # carry over 1 second if rounding goes to 10
        t += timedelta(seconds=1)
        tenth_sec = 0

    # Format as yyyymmddThhmmsss
    out = t.strftime("%Y%m%dT%H%M%S") + str(tenth_sec)
    return out

def write_headers_SPAM(cor_type):

    ### determine the value for SPAM based on coronagraph type
    if 'hlc' in cor_type:
        ## hlc NFOV imaging mode
        SPAM_H = 1001.3
        SPAM_V = 16627
        SPAMNAME = 'OPEN'
        SPAMSP_H = 1001.3
        SPAMSP_V = 16627

    if 'wide' in cor_type:
        ## spc WFOV imaging mode
        SPAM_H = 26254.7
        SPAM_V = 8657
        SPAMNAME = 'WFOV'
        SPAMSP_H = 26254.7
        SPAMSP_V = 8657

    if ('spec' in cor_type) & ('rotated' not in cor_type):
        ## spc spec mode, not rotated
        SPAM_H = 26250.4
        SPAM_V = 27254.4
        SPAMNAME = 'SPEC'
        SPAMSP_H = 26250.4
        SPAMSP_V = 27254.4

    if 'rotated' in cor_type:
        ## spc spec mode, rotated
        SPAM_H =44850.4
        SPAM_V = 8654.4
        SPAMNAME = 'SPECROT'
        SPAMSP_H = 44850.4
        SPAMSP_V = 8654.4

    return SPAM_H,SPAM_V,SPAMNAME,SPAMSP_H, SPAMSP_V





def write_headers_LSAM(cor_type, use_lyot_stop):
    if not use_lyot_stop:
        ## not use lyot stop
        LSAM_H = 20822
        LSAM_V = 17393.9
        LSAMNAME = 'OPEN'
        LSAMSP_H =  20822
        LSAMSP_V = 17393.9
    if use_lyot_stop:
        ### determine the value for LSAM based on coronagraph type
        if 'hlc' in cor_type:
            ## hlc NFOV imaging mode
            LSAM_H = 36898.7
            LSAM_V = 4636.2
            LSAMNAME = 'NFOV'
            LSAMSP_H = 36898.7
            LSAMSP_V = 4636.2

        if 'wide' in cor_type:
            ## spc WFOV imaging mode
            LSAM_H = 1424.3
            LSAM_V = 29440.2
            LSAMNAME = 'WFOV'
            LSAMSP_H = 1424.3
            LSAMSP_V = 29440.2

        if ('spec' in cor_type) & ('rotated' not in cor_type):
            ## spc spec mode, not rotated
            LSAM_H = 36936.3
            LSAM_V = 29389.3
            LSAMNAME = 'SPEC'
            LSAMSP_H = 36936.3
            LSAMSP_V = 29389.3

        if 'rotated' in cor_type:
            ## spc spec mode, rotated
            LSAM_H =  1426.6
            LSAM_V = 4581.4
            LSAMNAME = 'SPECROT'
            LSAMSP_H = 1426.6
            LSAMSP_V = 4581.4

    return LSAM_H, LSAM_V, LSAMNAME, LSAMSP_H, LSAMSP_V


def write_headers_CFAM(band_pass):
   
    ### determine the value for CFAM based on bandpass
    if band_pass == '1F':
        CFAM_H = 55829.2
        CFAM_V = 10002.7
        CFAMNAME = '1F'
        CFAMSP_H = 55829.2
        CFAMSP_V = 10002.7
    elif band_pass == '1A':
        CFAM_H = 43329.2
        CFAM_V = 10002.7
        CFAMNAME = '1A'
        CFAMSP_H = 43329.2
        CFAMSP_V = 10002.7
    elif band_pass == '1B':
        CFAM_H = 27329.2
        CFAM_V = 9202.7
        CFAMNAME = '1B'
        CFAMSP_H = 27329.2
        CFAMSP_V = 9202.7
    elif band_pass == '1C':
        CFAM_H = 14829.2
        CFAM_V = 10002.7
        CFAMNAME = '1C'
        CFAMSP_H = 14829.2
        CFAMSP_V = 10002.7
    elif band_pass == '2F':
        CFAM_H = 62079.2
        CFAM_V = 1002.7
        CFAMNAME = '2F'
        CFAMSP_H = 62079.2
        CFAMSP_V = 1002.7
    elif band_pass == '2A':
        CFAM_H = 49579.2
        CFAM_V = 1002.7
        CFAMNAME = '2A'
        CFAMSP_H = 49579.2
        CFAMSP_V = 1002.7
    elif band_pass == '2B':
        CFAM_H = 37079.2
        CFAM_V = 1002.7
        CFAMNAME = '2B'
        CFAMSP_H = 37079.2
        CFAMSP_V = 1002.7
    elif band_pass == '2C':
        CFAM_H = 21079.2
        CFAM_V = 1002.7
        CFAMNAME = '2C'
        CFAMSP_H = 21079.2
        CFAMSP_V = 1002.7
    elif band_pass == '3F':
        CFAM_H = 2329.2
        CFAM_V = 24002.7
        CFAMNAME = '3F'
        CFAMSP_H = 2329.2
        CFAMSP_V = 24002.7
    elif band_pass == '3A':
        CFAM_H = 14829.2
        CFAM_V = 24002.7
        CFAMNAME = '3A'
        CFAMSP_H = 14829.2
        CFAMSP_V = 24002.7
    elif band_pass == '3B':
        CFAM_H = 27329.2
        CFAM_V = 24002.7
        CFAMNAME = '3B'
        CFAMSP_H = 27329.2
        CFAMSP_V = 24002.7
    elif band_pass == '3C':
        CFAM_H = 43768.0
        CFAM_V = 24443.8
        CFAMNAME = '3C'
        CFAMSP_H = 43768.0
        CFAMSP_V = 24443.8
    elif band_pass == '3D':
        CFAM_H = 2329.2
        CFAM_V = 10002.7
        CFAMNAME = '3D'
        CFAMSP_H = 2329.2
        CFAMSP_V = 10002.7
    elif band_pass == '3E':
        CFAM_H = 8579.2
        CFAM_V = 1002.7
        CFAMNAME = '3E'
        CFAMSP_H = 8579.2
        CFAMSP_V = 1002.7
    elif band_pass == '3G':
        CFAM_H = 55829.2
        CFAM_V = 24002.7
        CFAMNAME = '3G'
        CFAMSP_H = 55829.2
        CFAMSP_V = 24002.7
    elif band_pass == '4F':
        CFAM_H = 8079.1
        CFAM_V = 33003.1
        CFAMNAME = '4F'
        CFAMSP_H = 8079.1
        CFAMSP_V = 33003.1
    elif band_pass == '4A':
        CFAM_H = 21079.2
        CFAM_V = 33002.7
        CFAMNAME = '4A'
        CFAMSP_H = 21079.2
        CFAMSP_V = 33002.7
    elif band_pass == '4B':
        CFAM_H = 37079.2
        CFAM_V = 33002.7
        CFAMNAME = '4B'
        CFAMSP_H = 37079.2
        CFAMSP_V = 33002.7
    elif band_pass == '4C':
        CFAM_H = 49579.2
        CFAM_V = 33002.7
        CFAMNAME = '4C'
        CFAMSP_H = 49579.2
        CFAMSP_V = 33002.7

    return CFAM_H, CFAM_V, CFAMNAME, CFAMSP_H, CFAMSP_V
    
    

def write_headers_DPAM(cor_mode, polarization_basis, prism, use_pupil_lens):
     ### determine the value for DPAM based on simulation mode and polarization basis
    if use_pupil_lens :
        ## pupil imaging
        DPAM_H = 62626.4
        DPAM_V = 21024.3
        DPAMNAME = 'PUPIL'
        DPAMSP_H = 62626.4
        DPAMSP_V = 21024.3
    if not use_pupil_lens:
        if (cor_mode == 'excam'):
            if polarization_basis == 'None':
                #no wollaston
                DPAM_H = 38917.1
                DPAM_V = 26016.9
                DPAMNAME = 'IMAGING'
                DPAMSP_H = 38917.1
                DPAMSP_V = 26016.9
            elif polarization_basis == '0/90 degrees':
                #0/90 polarization
                DPAM_H = 8991.3
                DPAM_V = 1261.3
                DPAMNAME = 'POL0'
                DPAMSP_H = 8991.3
                DPAMSP_V = 1261.3
            else:
                #45/135 polarization
                DPAM_H = 44660.1
                DPAM_V = 1261.3
                DPAMNAME = 'POL45'
                DPAMSP_H = 44660.1
                DPAMSP_V = 1261.3
        if cor_mode == 'spec':
            #raise ValueError('Spec mode has not been implemented')
            if prism == 'PRISM2':
                DPAM_H = 62496
                DPAM_V = 1261.3
                DPAMNAME = 'PRISM2'
                DPAMSP_H = 62496
                DPAMSP_V = 1261.3
            if prism == 'PRISM3':
                DPAM_H = 26824.2
                DPAM_V = 1261.3
                DPAMNAME = 'PRISM3'
                DPAMSP_H = 26824.2
                DPAMSP_V = 1261.3
            if prism == 'None':
                DPAM_H = 38917.1
                DPAM_V = 26016.9
                DPAMNAME = 'IMAGING'
                DPAMSP_H = 38917.1
                DPAMSP_V = 26016.9


    return DPAM_H, DPAM_V, DPAMNAME,  DPAMSP_H,  DPAMSP_V

def write_headers_FPAM(cor_type, band_pass,use_fpm,nd_filter):
    ### determine the value for FPAM based on coronagraph type and bandpass, and if use fpm
    if not use_fpm:
        ## not using foca plane mask, typically for non-occulted stars
        if (nd_filter == '0'):
            ## no ND filter
            if ('1' in band_pass) or ('2' in band_pass):
                FPAM_H = 3509.4
                FPAM_V = 32824.7
                FPAMNAME = 'OPEN_12'
                FPAMSP_H = 3509.4
                FPAMSP_V = 32824.7
            if ('3' in band_pass) or ('4' in band_pass):
                FPAM_H = 60251.2
                FPAM_V = 2248.5
                FPAMNAME = 'OPEN_34'
                FPAMSP_H = 60251.2
                FPAMSP_V = 2248.5
        if (nd_filter == '2.25'):
            ## ND filter1 
            FPAM_H = 61507.8
            FPAM_V = 25612.4
            FPAMNAME = 'ND225'
            FPAMSP_H = 61507.8
            FPAMSP_V = 25612.4
        if (nd_filter == '4.75fpam'):
            ## ND filter2
            FPAM_H = 2503.7
            FPAM_V = 6124.9
            FPAMNAME = 'ND475'
            FPAMSP_H = 2503.7
            FPAMSP_V = 6124.9
    else:
        if 'hlc' in cor_type:
            ##hlc NFOV imaging mode
            if '1' in band_pass:
                FPAM_H = 6757.2
                FPAM_V = 22424
                FPAMNAME = 'HLC12_C2R1'
                FPAMSP_H = 6757.2
                FPAMSP_V = 22424
            
            if '2' in band_pass:
                FPAM_H = 55306.4
                FPAM_V = 9901.9
                FPAMNAME = 'HLC34_R7C1'
                FPAMSP_H = 55306.4
                FPAMSP_V = 9901.9

            if '3' in band_pass:
                FPAM_H = 52005.1
                FPAM_V = 8004.2
                FPAMNAME = 'HLC34_R5C1'
                FPAMSP_H = 52005.1
                FPAMSP_V = 8004.2

            if '4' in band_pass:
                FPAM_H = 52003.8
                FPAM_V = 6104.2
                FPAMNAME = 'HLC34_R3C1'
                FPAMSP_H = 52003.8
                FPAMSP_V = 6104.2

        if 'wide' in cor_type:
            ##spc WFOV imaging mode
            if '1' in band_pass:
                FPAM_H = 23719.6
                FPAM_V = 2278.1
                FPAMNAME = 'SPC12_R1C1'
                FPAMSP_H = 23719.6
                FPAMSP_V = 2278.1

            if '4' in band_pass:
                FPAM_H = 35354.3
                FPAM_V =27622.6
                FPAMNAME = 'SPC34_R5C1'
                FPAMSP_H = 35354.3
                FPAMSP_V = 27622.6

        if ('spec' in cor_type) & ('rotated' not in cor_type):
                ## spc spec mode, not rotated
            if '2' in band_pass:
                FPAM_H = 25866.4
                FPAM_V = 7129.5
                FPAMNAME = 'SPC12_R3C2'
                FPAMSP_H = 25866.4
                FPAMSP_V = 7129.5

            if '3' in band_pass:
                FPAM_H = 37005.5
                FPAM_V = 22573
                FPAMNAME = 'SPC34_R2C2'
                FPAMSP_H = 37005.5
                FPAMSP_V = 22573

        if 'rotated' in cor_type:
            ## spc spec mode, rotated
            if '2' in band_pass:
                FPAM_H = 22666.4
                FPAM_V = 7127.4
                FPAMNAME = 'SPC12_R3C1'
                FPAMSP_H = 22666.4
                FPAMSP_V = 7127.4

            if '3' in band_pass:
                FPAM_H = 40505.5
                FPAM_V = 22573.8
                FPAMNAME = 'SPC34_R2C3'
                FPAMSP_H = 40505.5
                FPAMSP_V = 22573.8

    return FPAM_H, FPAM_V, FPAMNAME, FPAMSP_H, FPAMSP_V


def write_headers_FSAM(cor_type, band_pass,slit,polaxis,use_field_stop):
    if not use_field_stop:
        FSAM_H =30677.2
        FSAM_V = 2959.5
        FSAMNAME = 'OPEN'
        FSAMSP_H = 30677.2
        FSAMSP_V = 2959.5
    if use_field_stop:
        ### determine the value for FSAM based on coronagraph type and bandpass
        if 'hlc' in cor_type:
            if '1' in band_pass:
                FSAM_H = 29387
                FSAM_V = 12238
                FSAMNAME = 'R1C1'
                FSAMSP_H = 29387
                FSAMSP_V = 12238

            if '2' in band_pass:
                FSAM_H = 17937
                FSAM_V = 21238
                FSAMNAME = 'R3C3'
                FSAMSP_H = 17937
                FSAMSP_V = 21238

            if '3' in band_pass:
                FSAM_H = 13437
                FSAM_V = 21238
                FSAMNAME = 'R3C4'
                FSAMSP_H = 13437
                FSAMSP_V = 21238

            if '4' in band_pass:
                FSAM_H = 8937
                FSAM_V = 21238
                FSAMNAME = 'R3C5'
                FSAMSP_H = 8937
                FSAMSP_V = 21238

        if 'wide' in cor_type:
            if polaxis in ['0', '10']:
                FSAM_H =30677.2
                FSAM_V = 2959.5
                FSAMNAME = 'OPEN'
                FSAMSP_H = 30677.2
                FSAMSP_V = 2959.5
            else:
                raise ValueError("Polarimetry mode has not been implemented for FSAM")

        if ('spec' in cor_type) & ('rotated' not in cor_type):
            if '2' in band_pass:
                if slit == "R6C5":
                    FSAM_H = 11187
                    FSAM_V = 32638
                    FSAMNAME = 'R6C5'
                    FSAMSP_H = 11187
                    FSAMSP_V = 32638
                elif slit == "R3C1":
                    FSAM_H = 26937
                    FSAM_V = 21238
                    FSAMNAME = 'R3C1'
                    FSAMSP_H = 26937
                    FSAMSP_V = 21238
                if slit == "None":
                    FSAM_H =30677.2
                    FSAM_V = 2959.5
                    FSAMNAME = 'OPEN'
                    FSAMSP_H = 30677.2
                    FSAMSP_V = 2959.5
                    

            if '3' in band_pass:
                if slit == "R1C2":
                    FSAM_H = 24087
                    FSAM_V = 12238
                    FSAMNAME = 'R1C2'
                    FSAMSP_H = 24087
                    FSAMSP_V = 12238
                elif slit == "R3C1":
                    FSAM_H = 26937
                    FSAM_V = 21238
                    FSAMNAME = 'R3C1'
                    FSAMSP_H = 26937
                    FSAMSP_V = 21238
                if slit == "None":
                    FSAM_H =30677.2
                    FSAM_V = 2959.5
                    FSAMNAME = 'OPEN'
                    FSAMSP_H = 30677.2
                    FSAMSP_V = 2959.5

        if 'rotated' in cor_type:
            if '2' in band_pass:
                FSAM_H = 20187
                FSAM_V = 25038
                FSAMNAME = 'R4C3'
                FSAMSP_H = 20187
                FSAMSP_V = 25038

            if '3' in band_pass:
                FSAM_H = 24687
                FSAM_V = 17438
                FSAMNAME = 'R2C2'
                FSAMSP_H = 24687
                FSAMSP_V = 17438

    return FSAM_H, FSAM_V, FSAMNAME, FSAMSP_H, FSAMSP_V

def str2bool(value):
    """Convert string representations of booleans/ints to actual bools."""
    return str(value).lower() in ("true", "1")

       
       








