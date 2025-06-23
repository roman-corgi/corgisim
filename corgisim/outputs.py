from astropy.io import fits
from corgidrp import mocks
import os
from datetime import datetime, timezone, timedelta

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
    prihdr['ROLL'] = header_info['ROLL']


    ### wait this for tachi to add sattlite spots function
    #prihdr['SATSPOTS'] = header_info['SATSPOTS'] 
    
    time_in_name = isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"CGI_{prihdr['VISITID']}_{time_in_name}_L1_"
    prihdr['FILENAME'] =  f"{filename}.fits"

    

    exthdr['NAXIS'] = data.ndim
    exthdr['NAXIS1'] = data.shape[0]
    exthdr['NAXIS2'] = data.shape[0]
    exthdr['EXPTIME'] = header_info['EXPTIME']
    exthdr['EMGAIN_C'] = header_info['EMGAIN_C']
    exthdr['EMGAIN_A'] = header_info['EMGAIN_C']  
    exthdr['KGAINPAR'] =  header_info['KGAINPAR']
    if header_info['PHTCNT'] == True:
        exthdr['ISPC']= int(1)
    else:
        exthdr['ISPC']= int(0)

    for key in ['FSMX', 'FSMY']:
        exthdr[key] = header_info[key] if key in header_info else 0  # set the header from header_info or default in cgisim

    exthdr['SPAM_H'], exthdr['SPAM_V'], exthdr['SPAMNAME'], exthdr['SPAMSP_H'],exthdr['SPAMSP_V'] = write_headers_SPAM(header_info['cor_type'])
    exthdr['LSAM_H'], exthdr['LSAM_V'], exthdr['LSAMNAME'], exthdr['LSAMSP_H'],exthdr['LSAMSP_V'] = write_headers_LSAM(header_info['cor_type'])
    exthdr['CFAM_H'], exthdr['CFAM_V'], exthdr['CFAMNAME'], exthdr['CFAMSP_H'],exthdr['CFAMSP_V'] = write_headers_CFAM(header_info['bandpass'])
    exthdr['DPAM_H'], exthdr['DPAM_V'], exthdr['DPAMNAME'], exthdr['DPAMSP_H'],exthdr['DPAMSP_V'] = write_headers_DPAM(header_info['cgi_mode'], header_info['polaxis'])
    
    ##### need to update later
    exthdr['FPAM_H'], exthdr['FPAM_V'], exthdr['FPAMNAME'], exthdr['FPAMSP_H'],exthdr['FPAMSP_V'] = 'N/A','N/A','N/A','N/A','N/A'
    exthdr['FSAM_H'], exthdr['FSAM_V'], exthdr['FSAMNAME'], exthdr['FSAMSP_H'],exthdr['FSAMSP_V'] = 'N/A','N/A','N/A','N/A','N/A'



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


def save_hdu_to_fits( hdul, outdir=None, overwrite=True, write_as_L1=False, filename=None):
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
        SPAM_H = 1001.3
        SPAM_V = 16627
        SPAMNAME = 'OPEN'
        SPAMSP_H = 1001.3
        SPAMSP_V = 16627

    if 'wide' in cor_type:
        SPAM_H = 26254.7
        SPAM_V = 8657
        SPAMNAME = 'WFOV'
        SPAMSP_H = 26254.7
        SPAMSP_V = 8657

    if ('spec' in cor_type) & ('rotated' not in cor_type):
        SPAM_H = 26250.4
        SPAM_V = 27254.4
        SPAMNAME = 'SPEC'
        SPAMSP_H = 26250.4
        SPAMSP_V = 27254.4

    if 'rotated' in cor_type:
        SPAM_H =44850.4
        SPAM_V = 8654.4
        SPAMNAME = 'SPECROT'
        SPAMSP_H = 44850.4
        SPAMSP_V = 8654.4

    return SPAM_H,SPAM_V,SPAMNAME,SPAMSP_H, SPAMSP_V





def write_headers_LSAM(cor_type):

    ### determine the value for LSAM based on coronagraph type
    if 'hlc' in cor_type:
        LSAM_H = 36898.7
        LSAM_V = 4636.2
        LSAMNAME = 'NFOV'
        LSAMSP_H = 36898.7
        LSAMSP_V = 4636.2

    if 'wide' in cor_type:
        LSAM_H = 1424.3
        LSAM_V = 29440.2
        LSAMNAME = 'WFOV'
        LSAMSP_H = 1424.3
        LSAMSP_V = 29440.2

    if ('spec' in cor_type) & ('rotated' not in cor_type):
        LSAM_H = 36936.3
        LSAM_V = 29389.3
        LSAMNAME = 'SPEC'
        LSAMSP_H = 36936.3
        LSAMSP_V = 29389.3

    if 'rotated' in cor_type:
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
    
    

def write_headers_DPAM(cor_mode, polaxis):
     ### determine the value for DPAM based on simulation mode and polaxis number
    
    if (cor_mode == 'excam') & (polaxis in ['0', '10']):
        DPAM_H = 38917.1
        DPAM_V = 26016.9
        DPAMNAME = 'IMAGING'
        DPAMSP_H = 38917.1
        DPAMSP_V = 26016.9
    if cor_mode == 'spec':
        raise ValueError('Spec mode has not been implemented')
    if (cor_mode == 'excam') & (polaxis not in ['0', '10']):
        raise ValueError('Polarimetry mode has not been implemented')

    return DPAM_H, DPAM_V, DPAMNAME,  DPAMSP_H,  DPAMSP_V


