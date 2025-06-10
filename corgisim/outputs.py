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
    prihdr['PHTCNT'] = header_info['PHTCNT']

    ### currently we don't have sequence smulation, so the time per frame == exposure time
    ### it needs to be updated later
    prihdr['FRAMET'] = header_info['EXPTIME']

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


