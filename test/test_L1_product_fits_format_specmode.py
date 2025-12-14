from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os
import corgisim
from corgisim import outputs

def test_L1_product_fits_format_specmode():
    """Test the headers of saved L1 product FITS file for spec mode
    """
    #print('testrun')
    #### simulate using corgisim
    #### testing the defalut value pass to header

    Vmag = 6                            # V-band magnitude of the host star
    sptype = 'G0V'                      # Spectral type of the host star
    ref_flag = False                    # if the target is a reference star or not, default is False
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': False}

    # --- Companion Properties ---
    mag_companion = [22]           # List of magnitudes for each companion

    # Define their positions relative to the host star, in milliarcseconds (mas)
    # For reference: 1 λ/D at 550 nm with a 2.3 m telescope is ~49.3 mas
    mas_per_lamD = 63.72 # Band 3
    dx = [2 * mas_per_lamD]         # X positions in mas for each companion
    dy = [6 * mas_per_lamD]         # Y positions in mas for each companion

    # Construct a list of dictionaries for all companion point sources
    point_source_info = [
        {
            'Vmag': mag_companion[0],
            'magtype': 'vegamag',
            'position_x': dx[0],
            'position_y': dy[0]
        }
    ]

    # --- Create the Astrophysical Scene ---
    # This Scene object combines the host star and companion(s)
    base_scene = scene.Scene(host_star_properties, point_source_info)

    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'
    bandpass = '3F'
    cases = ['2e-8']      
    # cases = ['1e-9']      
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    ##  Define the polaxis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10
    # output_dim define the size of the output image
    output_dim = 121
    overfac = 5

    optics_keywords_slit_prism ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim, 
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,'nd':1,
                    'slit':'R1C2', 'slit_dec_offset_mas':base_scene.point_source_ddec[0], 'prism':'PRISM3', 'wav_step_um':2E-3}
    optics_slit_prism = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords_slit_prism, if_quiet=True,
                                    small_spc_grid = 1, oversampling_factor = overfac, return_oversample = False)


    sim_scene_slit_prism = optics_slit_prism.get_host_star_psf(base_scene)

    sim_scene_slit_prism = optics_slit_prism.inject_point_sources(base_scene,sim_scene_slit_prism)

    emccd_keywords ={'cr_rate':0.0}
    exptime = 3000

    detector = instrument.CorgiDetector( emccd_keywords)
    sim_scene_slit_prism = detector.generate_detector_image(sim_scene_slit_prism, exptime,full_frame=True,loc_x=512, loc_y=512)

    ### save the L1 product fits file to test/testdata folder
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
    outputs.save_hdu_to_fits(sim_scene_slit_prism.image_on_detector,outdir=outdir, write_as_L1=True)

    ### read the L1 product fits file
    prihdr = sim_scene_slit_prism.image_on_detector[0].header
    exthdr = sim_scene_slit_prism.image_on_detector[1].header
    time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"


    f = os.path.join( outdir , filename)

    with fits.open(f) as hdul:
        data = hdul[1].data
        prihr = hdul[0].header
        exthr = hdul[1].header
        
        # Check that the dtype is exactly uint16
        assert exthdr['SPAM_H'] ==  26250.4, f"Expected data SPAM_H=26250.4, but got {exthdr['SPAM_H']}"
        assert exthdr['SPAM_V']== 27254.4,  f"Expected data SPAM_V = 27254.4, but got {exthdr['SPAM_V']}"
        assert exthdr['SPAMNAME'] =='SPEC' , f"Expected data SPAMNAME ='SPEC', but got {exthdr['SPAMNAME']}"
        assert exthdr['SPAMSP_H']== 26250.4, f"Expected data SPAMSP_H=26250.4, but got {exthdr['SPAMSP_H']}"
        assert exthdr['SPAMSP_V'] == 27254.4, f"Expected data SPAMSP_V=27254.4, but got {exthdr['SPAMSP_V']}"

        assert exthdr['LSAM_H'] ==  36936.3, f"Expected data LSAM_H=36936.3, but got {exthdr['LSAM_H']}"
        assert exthdr['LSAM_V']== 29389.3,  f"Expected data LSAM_V = 29389.3, but got {exthdr['LSAM_V']}"
        assert exthdr['LSAMNAME'] =='SPEC' , f"Expected data LSAMNAME ='SPEC', but got {exthdr['LSAMNAME']}"
        assert exthdr['LSAMSP_H']== 36936.3, f"Expected data LSAMSP_H=36936.3, but got {exthdr['LSAMSP_H']}"
        assert exthdr['LSAMSP_V'] == 29389.3, f"Expected data LSAMSP_V=29389.3, but got {exthdr['LSAMSP_V']}"

        assert exthdr['CFAM_H'] == 2329.2, f"Expected data CFAM_H=2329.2, but got {exthdr['CFAM_H']}"
        assert exthdr['CFAM_V'] == 24002.7, f"Expected data CFAM_V=24002.7, but got {exthdr['CFAM_V']}"
        assert exthdr['CFAMNAME'] == '3F', f"Expected data CFAMNAME='3F', but got {exthdr['CFAMNAME']}"
        assert exthdr['CFAMSP_H'] == 2329.2, f"Expected data CFAMSP_H=2329.2, but got {exthdr['CFAMSP_H']}"
        assert exthdr['CFAMSP_V'] == 24002.7, f"Expected data CFAMSP_V=24002.7, but got {exthdr['CFAMSP_V']}"

        assert exthdr['DPAM_H'] == 26824.2, f"Expected data DPAM_H=26824.2, but got {exthdr['DPAM_H']}"
        assert exthdr['DPAM_V'] == 1261.3, f"Expected data DPAM_V=1261.3, but got {exthdr['DPAM_V']}"
        assert exthdr['DPAMNAME'] == 'PRISM3', f"Expected data DPAMNAME='PRISM3', but got {exthdr['DPAMNAME']}"
        assert exthdr['DPAMSP_H'] == 26824.2, f"Expected data DPAMSP_H=26824.2, but got {exthdr['DPAMSP_H']}"
        assert exthdr['DPAMSP_V'] == 1261.3, f"Expected data DPAMSP_V=1261.3, but got {exthdr['DPAMSP_V']}"

        assert exthdr['FPAM_H'] ==  37005.5, f"Expected data FPAM_H= 37005.5, but got {exthdr['FPAM_H']}"
        assert exthdr['FPAM_V'] == 22573, f"Expected data FPAM_V=22573, but got {exthdr['FPAM_V']}"
        assert exthdr['FPAMNAME'] == 'SPC34_R2C2', f"Expected data FPAMNAME='SPC34_R2C2', but got {exthdr['FPAMNAME']}"
        assert exthdr['FPAMSP_H'] ==  37005.5, f"Expected data FPAMSP_H=37005.5, but got {exthdr['FPAMSP_H']}"
        assert exthdr['FPAMSP_V'] == 22573, f"Expected data FPAMSP_V=22573, but got {exthdr['FPAMSP_V']}"

        assert exthdr['FSAM_H'] ==  24087, f"Expected data FSAM_H=24087, but got {exthdr['FSAM_H']}"
        assert exthdr['FSAM_V'] == 12238, f"Expected data FSAM_V=12238, but got {exthdr['FSAM_V']}"
        assert exthdr['FSAMNAME'] == 'R1C2', f"Expected data FSAMNAME='R1C2', but got {exthdr['FSAMNAME']}"
        assert exthdr['FSAMSP_H'] ==  24087, f"Expected data FSAMSP_H=24087, but got {exthdr['FSAMSP_H']}"
        assert exthdr['FSAMSP_V'] == 12238, f"Expected data FSAMSP_V=12238, but got {exthdr['FSAMSP_V']}"

        ### delete file after testing
        print('Deleted the FITS file after testing headers populated with default values')
        os.remove(f)

    # Test the L1 format for a no-slit + prism configuration
    mas_per_lamD = 63.72 # Band 3
    source_y_offset = 6.0 #lam/D

    noslit_optics_keywords ={'cor_type':cor_type, 'use_errors':0, 'polaxis':polaxis, 'output_dim':output_dim, 
        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0, 'use_lyot_stop':1, 'fsm_y_offset':source_y_offset,
        'prism':'PRISM3', 'wav_step_um':2E-3}

    optics_noslit_prism_cfam3F = instrument.CorgiOptics(cgi_mode, bandpass='3F', optics_keywords=noslit_optics_keywords, 
                                                        if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    sim_unocc_noslit_prism_cfam3F = optics_noslit_prism_cfam3F.get_host_star_psf(base_scene)

    short_exptime = 0.5
    emccd_keywords_unitygain = {'cr_rate':0.0, 'em_gain':1}
    detector_unitygain = instrument.CorgiDetector(emccd_keywords_unitygain)

    unoccstar_prism_fullframe_sim = detector_unitygain.generate_detector_image(sim_unocc_noslit_prism_cfam3F, short_exptime, 
                                                                               full_frame=True, loc_x=512, loc_y=512)

    ### save the L1 product fits file to test/testdata folder
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
    outputs.save_hdu_to_fits(sim_unocc_noslit_prism_cfam3F.image_on_detector,outdir=outdir, write_as_L1=True)

    ### read the L1 product fits file
    prihdr = sim_unocc_noslit_prism_cfam3F.image_on_detector[0].header
    exthdr = sim_unocc_noslit_prism_cfam3F.image_on_detector[1].header
    time_in_name = outputs.isotime_to_yyyymmddThhmmsss(exthdr['FTIMEUTC'])
    filename = f"cgi_{prihdr['VISITID']}_{time_in_name}_l1_.fits"

    f = os.path.join( outdir , filename)

    with fits.open(f) as hdul:
        data = hdul[1].data
        prihr = hdul[0].header
        exthr = hdul[1].header
        
        # Check that the dtype is exactly uint16
        assert exthdr['SPAM_H'] ==  26250.4, f"Expected data SPAM_H=26250.4, but got {exthdr['SPAM_H']}"
        assert exthdr['SPAM_V']== 27254.4,  f"Expected data SPAM_V = 27254.4, but got {exthdr['SPAM_V']}"
        assert exthdr['SPAMNAME'] =='SPEC' , f"Expected data SPAMNAME ='SPEC', but got {exthdr['SPAMNAME']}"
        assert exthdr['SPAMSP_H']== 26250.4, f"Expected data SPAMSP_H=26250.4, but got {exthdr['SPAMSP_H']}"
        assert exthdr['SPAMSP_V'] == 27254.4, f"Expected data SPAMSP_V=27254.4, but got {exthdr['SPAMSP_V']}"

        assert exthdr['LSAM_H'] ==  36936.3, f"Expected data LSAM_H=36936.3, but got {exthdr['LSAM_H']}"
        assert exthdr['LSAM_V']== 29389.3,  f"Expected data LSAM_V = 29389.3, but got {exthdr['LSAM_V']}"
        assert exthdr['LSAMNAME'] =='SPEC' , f"Expected data LSAMNAME ='SPEC', but got {exthdr['LSAMNAME']}"
        assert exthdr['LSAMSP_H']== 36936.3, f"Expected data LSAMSP_H=36936.3, but got {exthdr['LSAMSP_H']}"
        assert exthdr['LSAMSP_V'] == 29389.3, f"Expected data LSAMSP_V=29389.3, but got {exthdr['LSAMSP_V']}"

        assert exthdr['CFAM_H'] == 2329.2, f"Expected data CFAM_H=2329.2, but got {exthdr['CFAM_H']}"
        assert exthdr['CFAM_V'] == 24002.7, f"Expected data CFAM_V=24002.7, but got {exthdr['CFAM_V']}"
        assert exthdr['CFAMNAME'] == '3F', f"Expected data CFAMNAME='3F', but got {exthdr['CFAMNAME']}"
        assert exthdr['CFAMSP_H'] == 2329.2, f"Expected data CFAMSP_H=2329.2, but got {exthdr['CFAMSP_H']}"
        assert exthdr['CFAMSP_V'] == 24002.7, f"Expected data CFAMSP_V=24002.7, but got {exthdr['CFAMSP_V']}"

        assert exthdr['DPAM_H'] == 26824.2, f"Expected data DPAM_H=26824.2, but got {exthdr['DPAM_H']}"
        assert exthdr['DPAM_V'] == 1261.3, f"Expected data DPAM_V=1261.3, but got {exthdr['DPAM_V']}"
        assert exthdr['DPAMNAME'] == 'PRISM3', f"Expected data DPAMNAME='PRISM3', but got {exthdr['DPAMNAME']}"
        assert exthdr['DPAMSP_H'] == 26824.2, f"Expected data DPAMSP_H=26824.2, but got {exthdr['DPAMSP_H']}"
        assert exthdr['DPAMSP_V'] == 1261.3, f"Expected data DPAMSP_V=1261.3, but got {exthdr['DPAMSP_V']}"

        assert exthdr['FPAM_H'] ==  60251.2, f"Expected data FPAM_H= 60251.2, but got {exthdr['FPAM_H']}"
        assert exthdr['FPAM_V'] == 2248.5, f"Expected data FPAM_V=2248.5, but got {exthdr['FPAM_V']}"
        assert exthdr['FPAMNAME'] == 'OPEN_34', f"Expected data FPAMNAME='OPEN_34', but got {exthdr['FPAMNAME']}"
        assert exthdr['FPAMSP_H'] ==  60251.2, f"Expected data FPAMSP_H= 60251.2, but got {exthdr['FPAMSP_H']}"
        assert exthdr['FPAMSP_V'] == 2248.5, f"Expected data FPAMSP_V=2248.5, but got {exthdr['FPAMSP_V']}"

        assert exthdr['FSAM_H'] ==  30677.2, f"Expected data FSAM_H=30677.2, but got {exthdr['FSAM_H']}"
        assert exthdr['FSAM_V'] == 2959.5, f"Expected data FSAM_V=2959.5, but got {exthdr['FSAM_V']}"
        assert exthdr['FSAMNAME'] == 'OPEN', f"Expected data FSAMNAME='OPEN', but got {exthdr['FSAMNAME']}"
        assert exthdr['FSAMSP_H'] ==  30677.2, f"Expected data FSAMSP_H=30677.2, but got {exthdr['FSAMSP_H']}"
        assert exthdr['FSAMSP_V'] == 2959.5, f"Expected data FSAMSP_V=2959.5, but got {exthdr['FSAMSP_V']}"

        ### delete file after testing
        print('Deleted the FITS file after testing headers populated with default values')
        os.remove(f)

if __name__ == '__main__':
    #run_sim()
    test_L1_product_fits_format_specmode()

