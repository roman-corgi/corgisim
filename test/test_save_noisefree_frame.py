import corgisim
from corgisim import scene, instrument, outputs, inputs, observation
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import os, shutil
import glob

def test_save_noisefree_frame():
    
    Vmag = 8
    sptype = 'G0V'
    cgi_mode = 'excam'
    bandpass_corgisim = '1F'
    cor_type = 'hlc_band1'

    
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype': 'vegamag'}
    
    base_scene = scene.Scene(host_star_properties)

    cases = ['3e-8']       
    rootname = 'hlc_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':0, 'output_dim':51,\
                    'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)
    sim_scene = optics.get_host_star_psf(base_scene)

    ### save the L1 product fits file to test/testdata folder
    local_path = corgisim.lib_dir
    outdir = os.path.join(local_path.split('corgisim')[0], 'corgisim/test/testdata')
    outputs.save_hdu_to_fits(sim_scene.host_star_image,outdir=outdir, write_as_L1=False,
    filename='test_noisefree_frame.fits', overwrite=True)

    ### delete file after testing
    f = os.path.join(outdir, 'test_noisefree_frame.fits')
    print('Deleted the FITS file after testing')
    os.remove(f)


if __name__ == '__main__':
    test_save_noisefree_frame()