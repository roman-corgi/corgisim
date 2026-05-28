from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
import eetc
from synphot.models import  Empirical1D
from synphot import units, SourceSpectrum, SpectralElement
from synphot.units import VEGAMAG
from astropy import units as u
import os

#@pytest.mark.parametrize("interp_method", ['linear', 'cubic'])
def test_stellar_spectrum():

    mag = 5
    sptype_list = ['M5V','M0V','G5V','K0V','G0V','F5V','B3V','A5V','A0V']
    eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
    atlas_dir = os.path.join(eetc_path, 'flux_grid_generation', 'bpgs_atlas_csv')
    
    for sptype in sptype_list:
        filename = sptype + '.txt'
        spectrum_file = os.path.join(atlas_dir, filename)

        data = np.loadtxt(spectrum_file, delimiter=',', comments='#')
        wavelength_angstrom = data[:, 0]
        flux = data[:, 1]

        v_band = SpectralElement.from_filter('johnson_v')
        vega_spec = SourceSpectrum.from_vega()

        # Create a synphot SourceSpectrum using Empirical1D model
        sp_eetc = SourceSpectrum(
            Empirical1D,
            points=wavelength_angstrom * u.AA,
            lookup_table=flux * units.FLAM
        )
        sp_eetc_norm = sp_eetc.normalize(renorm_val=mag * VEGAMAG, band=v_band, vegaspec=vega_spec)



    
        #Define the host star properties
        host_star_properties = {'Vmag': mag, 'spectral_type': sptype, 'magtype': 'vegamag'}

        #Create a Scene object that holds all this information
        base_scene = scene.Scene(host_star_properties)

        sp=base_scene.stellar_spectrum
        for wave in sp.waveset:
            assert sp(wave)==sp_eetc_norm(wave), f"Values differ from two methods on sptype {sptype}"

if __name__ == '__main__':
    test_stellar_spectrum()
  