from corgisim import scene
from corgisim import instrument
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import pytest
import cgisim
from synphot.models import BlackBodyNorm1D, Box1D
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.units import validate_wave_unit, convert_flux, VEGAMAG
from astropy import units as u


#@pytest.mark.parametrize("interp_method", ['linear', 'cubic'])
def test_off_axis_source_spectrum():
    print('Test if the off axis flat spectrum is correct by comparing with analitical model')

    mag = 18
    #comp_mag = 18
    sptype = ['M5V','M0V','K5V','K0V','G0V','F5V','B3V','A5V','A0V']

    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'

    fig1= plt.figure(figsize=(15,12))

    for i in range(len(sptype)):
        cgisim_lam, cgisim_sp = cgisim.cgisim_read_spectrum(sptype[i].lower(), info_dir ) 
        cgisim_sp_scale = cgisim.cgisim_renormalize_spectrum( cgisim_lam, cgisim_sp, mag, 'V', info_dir )
        
    
        #Define the host star properties
        host_star_properties = {'Vmag': mag, 'spectral_type': sptype[i], 'magtype': 'vegamag'}
        point_source_info = [{'Vmag': mag, 'magtype': 'vegamag','position_x':4 , 'position_y':4}]

        #Create a Scene object that holds all this information
        base_scene = scene.Scene(host_star_properties, point_source_info)

        #sp=base_scene.stellar_spectrum
        #sp2=sp(cgisim_lam ).value
        pp = base_scene.off_axis_source_spectrum[0] 
        
        pp_test = pp(550).value
        analitical_value = 952.3* 10**(-0.4 * (mag-0.03))
        #print(pp_test,analitical_value )

        # Use pytest.approx to check similarity within a tolerance
        assert  pp_test  == pytest.approx(analitical_value, abs=1e-5)

        print('Pass Test')
       
  
        #ax=plt.subplot(3,3,i+1)
        #sp.plot(ax=ax)
        #pp2 = pp(cgisim_lam ).value
        #ax.plot(cgisim_lam, pp2,label='corgisim')
        #ax.plot(cgisim_lam, cgisim_sp_scale,label='cgisim')
        #ax.set_title(sptype[i])
        #ax.set_xlabel('wavelength (A)')
        #ax.set_ylabel('flux (photons/s/cm^2/A)')

        #plt.legend(loc='upper right')
        #plt.subplots_adjust(hspace=0.5, wspace=0.3)

    #plt.show()
    

def test_custom_spectrum_rescaled():
    """Test that Custom_Spectrum with Rescale_Custom_Spectrum=True is normalized to the target V magnitude."""
    print('Test that a custom spectrum is correctly rescaled to the target V-band magnitude')

    Vmag_target = 22.0
    host_star_properties = {'Vmag': 6.0, 'spectral_type': 'G2V', 'magtype': 'vegamag'}

    # Use a cool blackbody as a stand-in for a planet emission spectrum
    sp_custom = SourceSpectrum(BlackBodyNorm1D, temperature=1200)

    point_source_info = [
        {
            'Vmag': Vmag_target,
            'magtype': 'vegamag',
            'position_x': 100.0,
            'position_y': 100.0,
            'Custom_Spectrum': sp_custom,
            'Rescale_Custom_Spectrum': True,
        }
    ]

    base_scene = scene.Scene(host_star_properties, point_source_info)
    sp_result = base_scene.off_axis_source_spectrum[0]

    # Measure V-band magnitude of the rescaled output spectrum
    vega_spec = SourceSpectrum.from_vega()
    v_band = SpectralElement.from_filter('johnson_v')
    obs = Observation(sp_result, v_band)
    vmag_measured = obs.effstim(VEGAMAG, vegaspec=vega_spec).value

    assert vmag_measured == pytest.approx(Vmag_target, abs=0.01)
    print('Pass Test')


def test_custom_spectrum_no_rescale():
    """Test that Custom_Spectrum with Rescale_Custom_Spectrum=False is returned as-is (no flux scaling)."""
    print('Test that a custom spectrum is returned unchanged when Rescale_Custom_Spectrum=False')

    Vmag_target = 22.0
    host_star_properties = {'Vmag': 6.0, 'spectral_type': 'G2V', 'magtype': 'vegamag'}

    sp_custom = SourceSpectrum(BlackBodyNorm1D, temperature=1200)

    point_source_info = [
        {
            'Vmag': Vmag_target,
            'magtype': 'vegamag',
            'position_x': 100.0,
            'position_y': 100.0,
            'Custom_Spectrum': sp_custom,
            'Rescale_Custom_Spectrum': False,
        }
    ]

    base_scene = scene.Scene(host_star_properties, point_source_info)
    sp_result = base_scene.off_axis_source_spectrum[0]

    # Verify flux is unchanged at several wavelengths spanning the V band
    for wave_aa in [4000, 5500, 7000]:
        wave = wave_aa * u.AA
        assert sp_result(wave).value == pytest.approx(sp_custom(wave).value, rel=1e-6)

    print('Pass Test')


if __name__ == '__main__':
    test_off_axis_source_spectrum()
    test_custom_spectrum_rescaled()
    test_custom_spectrum_no_rescale()
  