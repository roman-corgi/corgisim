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
        cgisim_lam, cgisim_sp = cgisim.cgisim_read_spectrum(sptype[i], info_dir ) 
        cgisim_sp_scale = cgisim.cgisim_renormalize_spectrum( cgisim_lam, cgisim_sp, mag, 'V', info_dir )
        
    
        #Define the host star properties
        host_star_properties = {'Vmag': mag, 'spectral_type': sptype[i], 'magtype': 'vegamag'}
        point_source_info = {'Vmag': mag, 'magtype': 'vegamag','position_x':4 , 'position_y':4}

        #Create a Scene object that holds all this information
        base_scene = scene.Scene(host_star_properties, point_source_info)

        #sp=base_scene.stellar_spectrum
        #sp2=sp(cgisim_lam ).value
        pp = base_scene.off_axis_source_spectrum 
        
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
    

if __name__ == '__main__':
    test_off_axis_source_spectrum()
  