#import necessary packages
from corgisim import scene, instrument, prf_simulation, convolution
import matplotlib.pyplot as plt
import proper
import roman_preflight_proper
import astropy.io.fits as fits

def main():
    roman_preflight_proper.copy_here()

    bandpass = '1'
    # define coronagraph properties

    # --- Coronagraph / bandpass selection ---

    if bandpass == '1':
        # HLC band 1F (narrow field)
        bandpass_corgisim = "1F"
        cor_type = "hlc"
        cases = ["3e-8"]
        rootname = f"hlc_ni_{cases[0]}"
        output_dim = 51
    elif bandpass == '4':
        # SPC-wide band 4 (wide field)
        bandpass_corgisim = "4F"
        cor_type = "spc-wide"
        cases = ["2e-8"]
        rootname = f"spc-wide_ni_{cases[0]}"
        output_dim = 201

    cgi_mode = 'excam'

    # --- Load DM solutions ---
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

    optics_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':10, 'output_dim':output_dim,\
                        'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1, 'oversample_factor': 2, 'NCPUS':7}

    # --- Optics keywords (same as Tutorial 1) ---
    optics = instrument.CorgiOptics(cgi_mode, bandpass_corgisim, optics_keywords=optics_keywords, if_quiet=True)

    # --- Define the PRF sampling grid ---
    # Radial sampling (in λ/D)
    radii_lamD, radii_param = convolution.build_radial_grid(
        iwa=3,          # inner working angle (λ/D)
        owa=7,         # outer working angle (λ/D)
        max_radius=12,
        inner_step=3,   # step inside IWA (λ/D)
        mid_step=.5,     # step between IWA and OWA (λ/D)
        outer_step=1,   # step beyond OWA (λ/D)
    )

    # Azimuthal sampling (in degrees)
    azimuths_deg, azimuth_param = convolution.build_azimuth_grid(
        step_deg=60    # azimuthal step in degrees
    )
        # --- Metadata: record which DM solution/coronagraph case was used ---
    dm_solution = rootname

    # Build a dictionary describing this PRF cube configuration
    prf_dict = prf_simulation._generate_prf_dictionary(
        radii_param,
        azimuth_param,
        dm_solution
    )

    # Generate the PRF cube
    prf_hdu = prf_simulation.make_prf_cube(
        optics=optics,
        radii_lamD=radii_lamD,
        azimuths_deg=azimuths_deg,
        prf_dict=prf_dict,
        overwrite=True,
        prf_fname="prf_hlc_band_1_psf_small.fits"
    )

if __name__ == "__main__":
    main()
