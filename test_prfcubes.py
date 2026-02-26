import numpy as np 
import pytest 
import astropy.units as u



from corgisim import CorgiScene
return CorgiScene.from_twoD_image(np.ones((100, 100)) * u.Jy)

@pytest.fixture
def dummy_scene():

    #TODO - this should be a fixture that return a CorgiOptics object





    return optics


@pytest.fixture
def dummy_instance():
    obj = YourClass()
    obj.grid_dim_out = 51
    obj.lam0_um = 1.0
    obj.lam_um = np.array([1.0])  # single wavelength
    obj.nlam = 1
    obj.quiet = True
    obj.proper_keywords = {}  # minimal for test
    return obj

def test_prf_cube_shape(dummy_instance):
    r = [1.0]
    th = [45 * u.deg]
    prf_cube = dummy_instance.make_prf_cube(r, th)
    assert prf_cube.shape == (1, 51, 51)

def test_empty_inputs(dummy_instance):
    prf_cube = dummy_instance.make_prf_cube([], [])
    assert prf_cube.shape[0] == 0  # should be (0, Ny, Nx)

def test_multiple_positions(dummy_instance):
    r = [1.0, 2.0]
    th = [0 * u.deg, 90 * u.deg]
    prf_cube = dummy_instance.make_prf_cube(r, th)
    assert prf_cube.shape == (4, 51, 51)

def test_azimuth_unit_handling(dummy_instance):
    r = [1.0]
    th = [0, 90] * u.deg
    prf_cube = dummy_instance.make_prf_cube(r, th)
    assert prf_cube.shape == (2, 51, 51)

def test_custom_sed(dummy_instance):
    r = [1.0]
    th = [45 * u.deg]
    # simulate 3 wavelengths and custom source SED
    dummy_instance.lam_um = np.array([0.9, 1.0, 1.1])
    dummy_instance.nlam = 3
    sed = np.array([0.2, 0.5, 0.3])
    prf_cube = dummy_instance.make_prf_cube(r, th, bandwidth_frac=0.2, source_sed=sed)
    assert prf_cube.shape == (1, 51, 51)
    assert np.all(prf_cube >= 0)



# # 1. normalise
# psf_mono = prf_cube_mono[0]
# psf_band = prf_cube_flat[0]
# mono = psf_mono / psf_mono.sum()
# band = psf_band / psf_band.sum()

# # 2. radial profiles
# r = np.hypot(*np.indices(mono.shape) - np.array(mono.shape)[:,None,None]/2)
# bins = np.arange(0, r.max(), 0.5)                 # pixel bins
# rad_mono, _ = np.histogram(r, bins, weights=mono)
# rad_band, _ = np.histogram(r, bins, weights=band)
# rad_mono /= np.histogram(r, bins)[0]
# rad_band /= np.histogram(r, bins)[0]

# # 3. encircled energy
# ee_mono = np.cumsum(rad_mono) / mono.sum()
# ee_band = np.cumsum(rad_band) / band.sum()

# # 4. RMS difference
# diff_rms = np.sqrt(((band - mono)**2).mean())

# print("RMS difference between mono and band PRFs:", diff_rms)


#  plt.imshow(np.log10(np.abs(band - mono) / mono.max()))

# plt.plot(bins[:-1], np.log10(rad_mono), label="Mono PRF (log10 scale)")
# plt.plot(bins[:-1], np.log10(rad_band), label="Band PRF (log10 scale)")
# plt.xlabel("Radius (pixels)")
# plt.ylabel("Log10 Flux")
# plt.legend()
# plt.show()



def verify_center_preservation(original_prf, resized_prf, target_shape):
    """
    Helper function to verify that PSF center is preserved after resizing.
    
    Returns the shift in center position (should be close to 0,0).
    """
    # Find center of mass for verification
    orig_center_y, orig_center_x = ndimage.center_of_mass(original_prf[0])  # First PSF
    resized_center_y, resized_center_x = ndimage.center_of_mass(resized_prf[0])
    
    # Expected centers
    orig_expected_y = (original_prf.shape[1] - 1) / 2.0
    orig_expected_x = (original_prf.shape[2] - 1) / 2.0
    target_expected_y = (target_shape[0] - 1) / 2.0
    target_expected_x = (target_shape[1] - 1) / 2.0
    
    print(f"Original center: ({orig_center_y:.2f}, {orig_center_x:.2f}), expected: ({orig_expected_y:.2f}, {orig_expected_x:.2f})")
    print(f"Resized center:  ({resized_center_y:.2f}, {resized_center_x:.2f}), expected: ({target_expected_y:.2f}, {target_expected_x:.2f})")
    
    center_shift_y = resized_center_y - target_expected_y
    center_shift_x = resized_center_x - target_expected_x
    
    return center_shift_y, center_shift_x