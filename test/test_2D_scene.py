import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import astropy.units as u
import proper
import cgisim
from corgisim.scene import Scene, SimulatedImage

import pytest
import numpy as np
from unittest.mock import patch
from scipy.ndimage import rotate as scipy_rotate
 

DISK_VS_POINT_FLUX_RTOL = 0.10 # a few percent difference is expected.

# TODO - we need to change the tests in here due to the change of definitions
@pytest.fixture
def optics_env():
    with patch('proper.prop_run_multi') as prop_run, \
         patch('cgisim.cgisim_read_mode') as read_mode, \
         patch('cgisim.cgisim_roman_throughput') as throughput, \
         patch('cgisim.lib_dir', '/no/such/path'):

        # pretend PROPER gives us a 3×48×48 PRF cube
        prop_run.return_value = (np.ones((3, 48, 48)), {})

        # pretend cgisim_read_mode reports 3 wavelengths around 0.575 µm
        read_mode.return_value = (
            {'sampling_lamref_div_D': 2.0,
             'lamref_um': 0.575,
             'owa_lamref': 9.0,
             'sampling_um': 12.5},
            {'lam0_um': 0.575,
             'nlam': 3,
             'minlam_um': 0.55,
             'maxlam_um': 0.60}
        )

        # pretend throughput is constant 50% over 5500–6000 Å
        throughput.return_value = (
            np.linspace(5500, 6000, 100),
            np.full(100, 0.5)
        )

        yield prop_run, read_mode, throughput

@pytest.fixture
def basic_optics(optics_env):
    """Create a CorgiOptics instance using those stubs."""
    from corgisim.instrument import CorgiOptics

    keywords = {
        'cor_type':        'hlc',
        'polaxis':         10,
        'output_dim':      48,
        'use_errors':      1,
        'use_dm1':         1,
        'use_dm2':         1,
        'use_fpm':         1,
        'use_lyot_stop':   1,
        'use_field_stop':  1,
        'fsm_x_offset_mas': 0.0,
        'fsm_y_offset_mas': 0.0
    }

    optics = CorgiOptics('excam', '1', optics_keywords=keywords)
    optics.quiet = True
    return optics 

def test_build_radial_grid():
    """Test radial grid construction."""
    from corgisim.convolution import build_radial_grid
    
    radii, param = build_radial_grid(iwa=2.0, owa=8.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)
    
    assert isinstance(radii, np.ndarray)
    assert radii[0] == 0.0
    assert len(radii) > 0
    assert isinstance(param, dict)
    
    # Test error cases
    with pytest.raises(ValueError):
        build_radial_grid(iwa=-1, owa=8.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)
    
    with pytest.raises(ValueError):
        build_radial_grid(iwa=8.0, owa=2.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)

def test_build_azimuth_grid():
    """Test azimuth grid creation."""
    from corgisim.convolution import build_azimuth_grid

    azimuths, param = build_azimuth_grid(90.0)
        
    # Test error cases
    with pytest.raises(ValueError):
        build_azimuth_grid(-90.0)
    
    with pytest.raises(ValueError):
        build_azimuth_grid(37.0)  # Doesn't divide 360

def test_create_wavelength_grid_and_weights():
    """Test wavelength grid creation."""
    from corgisim.prf_simulation import create_wavelength_grid_and_weights
    
    # Test error cases
    with pytest.raises(ValueError):
        create_wavelength_grid_and_weights([1.0, 1.2], [-0.5, 1.0])  # Negative SED
    
    with pytest.raises(ValueError):
        create_wavelength_grid_and_weights([1.0, 1.2], [1.0])  # Length mismatch

def test_get_valid_positions():
    """Test polar position validation."""
    from corgisim.convolution import get_valid_polar_positions
    
    radii = [0.0, 1.0, 2.0]
    azimuths = [0.0, 90.0] * u.deg

    positions = get_valid_polar_positions(radii, azimuths)
    
    # 1+ on-axis position + 4 off-axis positions should be included: (1,0°), (1,90°), (2,0°), (2,90°)
    assert len(positions) == 5  # 1 on-axis + 4 off-axis (2 radii * 2 azimuths)
        
    # Verify expected positions are present
    assert (0.0, 0.0 * u.deg) in positions
    assert (1.0, 0.0 * u.deg) in positions
    assert (1.0, 90.0 * u.deg) in positions
    assert (2.0, 0.0 * u.deg) in positions
    assert (2.0, 90.0 * u.deg) in positions

def test_pixel_to_polar():
    """Test pixel coordinate conversion."""
    from corgisim.convolution import pixel_to_polar
    
    r_lamD, theta_deg = pixel_to_polar((21, 21), pix_scale_mas=21.8, res_mas=100.0)

    # Test the key coordinate system property: center should be at origin
    assert r_lamD[10, 10] == 0.0
    
    # Test error case
    with pytest.raises(ValueError):
        pixel_to_polar((10, 10), pix_scale_mas=-1.0, res_mas=100.0)

def test_resize_prf_cube():
    """Test PRF cube resizing."""
    from corgisim.prf_simulation import resize_prf_cube
    
    # Create PRF with a point at center to test centering logic
    prf_cube = np.zeros((1, 10, 10))
    prf_cube[0, 4, 4] = 1.0  # Center point
    
    # Test that center is preserved when resizing
    resized = resize_prf_cube(prf_cube, (20, 20))
    center_idx = np.unravel_index(np.argmax(resized[0]), resized[0].shape)
    # Center should be approximately preserved (allowing for rounding)
    assert abs(center_idx[0] - 9.5) <= 1
    assert abs(center_idx[1] - 9.5) <= 1

def test_nearest_id_map():
    """Test nearest neighbor ID mapping."""
    from corgisim.convolution import nearest_id_map
    
    r_lamD = np.array([[0, 1], [1, 2]])
    theta_deg = np.array([[0, 0], [90, 90]])
    radii_lamD = [0, 1, 2]
    azimuths_deg = [0, 90]
    
    prf_ids = nearest_id_map(r_lamD, theta_deg, radii_lamD, azimuths_deg)
        
    # Test that all IDs are valid indices
    assert np.all(prf_ids >= 0)
    assert np.all(prf_ids < len(radii_lamD) * len(azimuths_deg))

def test_bilinear_indices_weights():
    """Test bilinear interpolation setup."""
    from corgisim.convolution import bilinear_indices_weights
    
    r_lamD = np.array([[0.5]])
    theta_deg = np.array([[45]])
    radii_lamD = np.array([0, 1])  # Convert to numpy array
    azimuths_deg = [0, 90]
    
    indices, weights = bilinear_indices_weights(r_lamD, theta_deg, radii_lamD, azimuths_deg)
    
    # Test the key bilinear property: weights sum to 1
    total_weight = sum(w[0, 0] for w in weights)
    assert np.isclose(total_weight, 1.0)
    
    # Test that all indices are valid
    all_indices = [idx[0, 0] for idx in indices]
    assert all(idx >= 0 for idx in all_indices)

def test_convolve_with_prfs_basic():
    """Test basic convolution functionality."""
    from corgisim.convolution import _convolve_with_prfs
    
    # Simple scene
    obj = np.zeros((21, 21))
    obj[10, 10] = 1.0
    
    # Simple PRF cube
    radii_lamD = np.array([0, 1.0])  # Convert to numpy array
    azimuths_deg = [0, 180] * u.deg
    prfs = np.zeros((3, 21, 21))
    prfs[0, 10, 10] = 1.0  # Delta function PRFs
    prfs[1, 10, 10] = 1.0
    prfs[2, 10, 10] = 1.0
    
    result = _convolve_with_prfs(obj, prfs, radii_lamD, azimuths_deg, 
                               pix_scale_mas=21.8, res_mas=100.0)
    
    assert result.shape == obj.shape
    assert np.isfinite(result).all()
    assert np.sum(result) > 0  # Should produce some output

def test_make_prf_cube(basic_optics):
    """Test PRF cube generation."""
    with patch('corgisim.prf_simulation.create_wavelength_grid_and_weights') as mock_wl_grid, \
         patch('corgisim.convolution.get_valid_polar_positions') as mock_valid_pos, \
         patch('corgisim.outputs.create_hdu') as mock_hdu, \
         patch('corgisim.prf_simulation.compute_single_off_axis_psf') as mock_compute:

        mock_wl_grid.return_value = (np.array([0.575]), np.array([1.0]))

        # Return the correct 5 positions that get_valid_polar_positions actually returns
        # For radii=[0,1,2] and azimuths=[0,90], valid positions are:
        # (0,0°), (1,0°), (1,90°), (2,0°), (2,90°) = 5 total

        mock_valid_pos.return_value = [(0.0, 0 * u.deg), (1.0, 0 * u.deg), (1.0, 90 * u.deg), 
                                                 (2.0, 0 * u.deg), (2.0, 90 * u.deg)]

        # Mock PSF computation
        mock_compute.return_value = np.ones((48, 48))

        # Mock HDU creation
        mock_hdu_obj = Mock()
        mock_hdu_obj.writeto = Mock()
        mock_hdu.return_value = mock_hdu_obj

        radii = [0, 1.0, 2.0]
        azimuths = [0, 90] * u.deg

        from corgisim.prf_simulation import make_prf_cube
        prf_dict = {'test': 'data'}
        
        result = make_prf_cube(basic_optics, radii, azimuths, prf_dict, source_sed=None, output_dir='/tmp', overwrite=True)

        # Verify the workflow
        assert mock_wl_grid.called
        assert mock_valid_pos.called
        assert mock_compute.call_count == 5
        assert mock_hdu.called
        assert mock_hdu_obj.writeto.called
        assert result == mock_hdu_obj
        
        # Verify the PRF cube has correct shape
        call_args = mock_hdu.call_args
        prf_cube_arg = call_args[0][0]
        assert prf_cube_arg.shape == (5, 48, 48)

def test_full_pipeline():
    """Test complete pipeline from grid to convolution."""
    from corgisim.convolution import (build_radial_grid, build_azimuth_grid, 
                                     get_valid_polar_positions, _convolve_with_prfs)
    
    # Build grids
    radii, _ = build_radial_grid(2.0, 6.0, 0.5, 1.0, 2.0)
    azimuths, _ = build_azimuth_grid(90.0)
    positions = get_valid_polar_positions(radii, azimuths)
    
    # Create test scene and PRFs
    scene = np.zeros((25, 25))
    scene[12, 12] = 1.0
    
    n_prfs = len(positions)
    prfs = np.zeros((n_prfs, 25, 25))
    for i in range(n_prfs):
        prfs[i, 12, 12] = 1.0
    
    # Convolve
    result = _convolve_with_prfs(scene, prfs, radii, azimuths, 21.8, 100.0)
    
    assert result.shape == scene.shape
    assert np.sum(result) > 0

def test_single_prf_edge_case():
    """Test convolution with single on-axis PRF."""
    from corgisim.convolution import _convolve_with_prfs
    
    # Simple point source
    scene = np.zeros((15, 15))
    scene[7, 7] = 5.0
    
    # Single delta-function PRF at origin
    prf_cube = np.zeros((1, 15, 15))
    prf_cube[0, 7, 7] = 1.0
    
    result = _convolve_with_prfs(scene, prf_cube, [0], [0] * u.deg,
                               pix_scale_mas=21.8, res_mas=100.0)
    
    # Should preserve scene exactly (within floating point precision)
    assert np.allclose(result, scene, rtol=1e-12)

def test_simulate_2d_scene_missing_prf_path():
    from corgisim.instrument import CorgiOptics
    """Test error handling when PRF cube path is missing."""
    mock_scene = Mock()
    mock_scene.twoD_scene_info = {
        "disk_model_path": "/fake/path/to/disk_model.fits"
    }

    mock_optics = Mock()
    mock_optics.res_mas = 1.0
    mock_optics.cgi_mode = "excam_efield"  

    with pytest.raises(ValueError, match=r"No PRF cube path provided in the input:"):
        CorgiOptics.simulate_2d_scene(mock_optics, mock_scene, prf_cube_path=None)

def _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle):
    
    """Run simulate_2d_scene_disk_roll on disk and return the disk array it passes to convolution.
 
    simulate_2d_scene loads a disk model from a FITS file, optionally rotates it by
    the telescope roll angle, normalises the total flux, and then passes the results to
    to _convolve_with_prfs as the ``obj`` keyword argument.
    
    Args:
        disk (np.ndarray): 2-D disk brightness array in the sky frame, before
            any roll-angle rotation is applied.
        roll_angle (float): Telescope roll angle in degrees.  
 
    Returns:
        np.ndarray: The 2-D ``obj`` array that simulate_2d_scene passed to
        _convolve_with_prfs — the disk after roll-angle rotation and flux
        normalisation, ready for convolution with the PRF cube.
    """

    from corgisim.instrument import CorgiOptics

    # providing dict for prf sim info
    prf_sim_info = {
    'centred': 'True',
    'iwa': 2.0, 'owa': 8.0,
    'inner_step': 0.5, 'mid_step': 1.0, 'outer_step': 2.0,
    'max_radius': 10.0, 'step_deg': 90.0,
    }

    mock_scene = Mock()
    mock_scene.twoD_scene_info = {"disk_model_path": "/fake/path/to/disk_model.fits", 
                                  "surface_brightness_unit": "MJy/sr",}

    mock_optics = Mock()
    mock_optics.res_mas = 100.0
    mock_optics.cgi_mode = "excam"
    mock_optics.roll_angle = roll_angle

    def mock_getdata(path, *args, **kwargs):
        """Return disk for disk-model loads; a valid-shaped PRF cube otherwise."""
        if 'disk' in str(path).lower():
            return disk
        return np.zeros((45, disk.shape[0], disk.shape[1]))

    with patch('corgisim.instrument.fits.getdata', side_effect=mock_getdata), \
         patch('corgisim.instrument.disk_countrate.disk_to_countrate', return_value=disk.copy()), \
         patch('corgisim.prf_simulation._get_prf_sim_info', return_value=prf_sim_info), \
         patch('corgisim.convolution._convolve_with_prfs') as mock_conv, \
         patch('corgisim.convolution.flux_calibration_2D_scene',
         side_effect=lambda optics, scene, conv2d: conv2d), \
         patch('corgisim.convolution._set_2D_image_sim_info', return_value={}), \
         patch('corgisim.outputs.create_hdu', side_effect=lambda data, sim_info: data):

        mock_conv.side_effect = lambda obj, **kwargs: obj
        CorgiOptics.simulate_2d_scene(mock_optics, mock_scene,
                  prf_cube_path="/fake/path/to/prf_cube.fits")
    return mock_conv.call_args.kwargs['obj']

def test_disk_and_point_source_rotate_in_same_direction():
    """checking disk rotation in _simulate_2d_scene_disk_roll_before_convolution agrees in direction with skycoord_to_excamcoord.
 
    generates a disk with a single bright pixel to the right of centre (3 pixel to the right) in the disk image
    (dRA < 0, dDec = 0 in sky coordinates), then finds using skycoord_to_excamcoord where
    a point source at the same sky position would land on the detector after a 90 deg
    roll.  The disk peak after __simulate_2d_scene_disk_roll_before_convolution
    must land at the same pixel.
    """
    from corgisim.instrument import skycoord_to_excamcoord
 
    N = 11
    cy = cx = N // 2  # centre pixel: (5, 5)
    dcol = 3          # feature is 3 pixels to the right of centre
 
    disk = np.zeros((N, N))
    disk[cy, cx + dcol] = 1.0  # bright pixel at (5, 8)
 
    # Sky position of that pixel (pixel units stand in for mas here)
    dra  = -float(dcol)  # +col = -RA
    ddec =  0.0
 
    roll_angle = 90.0
 
    # skycoord_to_excamcoord predict this sky position
    excam_dx, excam_dy = skycoord_to_excamcoord(dra, ddec, roll_angle)
    expected_col = round(cx + excam_dx)
    expected_row = round(cy + excam_dy)  
 
    #simulate_2d_scene disk peak location
    rolled_disk = _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle)
    peak_row, peak_col = np.unravel_index(np.argmax(rolled_disk), rolled_disk.shape)

    # assert if the disk roll and skycoord_to_excamcoord peak pixel location are matching
    assert (peak_row, peak_col) == (expected_row, expected_col), (
        f"Disk peak at ({peak_row}, {peak_col}) but skycoord_to_excamcoord "
        f"predicts ({expected_row}, {expected_col}). "
    )

def test_roll_angle_rotates_disk_model():

    """Roll angle moves a disk feature off its original axis in the expected way.
 
    Three sub-cases are checked:
 
    roll_angle == 0
        The disk must not rotate. The bright pixel must still be at the same location after the 0 rotation.
 
    roll_angle == +90
        CW rotation in the detector frame moves the feature that was above centre to the right: expected peak (4, 6). 
 
    roll_angle == -90
       CCW rotation moves the feature to the left: expected peak (4, 2).
    """

    disk = np.zeros((9, 9))
    disk[6, 4] = 1.0  # bright pixel 2 rows below centre (centre is row=4, col=4)

    # ------------------------------------------------------------------ roll=0
    unrotated = _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle=0.0)
    assert np.unravel_index(np.argmax(unrotated), unrotated.shape) == (6, 4), (
    "roll_angle=0 should leave the disk unchanged"
)

    # ----------------------------------------------------------------- roll=+90
    rotated_90_pos = _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle=90.0)
    assert rotated_90_pos.shape == disk.shape, (
        f"rotation changed the array size: {rotated_90_pos.shape} vs {disk.shape} "
        "- the padded array was probably not cropped back"
    )
    row_pos, col_pos = np.unravel_index(np.argmax(rotated_90_pos), rotated_90_pos.shape)

    assert (row_pos, col_pos) == (4, 6)

    # ----------------------------------------------------------------- roll=-90
    rotated_90_neg = _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle=-90.0)
    assert rotated_90_neg.shape == disk.shape, (
        f"rotation changed the array size: {rotated_90_neg.shape} vs {disk.shape}"
    )
    row_neg, col_neg = np.unravel_index(np.argmax(rotated_90_neg), rotated_90_neg.shape)

    assert (row_neg, col_neg) == (4, 2)

# testing the normalization
@pytest.mark.parametrize("roll_angle", [0.0, 30.0, -30.0, 45.0, -45.0])
def test_disk_model_is_normalised_after_rotation(roll_angle):
    """The normalized disk flux before the convolution sums to 1 at every roll angle. 
    checking to see if there are nans or inf from the rolling
    """
    ring = np.random.default_rng(42)
    #disk with random values
    disk = ring.random((21, 21))
 
    #rolls the disk using the function
    rolled_disk = _simulate_2d_scene_disk_roll_before_convolution(disk, roll_angle)
 
    # check if any of the value is nan, inf in the rolled disk
    assert np.isfinite(rolled_disk).all()
    # fails if normalization is done before rolling or normalization is not handled correctly
    assert np.isclose(np.sum(rolled_disk), np.sum(disk), rtol=1e-4)
    with pytest.raises(NotImplementedError, match=r"Roll angle rotation is not implemented yet for 2D scene."):
        CorgiOptics.simulate_2d_scene(mock_optics, mock_scene, prf_cube_path=mock_scene.twoD_scene_info["prf_cube_path"])

@pytest.mark.parametrize("offset_pix", [(0, 0), (0, 8)])
def test_one_pixel_disk_matches_point_source_flux(offset_pix):
    """A one-pixel disk is physically a point source, so the 2D-scene path must
    give the same image as inject_point_sources for the same source.

    2D-scene path (same steps as simulate_2d_scene): a disk model that is zero
    except one pixel holding the contrast (pixel flux / host star flux), converted
    with disk_to_countrate('contrast/pixel') to ct/s, then convolved with the PRF
    cube by _convolve_with_prfs.

    Reference: a point source dmag fainter than the host at the same position,
    from inject_point_sources.

    Checks: total flux within DISK_VS_POINT_FLUX_RTOL, and the same peak pixel.

    offset_pix is (dy, dx) in EXCAM pixels from the frame centre (n // 2):
    (0, 0) is on-axis (on-axis PSF only), (0, 8) is off-axis (PRF cube of the
    on-axis PSF plus the off-axis PSF generated at exactly that pixel).

    Notes
    -----
    - The flux ratio is expected a few percent below 1
    - The point source is given the host spectrum (rescaled to host Vmag + dmag)
      so both paths use the same spectral shape; the default flat spectrum
      would add an extra offset.

    Example output (pytest -s, HLC band 1F, Vmag 8 host, dmag 16):
        [offset (0, 0)] flux point=1.6830e-06  disk=1.6155e-06 ct/s  ratio=0.9599
        peak (y, x) point=(25, 24)  disk=(25, 24)
        [offset (0, 8)] flux point=4.6733e-01  disk=4.4239e-01 ct/s  ratio=0.9466
        peak (y, x) point=(25, 33)  disk=(25, 33)
    """
    import warnings
    import roman_preflight_proper
    from corgisim import constants, instrument, scene
    from corgisim.convolution import _convolve_with_prfs
    from corgisim.disk_countrate import disk_to_countrate
    from corgisim.prf_simulation import (centre_prf_cube, compute_single_off_axis_psf,
                                         create_wavelength_grid_and_weights)

    output_dim = 51
    pix_scale_mas = constants.PIXEL_SCALE_ARCSEC * 1e3
    host_vmag, dmag = 8, 16.0
    contrast = 10 ** (-0.4 * dmag)
    dy, dx = offset_pix
    c = output_dim // 2

    host_star_properties = {'Vmag': host_vmag, 'spectral_type': 'G0V', 'magtype': 'vegamag'}
    host_spectrum = scene.Scene(host_star_properties).stellar_spectrum

    # Point source at the disk pixel (EXCAM +x = -RA, +y = +Dec at roll 0)
    point_source_info = [{'Vmag': host_vmag + dmag, 'magtype': 'vegamag',
                          'position_x': -dx * pix_scale_mas, 'position_y': dy * pix_scale_mas,
                          'Custom_Spectrum': host_spectrum, 'Rescale_Custom_Spectrum': True}]
    twoD_scene_info = {'disk_model_path': None, 'surface_brightness_unit': 'contrast/pixel'}
    base_scene = scene.Scene(host_star_properties, point_source_info=point_source_info,
                             twoD_scene_info=twoD_scene_info)

    rootname = 'hlc_ni_3e-8'
    dm1 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')
    optics_keywords = {'cor_type': 'hlc', 'use_errors': 2, 'polaxis': 10,
                       'output_dim': output_dim,
                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
                       'use_fpm': 1, 'use_lyot_stop': 1, 'use_field_stop': 1}
    optics = instrument.CorgiOptics('excam', '1F', optics_keywords=optics_keywords, if_quiet=True)

    # 1. Point source (ct/s). Ignore the "outside coronagraph FOV" warning on-axis.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        point_img = optics.inject_point_sources(base_scene).point_source_image.data.astype(float)

    # 2. PRF cube: on-axis PSF, plus the off-axis PSF generated at the disk pixel
    r_lamD = np.hypot(dx, dy) * pix_scale_mas / optics.res_mas
    az = (np.degrees(np.arctan2(dy, dx)) % 360) * u.deg
    wvl, wts = create_wavelength_grid_and_weights(optics.lam_um, None)
    on_axis = compute_single_off_axis_psf(optics, 0.0, 0 * u.deg, wvl, wts).astype(float)
    if r_lamD == 0:
        cube, radii, azimuths, positions = on_axis[np.newaxis], np.array([0.0]), [0.0] * u.deg, [(0.0, 0 * u.deg)]
    else:
        off_axis = compute_single_off_axis_psf(optics, r_lamD, az, wvl, wts).astype(float)
        cube = np.array([on_axis, off_axis])
        radii, azimuths = np.array([0.0, r_lamD]), [az.value] * u.deg
        positions = [(0.0, 0 * u.deg), (r_lamD, az)]
    cube = centre_prf_cube(cube, method='source_position', positions=positions,
                           res_mas=optics.res_mas, pix_scale_mas=pix_scale_mas)

    # 3. One-pixel disk at the point-source position, contrast/pixel -> ct/s
    #   (each pixel x the host star's count rate through the bandpass)
    disk = np.zeros((output_dim, output_dim))
    disk[c + dy, c + dx] = contrast
    disk_countrate = disk_to_countrate(disk, 'contrast/pixel', bandpass=optics.bp, area_cm2=optics.area, pixel_scale_arcsec=constants.PIXEL_SCALE_ARCSEC, wavelength_micron=optics.lam0_um, return_quantity=False,input_scene=base_scene)

    # 4. Convolve with the PRF cube
    disk_img = _convolve_with_prfs(disk_countrate, cube, radii, azimuths, pix_scale_mas=pix_scale_mas, res_mas=optics.res_mas, interpolate_prfs=True)

    # 5. Compare the two images (both ct/s, same pixel grid)
    flux_ratio = disk_img.sum() / point_img.sum()
    peak_point = np.unravel_index(np.argmax(point_img), point_img.shape)
    peak_disk = np.unravel_index(np.argmax(disk_img), disk_img.shape)

    # Shown with pytest -s
    print(f"\n[offset {offset_pix}] flux point={point_img.sum():.4e}  disk={disk_img.sum():.4e} ct/s  "
          f"ratio={flux_ratio:.4f}")
    print(f"peak (y, x) point={peak_point}  disk={peak_disk}")

    # Flux: total count rate over the frame
    assert abs(flux_ratio - 1) <= DISK_VS_POINT_FLUX_RTOL, \
        f"flux ratio disk/point = {flux_ratio:.4f} (allowed ±{DISK_VS_POINT_FLUX_RTOL:.0%})"

    # Shift: same brightest pixel (off-axis the PSF core, on-axis the brightest speckle)
    assert peak_disk == peak_point, f"peak pixel disk={peak_disk}, point={peak_point}"
