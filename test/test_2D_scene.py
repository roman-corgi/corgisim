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

    optics = CorgiOptics('excam', '1', proper_keywords=keywords)
    optics.quiet = True
    return optics 

def test_build_radial_grid():
    """Test radial grid construction."""
    from corgisim.convolution import build_radial_grid
    
    radii = build_radial_grid(iwa=2.0, owa=8.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)
    
    assert isinstance(radii, np.ndarray)
    assert radii[0] == 0.0
    assert len(radii) > 0
    
    # Test error cases
    with pytest.raises(ValueError):
        build_radial_grid(iwa=-1, owa=8.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)
    
    with pytest.raises(ValueError):
        build_radial_grid(iwa=8.0, owa=2.0, inner_step=0.5, mid_step=1.0, outer_step=2.0)

def test_build_azimuth_grid():
    """Test azimuth grid creation."""
    from corgisim.convolution import build_azimuth_grid
        
    # Test error cases
    with pytest.raises(ValueError):
        build_azimuth_grid(-90.0)
    
    with pytest.raises(ValueError):
        build_azimuth_grid(37.0)  # Doesn't divide 360

def test_create_wavelength_grid_and_weights():
    """Test wavelength grid creation."""
    from corgisim.convolution import create_wavelength_grid_and_weights
    
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
    
    # Should exclude (0, 90°) but include (0, 0°)
    assert (0.0, 0.0 * u.deg) in positions
    assert (0.0, 90.0 * u.deg) not in positions
    assert len(positions) == 5  # 1 + 2*2

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
    from corgisim.convolution import resize_prf_cube
    
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
    
    # Test the key business logic: center always maps to on-axis PRF (index 0)
    assert prf_ids[0, 0] == 0  # r=0 → on-axis PRF regardless of angle
    
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
    from corgisim.convolution import convolve_with_prfs
    
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
    
    result = convolve_with_prfs(obj, prfs, radii_lamD, azimuths_deg, 
                               pix_scale_mas=21.8, res_mas=100.0)
    
    assert result.shape == obj.shape
    assert np.isfinite(result).all()
    assert np.sum(result) > 0  # Should produce some output

def test_make_prf_cube(basic_optics):
    """Test PRF cube generation."""
    with patch('corgisim.convolution.create_wavelength_grid_and_weights') as mock_create_wavelength_grid_and_weights, \
         patch('corgisim.convolution.get_valid_polar_positions') as get_valid_polar_positions, \
         patch.object(basic_optics, '_compute_single_off_axis_psf') as mock_compute:

        mock_create_wavelength_grid_and_weights.return_value = (np.array([0.575]), np.array([1.0]))
        # FIXED: Return the correct 5 positions that get_valid_polar_positions actually returns
        # For radii=[0,1,2] and azimuths=[0,90], valid positions are:
        # (0,0°), (1,0°), (1,90°), (2,0°), (2,90°) = 5 total
        get_valid_polar_positions.return_value = [(0.0, 0 * u.deg), (1.0, 0 * u.deg), (1.0, 90 * u.deg), 
                                                 (2.0, 0 * u.deg), (2.0, 90 * u.deg)]
        mock_compute.return_value = np.ones((48, 48))

        radii = [0, 1.0, 2.0]
        azimuths = [0, 90] * u.deg
        
        prf_cube = basic_optics.make_prf_cube(radii, azimuths)
        
        assert prf_cube.shape[0] == 5  # 5 valid positions
        assert prf_cube.shape[1:] == (48, 48)
        assert prf_cube.dtype == np.float32

def test_convolve_2d_scene_parameter_validation(basic_optics):
    """Test parameter validation for convolve_2d_scene."""
    
    # Create scene that triggers the validation error we want to test
    scene = Mock()
    scene.twoD_scene_info = {
        'prf_cube_path': None,  # No PRF cube 
        'disk_model_path': '/fake/path/to/disk_model.fits',
        # FIXED: Add the missing keys that the function expects
        'radii_lamD': None,  # Will not be used in Mode 2
        'azimuths_deg': None  # Will not be used in Mode 2
    }
    scene.twoD_scene_spectrum = Mock()
    
    # FIXED: Mock fits.getdata to prevent FileNotFoundError
    with patch('astropy.io.fits.getdata') as mock_fits:
        mock_fits.return_value = np.ones((48, 48))
        
        # Test error when Mode 2 is missing required parameters
        with pytest.raises(ValueError, match="Missing parameters"):
            basic_optics.convolve_2d_scene(scene, owa=9.0)  # Missing other required params

def test_full_pipeline():
    """Test complete pipeline from grid to convolution."""
    from corgisim.convolution import (build_radial_grid, build_azimuth_grid, 
                                     get_valid_polar_positions, convolve_with_prfs)
    
    # Build grids
    radii = build_radial_grid(2.0, 6.0, 0.5, 1.0, 2.0)
    azimuths = build_azimuth_grid(90.0)
    positions = get_valid_polar_positions(radii, azimuths)
    
    # Create test scene and PRFs
    scene = np.zeros((25, 25))
    scene[12, 12] = 1.0
    
    n_prfs = len(positions)
    prfs = np.zeros((n_prfs, 25, 25))
    for i in range(n_prfs):
        prfs[i, 12, 12] = 1.0
    
    # Convolve
    result = convolve_with_prfs(scene, prfs, radii, azimuths, 21.8, 100.0)
    
    assert result.shape == scene.shape
    assert np.sum(result) > 0

def test_single_prf_edge_case():
    """Test convolution with single on-axis PRF."""
    from corgisim.convolution import convolve_with_prfs
    
    # Simple point source
    scene = np.zeros((15, 15))
    scene[7, 7] = 5.0
    
    # Single delta-function PRF at origin
    prf_cube = np.zeros((1, 15, 15))
    prf_cube[0, 7, 7] = 1.0
    
    result = convolve_with_prfs(scene, prf_cube, [0], [0] * u.deg,
                               pix_scale_mas=21.8, res_mas=100.0)
    
    # Should preserve scene exactly (within floating point precision)
    assert np.allclose(result, scene, rtol=1e-12)

def test_zero_radius_position_filtering():
    """Test that (r=0, θ≠0) positions are properly excluded."""
    from corgisim.convolution import get_valid_polar_positions
    
    radii = [0.0, 1.0]
    azimuths = [0.0, 90.0, 180.0] * u.deg
    
    positions = get_valid_polar_positions(radii, azimuths)
    
    # Should have: (0,0°), (1,0°), (1,90°), (1,180°) = 4 positions
    # Should exclude: (0,90°), (0,180°)
    assert len(positions) == 4
    
    # Check that (0,0°) is included but (0,90°) is not
    zero_radius_positions = [(r, theta) for r, theta in positions if r == 0.0]
    assert len(zero_radius_positions) == 1
    assert zero_radius_positions[0] == (0.0, 0.0 * u.deg)