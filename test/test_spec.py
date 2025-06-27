from corgisim import spec
import numpy as np
import pytest
import json
# import matplotlib.pyplot as plt
from importlib import resources

@pytest.fixture
def mock_optics():
    class MockOptics:
        proper_keywords = {'cor_type': 'spc-spec_band3'}
        lamref_um = 0.73
        slit_param_fname = ''
        slit = 'test_slit'
        slit_x_offset_mas = 0
        slit_y_offset_mas = 0
    return MockOptics()

@pytest.fixture
def mock_read_slit_params(monkeypatch):
    def mock_func(filename):
        return {
            'test_slit': {
                'width': 10,
                'height': 100
            }
        }
    monkeypatch.setattr(spec, 'read_slit_params', mock_func)

def test_get_slit_mask(mock_optics, mock_read_slit_params):
    # Test normal operation
    mask, dx = spec.get_slit_mask(mock_optics)
    
    # Test shape
    assert mask.shape == (160, 160), "Mask shape is incorrect"
    
    # Test values
    assert np.all((mask >= 0) & (mask <= 1)), "Mask values should be between 0 and 1"
    assert np.any(mask > 0), "Mask should have some non-zero values"
    assert np.any(mask < 1), "Mask should have some non-one values"
    
    # Test dx
    assert pytest.approx(dx, 1e-9) == 5e-6, "dx value is incorrect"
    
def test_invalid_binning(mock_optics, mock_read_slit_params):
    # Test invalid binning
    with pytest.raises(ValueError):
        spec.get_slit_mask(mock_optics, binfac=33)

def test_read_slit_params(tmp_path):
    # Create a temporary JSON file
    test_data = {
        "slit1": {"width": 10, "height": 100},
        "slit2": {"width": 20, "height": 200}
    }
    temp_file = tmp_path / "test_slit_params.json"
    with open(temp_file, "w") as f:
        json.dump(test_data, f)

    # Call the function
    result = spec.read_slit_params(str(temp_file))

    # Assert the result matches the test data
    assert result == test_data, "read_slit_params did not return the expected data"

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        spec.read_slit_params("non_existent_file.json")

    # Test with invalid JSON
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, "w") as f:
        f.write("This is not valid JSON")

    with pytest.raises(json.JSONDecodeError):
        spec.read_slit_params(str(invalid_json_file))

def test_read_prism_params(tmp_path):
    # Create a temporary NumPy file
    test_data = {
        'pos_vs_wavlen_polycoeff': np.array([1.0, 2.0, 3.0]),
        'clocking_angle': 45.0
    }
    temp_file = tmp_path / "test_prism_params.npz"
    np.savez(temp_file, **test_data)

    # Call the function
    result = spec.read_prism_params(str(temp_file))

    # Assert the result matches the test data
    np.testing.assert_array_equal(result['pos_vs_wavlen_polycoeff'], test_data['pos_vs_wavlen_polycoeff'])
    assert result['clocking_angle'] == test_data['clocking_angle']

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        spec.read_prism_params("non_existent_file.npz")

    # Test with missing required parameter
    incomplete_data = {'clocking_angle': 45.0}
    incomplete_file = tmp_path / "incomplete_prism_params.npz"
    np.savez(incomplete_file, **incomplete_data)
    
    with pytest.raises(KeyError):
        spec.read_prism_params(str(incomplete_file))

    # Test with invalid file format
    invalid_file = tmp_path / "invalid_file.txt"
    with open(invalid_file, "w") as f:
        f.write("This is not a NumPy file")

    with pytest.raises((ValueError, OSError)):  # The exact error might depend on the NumPy version
        spec.read_prism_params(str(invalid_file))

def test_apply_prism():
    class Band3PrismMockConfig:
        with resources.path('corgisim.data', 'TVAC_PRISM3_dispersion_profile.npz') as data_path:
            prism_param_fname = data_path
        lam_um = np.linspace(0.675, 0.785, 5)
        wav_step_um = 0.002
        lamref_um = 0.73
        sampling_um = 13.0
        oversampling_factor = 5
    class Band2PrismMockConfig:
        with resources.path('corgisim.data', 'TVAC_PRISM2_dispersion_profile.npz') as data_path:
            prism_param_fname = data_path
        lam_um = np.linspace(0.610, 0.710, 5)
        wav_step_um = 0.002
        lamref_um = 0.65
        sampling_um = 13.0
        oversampling_factor = 5

    prism3_config = Band3PrismMockConfig()
    prism2_config = Band2PrismMockConfig()

    # Create a mock image cube
    mock_imwidth = 250
    mock_nlam = 5
    image_cube = np.zeros((mock_nlam, mock_imwidth, mock_imwidth))
    image_cube[:, mock_imwidth // 2, mock_imwidth // 2] = 1  # Single bright pixel in center for all wavelengths

    for config in [prism3_config, prism2_config]:
        dispersed_cube, interp_wavs = spec.apply_prism(config, image_cube)
    
        print("Input image cube shape:", image_cube.shape)
        print("Output dispersed cube shape:", dispersed_cube.shape)
        print("Interpolated wavelengths shape:", interp_wavs.shape)
    
        # Check output shapes
        assert dispersed_cube.shape[0] > image_cube.shape[0], "Dispersed cube should have more wavelength slices"
        assert dispersed_cube.shape[1:] == image_cube.shape[1:], "Spatial dimensions should remain the same"
    
        # Check that the output is not all zeros
        assert np.any(dispersed_cube != 0), "Dispersed cube should not be all zeros"
    
        # Check that dispersion occurred
        center_flux = np.sum(dispersed_cube[:, 10, 10])
        total_flux = np.sum(dispersed_cube)
        print(f"Flux in center pixel: {center_flux}")
        print(f"Total flux in dispersed cube: {total_flux}")
    
        assert center_flux < total_flux, "Dispersion should spread flux from center"
        assert np.isclose(total_flux, image_cube.sum(), rtol=1e-5), "Total flux should be conserved"
    
        # Check wavelength array
        assert len(interp_wavs) == dispersed_cube.shape[0], "Wavelength array should match dispersed cube size"
        assert np.min(interp_wavs) >= np.min(config.lam_um), "Min wavelength should not decrease"
        assert np.max(interp_wavs) <= np.max(config.lam_um), "Max wavelength should not increase"

if __name__ == '__main__':
    pytest.main([__file__])