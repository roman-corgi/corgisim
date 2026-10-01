from corgisim import spec
import numpy as np
import pytest
import json
# import matplotlib.pyplot as plt
from importlib import resources

@pytest.fixture
def mock_optics():
    class MockOptics:
        optics_keywords = {'cor_type': 'spc-spec_band3'}
        lamref_um = 0.73
        slit_param_fname = ''
        slit = 'test_slit'
        slit_x_offset_mas = 50
        slit_y_offset_mas = 100
        slit_ra_offset_mas = 0
        slit_dec_offset_mas = 0
        cor_type = "spc-spec_band3"
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
    mask, dx_fsam_meters = spec.get_slit_mask(mock_optics, dx_fsam_um = 2.0, hires_dim_um = 200)
    
    # Test shape
    assert mask.shape == (100, 100), "Mask shape is incorrect"
    
    # Test values
    assert np.all((mask >= 0) & (mask <= 1)), "Mask values should be between 0 and 1"
    assert np.any(mask > 0), "Mask should have some non-zero values"
    assert np.any(mask < 1), "Mask should have some non-one values"
    
    # Test dx_fsam_meters
    assert pytest.approx(dx_fsam_meters, 1E-9) == 2E-6, "dx value is incorrect"
    
def test_invalid_binning(mock_optics, mock_read_slit_params):
    # Test invalid array dimension and sampling combination
    with pytest.raises(ValueError):
        spec.get_slit_mask(mock_optics, dx_fsam_um = 3.0, hires_dim_um = 200)

def test_read_slit_params(tmp_path):
    # Create a temporary JSON file
    test_data = {
        "slit1": {"width": 10, "height": 100},
        "slit2": {"width": 20, "height": 100}
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
        'wavlen_vs_pos_polycoeff': np.array([3.0, 2.0, 1.0]),
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
        model_x_sign = 1  # nominal SPC: model x runs the same way as EXCAM x
    class Band2PrismMockConfig:
        with resources.path('corgisim.data', 'TVAC_PRISM2_dispersion_profile.npz') as data_path:
            prism_param_fname = data_path
        lam_um = np.linspace(0.610, 0.710, 5)
        wav_step_um = 0.002
        lamref_um = 0.65
        sampling_um = 13.0
        oversampling_factor = 5
        model_x_sign = 1  # nominal SPC: model x runs the same way as EXCAM x

    prism3_config = Band3PrismMockConfig()
    prism2_config = Band2PrismMockConfig()

    # Create a mock image cube
    mock_imwidth = 250
    mock_nlam = 5
    hwbox = 3 
    (xc, yc) = (mock_imwidth//2, mock_imwidth//2)
    image_cube = np.zeros((mock_nlam, mock_imwidth, mock_imwidth))
    image_cube[:, yc-hwbox:yc+hwbox, xc-hwbox:xc+hwbox] = 1  # Mock image cube is a box of bright pixels at center for all wavelengths

    for config in [prism3_config, prism2_config]:
        dispersed_cube, interp_wavs, disp_shift_lam0_x, disp_shift_lam0_y = spec.apply_prism(config, image_cube)
    
        print("Input image cube shape:", image_cube.shape)
        print("Output dispersed cube shape:", dispersed_cube.shape)
        print("Interpolated wavelengths shape:", interp_wavs.shape)
    
        # Check output shapes
        assert dispersed_cube.shape[0] > image_cube.shape[0], "Dispersed cube should have more wavelength slices"
        assert dispersed_cube.shape[1:] == image_cube.shape[1:], "Spatial dimensions should remain the same"
    
        # Check that the output is not all zeros
        assert np.any(dispersed_cube != 0), "Dispersed cube should not be all zeros"
    
        # Check wavelength array
        assert len(interp_wavs) == dispersed_cube.shape[0], "Wavelength array should match dispersed cube size"
        assert np.min(interp_wavs) >= np.min(config.lam_um), "Min wavelength should not decrease"
        assert np.max(interp_wavs) <= np.max(config.lam_um), "Max wavelength should not increase"

        # Verify dispersion direction
        # Find the row with maximum intensity for the shortest and longest wavelengths
        shortest_wave_row = np.argmax(dispersed_cube[0].sum(axis=1))
        longest_wave_row = np.argmax(dispersed_cube[-1].sum(axis=1))
        
        print(f"Row of maximum intensity for shortest wavelength: {shortest_wave_row}")
        print(f"Row of maximum intensity for longest wavelength: {longest_wave_row}")
        
        # Assert that the longest wavelength is dispersed to a lower row number
        assert longest_wave_row < shortest_wave_row, "Longer wavelengths should be shifted to lower row numbers"

def test_apply_prism_mirrored_x():
    """The dispersion x component reverses when the model x axis is mirrored (rotated SPC).

    The clocking angle is calibrated in EXCAM coordinates, so the trace must be built mirrored in
    the model frame for the left-right image flip applied downstream to restore it.
    """
    class Band3PrismMockConfig:
        with resources.path('corgisim.data', 'TVAC_PRISM3_dispersion_profile.npz') as data_path:
            prism_param_fname = data_path
        lam_um = np.linspace(0.675, 0.785, 5)
        wav_step_um = 0.002
        lamref_um = 0.73
        sampling_um = 13.0
        oversampling_factor = 5
        model_x_sign = 1

    mock_imwidth, hwbox = 250, 3
    centre = mock_imwidth // 2
    image_cube = np.zeros((len(Band3PrismMockConfig.lam_um), mock_imwidth, mock_imwidth))
    image_cube[:, centre-hwbox:centre+hwbox, centre-hwbox:centre+hwbox] = 1

    def trace_x_drift(cube):
        """x centroid of the reddest slice minus that of the bluest."""
        xs = np.arange(cube.shape[2])
        cx = [(slice_2d.sum(axis=0) * xs).sum() / slice_2d.sum() for slice_2d in (cube[0], cube[-1])]
        return cx[1] - cx[0]

    config = Band3PrismMockConfig()
    config.model_x_sign = 1
    cube_nom, _, lam0_x_nom, lam0_y_nom = spec.apply_prism(config, image_cube)
    config.model_x_sign = -1
    cube_mir, _, lam0_x_mir, lam0_y_mir = spec.apply_prism(config, image_cube)

    assert lam0_x_mir == -lam0_x_nom, "The lam0 x shift must reverse when the model x axis is mirrored"
    assert lam0_y_mir == lam0_y_nom, "The y dispersion must not change: the flip is in x only"
    assert lam0_x_nom != 0, "PRISM3 has a non-zero x dispersion component to reverse"

    drift_nom, drift_mir = trace_x_drift(cube_nom), trace_x_drift(cube_mir)
    assert drift_nom * drift_mir < 0, "The trace must drift the opposite way in x when mirrored"
    assert drift_mir == pytest.approx(-drift_nom, rel=1E-6), "Mirroring must preserve the magnitude"

@pytest.mark.parametrize("cor_type, bandpass, mirrored", [('spc-spec_band3', '3F', False),
                                                          ('spc-spec_band2', '2F', False),
                                                          ('spc-spec_band3_rotated', '3F', True),
                                                          ('spc-spec_band2_rotated', '2F', True)])
def test_specrot_fsm_x_sign(cor_type, bandpass, mirrored):
    """The FSM x offset passed to PROPER is mirrored for the rotated SPC, the attribute is not."""
    from packaging.version import Version
    import roman_preflight_proper
    from corgisim import instrument

    # The compensation only applies while the rotated SPC masks are mirrored in the model.
    expected_sign = -1 if (mirrored and Version(roman_preflight_proper.__version__) <= Version('2.0.3')) else 1
    prism = 'PRISM2' if bandpass == '2F' else 'PRISM3'
    optics_keywords = {'cor_type': cor_type, 'polaxis': 0, 'output_dim': 51, 'prism': prism,
                       'fsm_x_offset_mas': 50.0, 'fsm_y_offset_mas': 20.0}
    optics = instrument.CorgiOptics('spec', bandpass, optics_keywords=optics_keywords, if_quiet=True)

    assert optics.model_x_sign == expected_sign
    assert optics.optics_keywords['fsm_x_offset_mas'] == expected_sign * 50.0, \
        "FSM x offset handed to PROPER does not match the model x orientation"
    assert optics.fsm_x_offset_mas == 50.0, "The attribute must keep the EXCAM-frame value"
    assert optics.optics_keywords['fsm_y_offset_mas'] == 20.0, "FSM y offset must not be touched"

    # An offset that was never requested must not appear in the keywords handed to PROPER.
    optics_keywords.pop('fsm_x_offset_mas')
    optics = instrument.CorgiOptics('spec', bandpass, optics_keywords=optics_keywords, if_quiet=True)
    assert 'fsm_x_offset_mas' not in optics.optics_keywords
    assert optics.fsm_x_offset_mas == 0.0

if __name__ == '__main__':
    pytest.main([__file__])