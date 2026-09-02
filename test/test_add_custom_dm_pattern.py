from corgisim import scene, instrument
from corgisim.sat_spots import add_custom_pattern_dm
import pytest
import numpy as np
import proper
import roman_preflight_proper


def make_gaussian_probe(shape=(48, 48), x_act=13, y_act=8, sigma_act=1.0, peak_volts=0.8):
    """
    Synthetic Gaussian DM probe mirroring the delivered Alternate Probe files
    (dmrel_nfov_band1_360deg_ni5e-07_x13_y8_gauss0.fits and siblings):
    a Gaussian bump of sigma = 1 actuator pitch, peaked at the center actuator
    (24, 24) offset by (x_act, y_act) in the numpy [row, col] = [y, x]
    convention, with peak amplitude in volts comparable to the delivery
    (~0.76-0.84 V).
    """
    ny, nx = shape
    y, x = np.indices((ny, nx))
    x0, y0 = 24 + x_act, 24 + y_act
    return peak_volts * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma_act ** 2))


def test_add_custom_pattern_dm_math():
    """Verify the pure-numpy pattern addition: sign handling and no input mutation."""
    rng = np.random.default_rng(0)
    dm = rng.uniform(0, 100, size=(48, 48))
    dm_orig = dm.copy()
    pattern = make_gaussian_probe()
    scale = 0.5

    dm_pos = add_custom_pattern_dm(dm, pattern, scale)
    dm_neg = add_custom_pattern_dm(dm, pattern, scale, sign="negative")

    assert np.array_equal(dm_pos, dm + scale * pattern)
    assert np.array_equal(dm_neg, dm - scale * pattern)
    # inputs must not be modified
    assert np.array_equal(dm, dm_orig)
    # explicit sign="positive" matches the default
    assert np.array_equal(add_custom_pattern_dm(dm, pattern, scale, sign="positive"), dm_pos)
    # numpy float scales (e.g. from parsed configs) are accepted
    assert np.array_equal(add_custom_pattern_dm(dm, pattern, np.float64(scale)), dm_pos)


def test_add_custom_pattern_dm_validation():
    """Verify input validation, in particular that scale is required (no default)."""
    dm = np.zeros((48, 48))
    pattern = make_gaussian_probe()

    # scale is deliberately REQUIRED: the 0.3 (legacy HOWFSC) vs 1.0 (delivered
    # amplitude) choice must be made explicitly by the caller
    with pytest.raises(TypeError):
        add_custom_pattern_dm(dm, pattern)

    with pytest.raises(ValueError):
        add_custom_pattern_dm(dm, pattern[:24, :24], 1.0)  # shape mismatch

    bad = pattern.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        add_custom_pattern_dm(dm, bad, 1.0)  # non-finite values

    with pytest.raises(ValueError):
        add_custom_pattern_dm(dm, pattern, 1.0, sign="plus")  # invalid sign

    with pytest.raises(TypeError):
        add_custom_pattern_dm(dm, pattern, "1.0")  # non-numeric scale

    with pytest.raises(TypeError):
        add_custom_pattern_dm(dm, pattern, True)  # bool is not a valid scale


def test_add_custom_pattern_dm_scale_types():
    """Scale accepts real scalar numerics (incl. NumPy scalars); rejects non-real/non-scalar."""
    dm = np.zeros((48, 48))
    pattern = make_gaussian_probe()
    expected = 0.5 * pattern

    # accepted: Python and NumPy real scalar types (0.5 exactly representable in all)
    for scale in (0.5, np.float16(0.5), np.float32(0.5), np.float64(0.5)):
        assert np.array_equal(add_custom_pattern_dm(dm, pattern, scale), expected)
    for scale in (2, np.int32(2), np.int64(2), np.uint8(2)):
        assert np.array_equal(add_custom_pattern_dm(dm, pattern, scale), 2 * pattern)

    # rejected: non-real or non-scalar values
    for bad in (True, np.bool_(True), 1j, np.complex128(0.5), "0.5", None,
                np.array(0.5), np.array([0.5]), [0.5]):
        with pytest.raises(TypeError):
            add_custom_pattern_dm(dm, pattern, bad)


def test_add_satspot_custom_pattern_keywords():
    """
    Verify the CorgiOptics.add_satspot custom-pattern keyword set:
    validation errors, SATSPOTS flag, DM mutation/restoration, provenance info.
    """
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    rootname = 'hlc_ni_3e-8'
    dm1 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')

    optics_keywords = {'cor_type': cor_type, 'use_errors': 1, 'polaxis': 10, 'output_dim': 101,
                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
                       'use_fpm': 1, 'use_lyot_stop': 1, 'use_field_stop': 1}

    probe = make_gaussian_probe()
    scale = 1.0

    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    assert optics.SATSPOTS == 0
    assert optics.satspot_info is None

    # ---- validation errors (raised before any DM mutation) ----
    with pytest.raises(KeyError):
        # scale is required with custom_pattern and has no default
        optics.add_satspot({'custom_pattern': probe})

    with pytest.raises(KeyError):
        # analytical keywords must not be mixed with custom_pattern
        optics.add_satspot({'custom_pattern': probe, 'scale': scale, 'contrast': 1e-5})

    assert optics.SATSPOTS == 0
    assert np.array_equal(optics.optics_keywords['dm1_v'], dm1)

    # ---- add: DM updated, flag set, provenance recorded ----
    satspot_keywords = {'custom_pattern': probe, 'scale': scale, 'sign': 'positive',
                        'pattern_name': 'unit-test-gauss'}
    optics.add_satspot(satspot_keywords)
    assert optics.SATSPOTS == 1
    assert np.array_equal(optics.optics_keywords['dm1_v'], dm1 + scale * probe)
    assert optics.satspot_info == {'satspot_pattern_name': 'unit-test-gauss',
                                   'satspot_scale': scale,
                                   'satspot_sign': 'positive'}
    # the caller's DM array is not modified in place
    assert np.array_equal(dm1, proper.prop_fits_read(
        roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits'))

    # ---- remove: DM restored exactly, flag and provenance cleared ----
    optics.remove_satspot(satspot_keywords)
    assert optics.SATSPOTS == 0
    assert optics.satspot_info is None
    assert np.allclose(optics.optics_keywords['dm1_v'], dm1, rtol=1e-15, atol=1e-15)

    # ---- constructor pass-through matches add_satspot on a plain object ----
    optics_ctor = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords,
                                         satspot_keywords={'custom_pattern': probe, 'scale': scale},
                                         if_quiet=True)
    assert optics_ctor.SATSPOTS == 1
    assert np.array_equal(optics_ctor.optics_keywords['dm1_v'], dm1 + scale * probe)

    # ---- negative sign path ----
    optics.add_satspot({'custom_pattern': probe, 'scale': scale, 'sign': 'negative'})
    assert optics.SATSPOTS == 1
    assert np.array_equal(optics.optics_keywords['dm1_v'], dm1 - scale * probe)
    optics.remove_satspot({'custom_pattern': probe, 'scale': scale, 'sign': 'negative'})
    assert optics.SATSPOTS == 0
    assert np.allclose(optics.optics_keywords['dm1_v'], dm1, rtol=1e-15, atol=1e-15)


def test_custom_probe_difference_image():
    """
    Verify the pairwise-probing observable end to end: simulate unprobed,
    positive-probe, and negative-probe noise-free images with a synthetic
    Gaussian DM probe; check that median(pos, neg) - unprobed recovers a
    positive probe-intensity image (|E_probe|^2 >= 0), that probed images
    differ from the unprobed one, and that the probe provenance appears in
    the simulated-image header COMMENTs.
    """
    host_star_properties = {'Vmag': 5, 'spectral_type': 'G0V', 'magtype': 'vegamag'}
    base_scene = scene.Scene(host_star_properties)

    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    rootname = 'hlc_ni_3e-8'
    dm1 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm1_v.fits')
    dm2 = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/' + rootname + '_dm2_v.fits')

    optics_keywords = {'cor_type': cor_type, 'use_errors': 1, 'polaxis': 10, 'output_dim': 101,
                       'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
                       'use_fpm': 1, 'use_lyot_stop': 1, 'use_field_stop': 1}

    probe = make_gaussian_probe()
    satspot_keywords = {'custom_pattern': probe, 'scale': 1.0, 'pattern_name': 'unit-test-gauss'}

    # unprobed image
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    image_unprobed = optics.get_host_star_psf(base_scene).host_star_image.data

    # positive probe
    optics.add_satspot(satspot_keywords)
    sim_scene_pos = optics.get_host_star_psf(base_scene)
    image_pos = sim_scene_pos.host_star_image.data

    # probe provenance recorded in the simulated-product header COMMENTs
    comments = '\n'.join(str(c) for c in sim_scene_pos.host_star_image.header['COMMENT'])
    assert 'SATSPOTS : 1' in comments
    assert 'satspot_pattern_name : unit-test-gauss' in comments
    assert 'satspot_scale : 1.0' in comments
    assert 'satspot_sign : positive' in comments

    # negative probe
    optics.remove_satspot(satspot_keywords)
    neg_keywords = dict(satspot_keywords, sign='negative')
    optics.add_satspot(neg_keywords)
    sim_scene_neg = optics.get_host_star_psf(base_scene)
    image_neg = sim_scene_neg.host_star_image.data
    comments_neg = '\n'.join(str(c) for c in sim_scene_neg.host_star_image.header['COMMENT'])
    assert 'satspot_sign : negative' in comments_neg

    # restore and check the unprobed state is recovered
    optics.remove_satspot(neg_keywords)
    assert optics.SATSPOTS == 0
    assert np.allclose(optics.optics_keywords['dm1_v'], dm1, rtol=1e-15, atol=1e-15)

    # the probe visibly modifies the image
    assert np.max(np.abs(image_pos - image_unprobed)) > 0
    assert np.max(np.abs(image_neg - image_unprobed)) > 0

    # pairwise-probing observable: median(pos, neg) - unprobed ~= |E_probe|^2 >= 0
    image_med = np.median(np.stack([image_pos, image_neg]), axis=0)
    image_probe = image_med - image_unprobed

    assert np.max(image_probe) > 0
    # the probe intensity term must dominate any (nonlinear/numerical) negative residuals
    pos_sum = np.sum(image_probe[image_probe > 0])
    neg_sum = np.sum(np.abs(image_probe[image_probe < 0]))
    assert pos_sum > 10 * neg_sum


if __name__ == '__main__':
    test_add_custom_pattern_dm_math()
    test_add_custom_pattern_dm_validation()
    test_add_custom_pattern_dm_scale_types()
    test_add_satspot_custom_pattern_keywords()
    test_custom_probe_difference_image()
