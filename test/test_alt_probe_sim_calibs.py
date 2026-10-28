"""Numerical checks of nominal camera calibrations through native DRP functions."""

import importlib.util
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

os.environ['CORGIDRP_DO_NOT_AUTO_INIT_CALDB'] = '1'
pytest.importorskip('corgidrp')
from corgidrp import check, data, mocks
from corgidrp.detector import ENF
from corgidrp.l2a_to_l2b import (convert_to_electrons, em_gain_division,
                               add_shot_noise_to_err, dark_subtraction, desmear)

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/alt_probe_sim_calibs.py'
spec = importlib.util.spec_from_file_location('alt_probe_sim_calibs', SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_camera_calibration_reload_conversion_noise_and_dark(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    truth = dict(nonlin_path=None, flat_path=None, use_traps=False, fast_gain_mode='auto',
                 gain_CIC_Q='roman', full_well_image=90000., full_well_serial=105000.,
                 e_per_dn=8.7, read_noise=165., row_read_time=.0002235,
                 numel_gain_register=604, dark_rate=.001, cic_noise=.0088)
    for gain, exposure in ((281., 19.), (1181., 28.)):
        pri, hdr = mocks.create_default_L1_headers('SCI')
        hdr['DATALVL'], hdr['BUNIT'] = 'L2a', 'DN'
        hdr['EMGAIN_C'] = hdr['EMGAIN_A'] = gain
        hdr['EXPTIME'] = exposure
        for key, value in dict(truth, em_gain=gain).items():
            pri['COMMENT'] = f'{key} : {value}'
        path = inputs / f'gain_{gain:g}.fits'
        fits.HDUList([fits.PrimaryHDU(header=pri),
                      fits.ImageHDU(np.zeros((4, 4)), header=hdr)]).writeto(path)
    calibs = helper.create_simulated_calibrations(inputs, tmp_path / 'calibs', 'L2b')
    detector = data.DetectorParams(str(calibs / 'sim_detector_params.fits'))
    for gain, exposure in ((281., 19.), (1181., 28.)):
        camera = calibs / f'gain_{gain:g}_exptime_{exposure:g}'
        kgain = data.KGain(str(camera / 'sim_kgain.fits'))
        dark = data.Dark(str(camera / 'sim_dark.fits'))
        assert kgain.error == 0 and kgain.ext_hdr['RN'] == 165.
        assert (data.FlatField(str(camera / 'sim_flat.fits')).data == 1).all()
        assert (data.BadPixelMap(str(camera / 'sim_bad_pixels.fits')).data == 0).all()

        # Independent branching recurrence, including Bernoulli CIC production.
        p = gain**(1 / 604) - 1
        q = p / 40
        mean, variance, partial_mean, partial_variance = 1., 0., 0., 0.
        for _ in range(1, 604):
            variance = (1 + p)**2 * variance + p * (1 - p) * mean
            mean *= 1 + p
            partial_mean += q * mean / gain
            partial_variance += (q * variance + q * (1 - q) * mean**2) / gain**2
        np.testing.assert_allclose(helper.partial_cic_moments(gain, 604),
                                   (partial_mean, partial_variance), rtol=1.e-12)
        np.testing.assert_allclose(dark.data, .001 * exposure + .0088 + partial_mean, rtol=1.e-7)

        image = data.Image(str(inputs / f'gain_{gain:g}.fits'))
        signal = np.tile([-10., 0., 1., 10.], (4, 1))
        image.data[:] = signal * gain / 8.7
        image.err[:] = .1
        dataset = data.Dataset([image])
        converted = em_gain_division(convert_to_electrons(dataset, kgain))
        np.testing.assert_allclose(converted.all_data[0], signal, atol=1.e-12)
        errors = add_shot_noise_to_err(converted, kgain, detector)
        expected_variance = ((.1 * 8.7 / gain)**2 + (165. / gain)**2 + partial_variance
                             + ENF(gain, 604)**2 * np.maximum(signal - partial_mean, 0))
        np.testing.assert_allclose(errors.all_err[0, 0]**2, expected_variance, rtol=5.e-5)
        assert np.isfinite(errors.all_err).all() and (errors.all_err >= 0).all()
        dark.data = dark.data[:4, :4].copy()
        dark.err = dark.err[:, :4, :4].copy()
        dark.dq = dark.dq[:4, :4].copy()
        subtracted = dark_subtraction(errors, dark=dark)
        np.testing.assert_allclose(subtracted.all_data[0], signal - dark.data[0, 0], atol=1.e-8)
    assert helper.partial_cic_moments(200., 604) == (0., 0.)
    assert helper.partial_cic_moments(1., 604) == (0., 0.)


def test_native_desmear_inverts_current_row_included():
    pri, hdr = mocks.create_default_L1_headers('SCI')
    hdr['EXPTIME'] = 19.
    truth = np.arange(1., 17.).reshape(4, 4)
    alpha = .0002235 / 19.
    smeared = truth + alpha * np.cumsum(truth, axis=0)
    dataset = data.Dataset([data.Image(smeared, pri_hdr=pri, ext_hdr=hdr)])
    detector = data.DetectorParams({'ROWREADT': .0002235})
    restored = desmear(dataset, detector, detector_regions={'SCI': {'image': {'rows': 4, 'cols': 4}}})
    np.testing.assert_allclose(restored.all_data[0], truth, atol=1.e-12)


@pytest.mark.parametrize('key,value', [('flat_path', 'response.fits'), ('use_traps', 'True'),
                                      ('fast_gain_mode', 'True'), ('gain_CIC_Q', '0.01')])
def test_l2b_helper_rejects_unsupported_simulator_models(tmp_path, key, value):
    truth = dict(nonlin_path='None', flat_path='None', use_traps='False', fast_gain_mode='auto',
                 gain_CIC_Q='roman', full_well_image=90000., full_well_serial=105000.,
                 e_per_dn=8.7, read_noise=165., row_read_time=.0002235, numel_gain_register=604)
    truth[key] = value
    header = fits.Header()
    for name, setting in truth.items():
        header['COMMENT'] = f'{name} : {setting}'
    path = tmp_path / 'input.fits'
    fits.PrimaryHDU(header=header).writeto(path)
    with pytest.raises(ValueError, match=key):
        helper.create_simulated_calibrations(path, tmp_path / 'calibs', 'L2b')
    assert not (tmp_path / 'calibs').exists()
