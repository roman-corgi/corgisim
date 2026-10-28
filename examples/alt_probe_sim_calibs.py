#!/usr/bin/env python
"""Write nominal model calibrations for Corgisim analog frames through L2b.

These describe the simulator, not measured detector calibration or uncertainty.
Prescan subtraction assumes no extra image/prescan bias offset.
"""

import argparse
import os
from pathlib import Path

from astropy.io import fits
from astropy.time import Time
import numpy as np


def partial_cic_moments(gain, stages):
    """Input-referred mean/variance for installed EMCCDDetect's auto gain mode.

    Partial CIC enters stages 1..N-1 as independent Bernoulli events. Each
    electron then follows the uniform binomial gain branching process.
    """
    if gain <= 200:
        return 0., 0.
    probability = gain**(1 / stages) - 1
    q = probability / 40
    stage_gains = (1 + probability)**np.arange(1, stages)
    stage_variance = (1 - probability) / (1 + probability) * stage_gains * (stage_gains - 1)
    mean = q * stage_gains.sum() / gain
    variance = (q * stage_variance + q * (1 - q) * stage_gains**2).sum() / gain**2
    return mean, variance


def create_simulated_calibrations(input_products, output_dir, stop_after='L2a'):
    """Read simulator COMMENT truth; optionally write each exposure/gain camera set.

    L2b assumes no flat response variations, traps or extra detector defects.
    Dark means approximate prescan median centering at the amplifier bias.
    """
    if stop_after not in ('L2a', 'L2b'):
        raise ValueError('Calibration endpoint must be L2a or L2b')
    if isinstance(input_products, (str, Path)):
        input_products = [input_products]
    paths = []
    for item in input_products:
        item = Path(item).expanduser().resolve()
        paths.extend(sorted(item.glob('*.fits')) + sorted(item.glob('V*/*.fits'))
                     if item.is_dir() else [item])
    if not paths:
        raise ValueError('No simulator input products found')
    names = {
        'FWC_PP_E': 'full_well_image', 'FWC_EM_E': 'full_well_serial',
        'KGAINPAR': 'e_per_dn', 'READ_N': 'read_noise',
        'ROWREADT': 'row_read_time', 'NEMGAIN': 'numel_gain_register',
    }
    cameras = {}
    params = None
    for path in paths:
        with fits.open(path) as hdus:
            truth = dict(str(card).split(' : ', 1) for card in hdus[0].header['COMMENT']
                         if ' : ' in str(card))
            hdr = hdus[1].header if len(hdus) > 1 else fits.Header()
        if truth['nonlin_path'] != 'None':
            raise ValueError('Identity calibration requires simulated nonlin_path=None')
        this_params = {key: float(truth[value]) for key, value in names.items()}
        if params is not None and params != this_params:
            raise ValueError('Input detector parameters differ; use separate calibration directories')
        params = this_params
        if stop_after == 'L2b':
            for key, value in dict(flat_path='None', use_traps='False',
                                   fast_gain_mode='auto', gain_CIC_Q='roman').items():
                if truth[key] != value:
                    raise ValueError(f'Unsupported simulated {key}={truth[key]}')
            gain = hdr.get('EMGAIN_M', hdr['EMGAIN_A'] if hdr['EMGAIN_A'] > 0 else hdr['EMGAIN_C'])
            if (hdr['ISPC'] != 0 or gain != float(truth['em_gain']) or gain < 1
                    or gain != hdr['EMGAIN_C']):
                raise ValueError('Requires analog frames with a matching simulated EM gain')
            key = (hdr['EXPTIME'], gain)
            if key in cameras and any(cameras[key][1][k] != truth[k] for k in ('dark_rate', 'cic_noise')):
                raise ValueError('Input dark/CIC parameters differ within a camera')
            cameras.setdefault(key, (path, truth))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f'Calibration destination is not empty: {output_dir}')
    os.environ['CORGIDRP_DO_NOT_AUTO_INIT_CALDB'] = '1'
    from corgidrp import check
    from corgidrp.data import (Dataset, DetectorParams, NonLinearityCalibration,
                              KGain, Dark, FlatField, BadPixelMap)
    from corgidrp.detector import ENF

    dataset = Dataset([str(paths[0])])
    detector = DetectorParams(params, date_valid=Time(dataset[0].ext_hdr['SCTSRT']))
    identity = NonLinearityCalibration(
        np.array([[np.nan, 1., 8000.], [-65536., 1., 1.], [65536., 1., 1.]]),
        pri_hdr=dataset[0].pri_hdr.copy(), ext_hdr=dataset[0].ext_hdr.copy(),
        input_dataset=dataset,
    )
    def save(calibration, directory, filename, source):
        calibration.pri_hdr['ORIGIN'] = 'Corgisim model truth'
        calibration.ext_hdr['HISTORY'] = 'Nominal simulator truth; not a measured calibration.'
        calibration.ext_hdr['HISTORY'] = 'Analytical model; source FITS supplies parameters, not a measured stack.'
        calibration.ext_hdr['HISTORY'] = f'Simulator metadata source: {source.name}'
        calibration.save(filedir=str(directory), filename=filename)

    save(detector, output_dir, 'sim_detector_params.fits', paths[0])
    save(identity, output_dir, 'sim_identity_nonlin.fits', paths[0])
    for (exposure, gain), (path, truth) in cameras.items():
        dataset = Dataset([str(path)])
        pri, hdr = dataset[0].pri_hdr.copy(), dataset[0].ext_hdr.copy()
        partial_mean, partial_variance = partial_cic_moments(gain, int(params['NEMGAIN']))
        enf = ENF(gain, params['NEMGAIN'])
        read_variance = (params['READ_N'] / gain)**2
        max_signal = 65536 * params['KGAINPAR'] / gain
        signal = np.r_[0., np.geomspace(1.e-6, max_signal, 2048)]
        sigma = np.sqrt(signal + (read_variance + partial_variance) / enf**2)
        # Native DRP multiplies this input-electron PTC by ENF after gain division.
        # A constant negative branch keeps extrapolation finite and nonnegative.
        ptc = np.column_stack((np.r_[-1.e6, signal + partial_mean], np.r_[sigma[0], sigma]))
        kgain = KGain(params['KGAINPAR'], err=0., ptc=ptc,
                      pri_hdr=pri.copy(), ext_hdr=hdr.copy(), input_dataset=dataset)
        kgain.ext_hdr['RN'], kgain.ext_hdr['RN_ERR'] = params['READ_N'], 0.
        kgain.ptc_hdr['COMMENT'] = 'Input-electron signal and sigma before native ENF multiplication.'
        camera_dir = output_dir / f'gain_{gain:g}_exptime_{exposure:g}'
        camera_dir.mkdir()
        save(kgain, camera_dir, 'sim_kgain.fits', path)
        # Direct-dark matching uses exact equality with the persisted KGain value.
        hdr['KGAINPAR'] = KGain(str(camera_dir / 'sim_kgain.fits')).value
        dark_mean = float(truth['dark_rate']) * exposure + float(truth['cic_noise']) + partial_mean
        dark = Dark(np.full((1024, 1024), dark_mean), err=np.zeros((1, 1024, 1024)),
                    dq=np.zeros((1024, 1024), dtype=np.uint16),
                    pri_hdr=pri.copy(), ext_hdr=hdr.copy(), input_dataset=dataset)
        dark.ext_hdr['HISTORY'] = 'Dark+entrance CIC+partial CIC mean; nominal zero model-parameter error.'
        dark.ext_hdr['HISTORY'] = 'Prescan stochastic median assumed centered at amplifier bias.'
        flat = FlatField(np.ones((1024, 1024)), input_dataset=dataset)
        bad_pixels = BadPixelMap(np.zeros((1024, 1024), dtype=np.uint16),
                                input_dataset=Dataset([dark]))
        for calibration, filename in ((dark, 'sim_dark.fits'),
                                      (flat, 'sim_flat.fits'), (bad_pixels, 'sim_bad_pixels.fits')):
            save(calibration, camera_dir, filename, path)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_products', type=Path, nargs='+')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--stop-after', choices=('L2a', 'L2b'), default='L2a')
    args = parser.parse_args(argv)
    return create_simulated_calibrations(args.input_products, args.output_dir, args.stop_after)


if __name__ == '__main__':
    main()
