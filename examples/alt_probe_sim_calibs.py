#!/usr/bin/env python
"""Write L2a model-truth calibrations for Corgisim frames without nonlinearity.

These describe the simulator, not measured detector calibration or uncertainty.
Prescan subtraction assumes no extra image/prescan bias offset.
"""

import argparse
import os
from pathlib import Path

from astropy.io import fits
from astropy.time import Time
import numpy as np


def create_simulated_calibrations(input_product, output_dir):
    """Read detector truth from primary COMMENT cards of one representative L1."""
    with fits.open(input_product) as hdus:
        truth = dict(str(card).split(' : ', 1) for card in hdus[0].header['COMMENT']
                     if ' : ' in str(card))
    if truth['nonlin_path'] != 'None':
        raise ValueError('Identity calibration requires simulated nonlin_path=None')
    names = {
        'FWC_PP_E': 'full_well_image', 'FWC_EM_E': 'full_well_serial',
        'KGAINPAR': 'e_per_dn', 'READ_N': 'read_noise',
        'ROWREADT': 'row_read_time', 'NEMGAIN': 'numel_gain_register',
    }
    params = {key: float(truth[value]) for key, value in names.items()}
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f'Calibration destination is not empty: {output_dir}')
    os.environ['CORGIDRP_DO_NOT_AUTO_INIT_CALDB'] = '1'
    from corgidrp import check
    from corgidrp.data import Dataset, DetectorParams, NonLinearityCalibration

    dataset = Dataset([str(input_product)])
    detector = DetectorParams(params, date_valid=Time(dataset[0].ext_hdr['SCTSRT']))
    identity = NonLinearityCalibration(
        np.array([[np.nan, 1., 8000.], [-65536., 1., 1.], [65536., 1., 1.]]),
        pri_hdr=dataset[0].pri_hdr.copy(), ext_hdr=dataset[0].ext_hdr.copy(),
        input_dataset=dataset,
    )
    for calibration, filename in ((detector, 'sim_detector_params.fits'),
                                  (identity, 'sim_identity_nonlin.fits')):
        calibration.pri_hdr['ORIGIN'] = 'Corgisim model truth'
        calibration.ext_hdr['HISTORY'] = 'Nominal simulator truth; not a measured calibration.'
        calibration.ext_hdr['HISTORY'] = f'Simulator metadata source: {Path(input_product).name}'
        calibration.save(filedir=str(output_dir), filename=filename)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_product', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args(argv)
    return create_simulated_calibrations(args.input_product, args.output_dir)


if __name__ == '__main__':
    main()
