#!/usr/bin/env python
"""Reduce saved analog probe frames with the stock CorgiDRP L2a recipe."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from astropy.io import fits


# --- Processing configuration ---------------------------------------------
INPUT_PRODUCTS = Path('/path/to/data/new_campaign')
OUTPUT_DIR = INPUT_PRODUCTS / 'reduced'
STOP_AFTER = 'L2a'
CALIBRATION_DIR = Path('/path/to/calibrations')
CHUNK_SIZE = 4


def reduce_products(inputs, output_dir=OUTPUT_DIR, stop_after=STOP_AFTER,
                    calibration_dir=CALIBRATION_DIR):
    """Save individual L2a frames under output_dir/L2a/V<visit>.

    Inputs are FITS paths or a directory containing FITS directly or in V* folders.
    Existing products at the requested level are returned without processing.
    A destination visit must be empty; use a new destination after a failed run.
    Calibrations are explicit, with no automatic database lookup.
    """
    if stop_after not in ('L1', 'L2a'):
        raise ValueError('Only L1 and L2a endpoints are implemented')
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths = []
    for item in inputs:
        item = Path(item).expanduser().resolve()
        if item.is_dir():
            paths.extend(sorted(item.glob('*.fits')))
            paths.extend(sorted(item.glob('V*/*.fits')))
        else:
            paths.append(item)
    if not paths:
        raise ValueError('No input FITS products found')
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate input products')

    groups = defaultdict(list)
    levels = set()
    for path in paths:
        with fits.open(path) as hdus:
            pri, hdr = hdus[0].header, hdus[1].header
            level = hdr['DATALVL']
            levels.add(level)
            if level not in ('L1', 'L2a'):
                raise ValueError(f'{path}: unsupported input level {level}')
            if hdr['ISPC'] != 0:
                raise ValueError(f'{path}: photon-counted inputs are unsupported')
            if hdr['ARRTYPE'] != 'SCI':
                raise ValueError(f'{path}: only SCI detector arrays are supported')
            if level == 'L1' and hdus[1].shape != (1200, 2200):
                raise ValueError(f'{path}: expected a full 1200 x 2200 L1 frame')
            visit = ''.join(str(pri[k]).strip() for k in
                            ('PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'OBSNUM', 'VISNUM'))
            key = (visit, hdr['EXPTIME'], hdr['EMGAIN_C'], hdr['EMGAIN_A'])
            groups[key].append(str(path))
    if len(levels) != 1:
        raise ValueError('Input products must have one consistent data level')
    level = levels.pop()
    if level == 'L2a' and stop_after == 'L1':
        raise ValueError('Requested endpoint precedes input level L2a')
    if level == stop_after:
        return paths

    names_by_visit = defaultdict(list)
    for key, files in groups.items():
        names_by_visit[key[0]].extend(Path(path).name for path in files)
    if any(len(names) != len(set(names)) for names in names_by_visit.values()):
        raise ValueError('Input basenames must be unique within each visit')

    output_dir = Path(output_dir).expanduser().resolve()
    destinations = {key[0]: output_dir / 'L2a' / f'V{key[0]}' for key in groups}
    for destination in destinations.values():
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Destination is not empty: {destination}')
    calibration_dir = Path(calibration_dir).expanduser().resolve()
    calibrations = {
        'DetectorNoiseMaps': None,
        'DetectorParams': str(calibration_dir / 'sim_detector_params.fits'),
        'KGain': None,
        'NonLinearityCalibration': str(calibration_dir / 'sim_identity_nonlin.fits'),
    }
    for path in calibrations.values():
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(path)

    # DRP reads existing user configuration on import. Disable database creation
    # before importing; recipe settings confine any later database use to output.
    os.environ['CORGIDRP_DO_NOT_AUTO_INIT_CALDB'] = '1'
    import corgidrp
    from corgidrp.walker import run_recipe

    template = Path(corgidrp.__file__).parent / 'recipe_templates/l1_to_l2a_basic.json'
    outputs = []
    for (visit, exposure, gain, applied_gain), files in groups.items():
        destination = destinations[visit]
        destination.mkdir(parents=True, exist_ok=True)
        recipe = json.loads(template.read_text())
        recipe['inputs'] = files
        recipe['outputdir'] = str(destination)
        recipe['drpconfig'].update(
            caldb_filepath=str(output_dir / 'caldb.csv'), jit_calib_id=False,
            skip_missing_cal_steps=False, chunk_size=CHUNK_SIZE,
        )
        for step in recipe['steps']:
            for name in step.get('calibs', {}):
                step['calibs'][name] = calibrations[name]
        saved = run_recipe(recipe) or []
        print(f'V{visit} {exposure:g}s gain={gain:g}: {len(files)} inputs, '
              f'{len(saved)} L2a outputs, {len(files) - len(saved)} filtered')
        outputs.extend(Path(path) for path in saved)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, default=[INPUT_PRODUCTS])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--stop-after', choices=('L1', 'L2a'), default=STOP_AFTER)
    parser.add_argument('--calibration-dir', type=Path, default=CALIBRATION_DIR)
    args = parser.parse_args(argv)
    return reduce_products(args.inputs, args.output_dir, args.stop_after, args.calibration_dir)


if __name__ == '__main__':
    main()
