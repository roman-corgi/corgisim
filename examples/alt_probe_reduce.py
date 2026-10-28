#!/usr/bin/env python
"""Reduce saved analog probe frames with stock CorgiDRP recipes through L3."""

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
CROP_SIZE = 255
COSMIC_TAIL = 'auto'  # Match the simulator's Roman tail length, round(EM gain * 40/1000).


def reduce_products(inputs, output_dir=OUTPUT_DIR, stop_after=STOP_AFTER,
                    calibration_dir=CALIBRATION_DIR, crop_size=CROP_SIZE):
    """Save individual calibrated frames under output_dir/<level>/V<visit>.

    Inputs are FITS paths or a directory containing FITS directly or in V* folders.
    Existing products at the requested level are returned without processing.
    A destination visit must be empty; use a new destination after a failed run.
    Calibrations are explicit, with no automatic database lookup.
    """
    stages = ('L1', 'L2a', 'L2b', 'L3')
    if stop_after not in stages:
        raise ValueError('Only L1, L2a, L2b and L3 endpoints are implemented')
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
            if level not in stages:
                raise ValueError(f'{path}: unsupported input level {level}')
            if hdr['ISPC'] != 0:
                raise ValueError(f'{path}: photon-counted inputs are unsupported')
            if hdr['ARRTYPE'] != 'SCI':
                raise ValueError(f'{path}: only SCI detector arrays are supported')
            if level == 'L1' and hdus[1].shape != (1200, 2200):
                raise ValueError(f'{path}: expected a full 1200 x 2200 L1 frame')
            visit = ''.join(str(pri[k]).strip() for k in
                            ('PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'OBSNUM', 'VISNUM'))
            gain = hdr.get('EMGAIN_M', hdr['EMGAIN_A'] if hdr['EMGAIN_A'] > 0 else hdr['EMGAIN_C'])
            key = (visit, hdr['EXPTIME'], hdr['EMGAIN_C'], gain)
            groups[key].append(str(path))
    if len(levels) != 1:
        raise ValueError('Input products must have one consistent data level')
    level = levels.pop()
    if stages.index(stop_after) < stages.index(level):
        raise ValueError(f'Requested endpoint precedes input level {level}')
    if level == stop_after:
        return paths

    names_by_visit = defaultdict(list)
    for key, files in groups.items():
        names_by_visit[key[0]].extend(Path(path).name for path in files)
    if any(len(names) != len(set(names)) for names in names_by_visit.values()):
        raise ValueError('Input basenames must be unique within each visit')

    output_dir = Path(output_dir).expanduser().resolve()
    calibration_dir = Path(calibration_dir).expanduser().resolve()
    outputs = []
    for target in stages[stages.index(level) + 1:stages.index(stop_after) + 1]:
        if not groups:
            break
        destinations = {key[0]: output_dir / target / f'V{key[0]}' for key in groups}
        for destination in destinations.values():
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f'Destination is not empty: {destination}')
        runs = []
        references = {}
        for key, files in groups.items():
            visit, exposure, commanded_gain, gain = key
            camera = calibration_dir / f'gain_{gain:g}_exptime_{exposure:g}'
            if target == 'L3':
                calibrations = {'AstrometricCalibration': str(calibration_dir / 'sim_astrometry.fits')}
                geometry = set()
                for file in files:
                    hdr = fits.getheader(file, 1)
                    geometry.add((hdr['EACQ_COL'], hdr['EACQ_ROW'], hdr['NAXIS1'], hdr['NAXIS2']))
                if len(geometry) != 1:
                    raise ValueError('L3 requires consistent EACQ/shape within each camera group')
                x, y, width, height = references[key] = geometry.pop()
                if not isinstance(crop_size, int) or crop_size < 1:
                    raise ValueError('crop_size must be a positive integer')
                for center, length in ((x, width), (y, height)):
                    center = round(center - .5) + .5 if crop_size % 2 == 0 else round(center)
                    start = int(center + .5 - crop_size / 2)
                    if start < 0 or start + crop_size > length:
                        raise ValueError('Requested L3 crop extends beyond the input image')
            else:
                calibrations = {'DetectorNoiseMaps': None,
                                'DetectorParams': str(calibration_dir / 'sim_detector_params.fits')}
                if target == 'L2a':
                    calibrations.update(KGain=None,
                                        NonLinearityCalibration=str(calibration_dir / 'sim_identity_nonlin.fits'))
                else:
                    calibrations.update(KGain=str(camera / 'sim_kgain.fits'),
                                        Dark=str(camera / 'sim_dark.fits'), TrapCalibration=None,
                                        FlatField=str(camera / 'sim_flat.fits'),
                                        BadPixelMap=str(camera / 'sim_bad_pixels.fits'))
            for path in calibrations.values():
                if path is not None and not Path(path).is_file():
                    raise FileNotFoundError(path)
            runs.append((key, files, calibrations))

        # Existing user configuration is read on import. Database creation is
        # disabled; recipe settings confine later database use to output.
        os.environ['CORGIDRP_DO_NOT_AUTO_INIT_CALDB'] = '1'
        import corgidrp
        from corgidrp.walker import run_recipe

        template_name = {'L2a': 'l1_to_l2a_basic.json', 'L2b': 'l2a_to_l2b.json',
                         'L3': 'l2b_to_l3.json'}[target]
        template = Path(corgidrp.__file__).parent / 'recipe_templates' / template_name
        next_groups = {}
        for key, files, calibrations in runs:
            visit, exposure, commanded_gain, gain = key
            destination = destinations[visit]
            destination.mkdir(parents=True, exist_ok=True)
            recipe = json.loads(template.read_text())
            recipe['inputs'] = files
            recipe['outputdir'] = str(destination)
            recipe['process_in_chunks'] = True
            recipe['drpconfig'].update(
                caldb_filepath=str(output_dir / 'caldb.csv'), jit_calib_id=False,
                skip_missing_cal_steps=False, chunk_size=CHUNK_SIZE,
            )
            for step in recipe['steps']:
                if step['name'] == 'detect_cosmic_rays':
                    step.setdefault('keywords', {})['cosm_tail'] = COSMIC_TAIL
                if step['name'] == 'create_wcs':
                    x, y, width, height = references[key]
                    # EACQ is a NumPy pixel coordinate; FITS CRPIX is one-based.
                    # Translate stock DRP's default reference to that same star.
                    step['keywords'] = {'offset': [x + 1 - (width - 1) // 2,
                                                   y + 1 - (height - 1) // 2]}
                if step['name'] == 'crop':
                    step['keywords'] = {'sizexy': crop_size}
                if step['name'] == 'dark_subtraction':
                    step['calibs'] = {'DetectorNoiseMaps': None, 'Dark': calibrations['Dark']}
                    step['keywords']['outputdir'] = str(destination)
                for name in step.get('calibs', {}):
                    step['calibs'][name] = calibrations[name]
            saved = run_recipe(recipe) or []
            print(f'V{visit} {exposure:g}s gain={gain:g}: {len(files)} inputs, '
                  f'{len(saved)} {target} outputs, {len(files) - len(saved)} filtered')
            if saved:
                next_groups[key] = saved
        groups = next_groups
        outputs = [Path(path) for files in groups.values() for path in files]
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, default=[INPUT_PRODUCTS])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--stop-after', choices=('L1', 'L2a', 'L2b', 'L3'), default=STOP_AFTER)
    parser.add_argument('--calibration-dir', type=Path, default=CALIBRATION_DIR)
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE, help='L3 square crop width in pixels')
    args = parser.parse_args(argv)
    return reduce_products(args.inputs, args.output_dir, args.stop_after, args.calibration_dir, args.crop_size)


if __name__ == '__main__':
    main()
