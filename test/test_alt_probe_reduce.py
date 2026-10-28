"""Contracts around the stock recipe; detector algorithms belong to CorgiDRP."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from astropy.io import fits
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'examples/alt_probe_reduce.py'
spec = importlib.util.spec_from_file_location('alt_probe_reduce', SCRIPT)
reducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reducer)


def product(directory, name, gain=281., level='L1', pc=0):
    directory.mkdir(parents=True, exist_ok=True)
    pri = fits.Header(dict(PROGNUM='0200', EXECNUM='001', CAMPAIGN='001',
                           SEGMENT='001', OBSNUM='911', VISNUM='001'))
    hdr = fits.Header(dict(DATALVL=level, ISPC=pc, ARRTYPE='SCI',
                           EXPTIME=19. if gain == 281 else 28.,
                           EMGAIN_C=gain, EMGAIN_A=gain, EACQ_COL=512, EACQ_ROW=512))
    image = np.zeros((1200, 2200) if level == 'L1' else (1024, 1024), dtype=np.uint16)
    path = directory / name
    fits.HDUList([fits.PrimaryHDU(header=pri), fits.ImageHDU(image, header=hdr)]).writeto(path)
    return path


@pytest.fixture
def stock_walker(tmp_path, monkeypatch):
    drp = ModuleType('corgidrp')
    drp.__file__ = str(tmp_path / 'drp/__init__.py')
    template_dir = tmp_path / 'drp/recipe_templates'
    template_dir.mkdir(parents=True)
    template = dict(name='l1_to_l2a_basic', drpconfig={}, steps=[
        dict(name='prescan_biassub', calibs={'DetectorNoiseMaps': 'AUTOMATIC,OPTIONAL'}),
        dict(name='detect_cosmic_rays', calibs={'DetectorParams': 'AUTOMATIC', 'KGain': 'AUTOMATIC'}),
        dict(name='correct_nonlinearity', calibs={'NonLinearityCalibration': 'AUTOMATIC'}),
    ])
    (template_dir / 'l1_to_l2a_basic.json').write_text(json.dumps(template))
    (template_dir / 'l2a_to_l2b.json').write_text(json.dumps(dict(
        name='l2a_to_l2b', drpconfig={}, steps=[
            dict(name='convert_to_electrons', calibs={'KGain': 'AUTOMATIC'}),
            dict(name='dark_subtraction', calibs={'DetectorNoiseMaps': 'AUTOMATIC'},
                 keywords={'outputdir': 'AUTOMATIC'}),
            dict(name='cti_correction', calibs={'TrapCalibration': 'AUTOMATIC,OPTIONAL'}),
            dict(name='flat_division', calibs={'FlatField': 'AUTOMATIC'}),
            dict(name='correct_bad_pixels', calibs={'BadPixelMap': 'AUTOMATIC'}),
        ])))
    (template_dir / 'l2b_to_l3.json').write_text(json.dumps(dict(
        name='l2b_to_l3', drpconfig={}, steps=[
            dict(name='create_wcs', calibs={'AstrometricCalibration': 'AUTOMATIC'}),
            dict(name='divide_by_exptime'), dict(name='crop'),
        ])))
    calls = []

    def run(recipe):
        calls.append(recipe)
        saved = [Path(recipe['outputdir']) / Path(p).name for p in recipe['inputs']]
        for path in saved:
            path.write_bytes(b'saved product')
        return [str(path) for path in saved]

    walker = ModuleType('corgidrp.walker')
    walker.run_recipe = run
    monkeypatch.setitem(sys.modules, 'corgidrp', drp)
    monkeypatch.setitem(sys.modules, 'corgidrp.walker', walker)
    calibs = tmp_path / 'calibs'
    calibs.mkdir()
    for name in ('sim_detector_params.fits', 'sim_identity_nonlin.fits'):
        (calibs / name).touch()
    return calls, calibs


def test_explicit_recipe_camera_groups_and_directory_scope(tmp_path, stock_walker):
    calls, calibs = stock_walker
    a = product(tmp_path / 'inputs/V001', 'a.fits')
    b = product(tmp_path / 'inputs/V001', 'b.fits')
    c = product(tmp_path / 'inputs/V001', 'c.fits', gain=1181.)
    product(tmp_path / 'inputs/reduced/L2a/V001', 'unrelated.fits', level='L2a')
    outputs = reducer.reduce_products(tmp_path / 'inputs', tmp_path / 'out', 'L2a', calibs)
    assert len(outputs) == 3
    assert [recipe['inputs'] for recipe in calls] == [[str(a), str(b)], [str(c)]]
    for recipe in calls:
        assert recipe['outputdir'] == str(tmp_path / 'out/L2a/V0200001001001911001')
        assert recipe['drpconfig']['caldb_filepath'] == str(tmp_path / 'out/caldb.csv')
        assert recipe['drpconfig']['jit_calib_id'] is False
        assert recipe['drpconfig']['skip_missing_cal_steps'] is False
        assert recipe['steps'][0]['calibs']['DetectorNoiseMaps'] is None
        assert recipe['steps'][1]['calibs']['KGain'] is None
        assert recipe['steps'][1]['keywords']['cosm_tail'] == 'auto'
        assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] == str(calibs / 'sim_identity_nonlin.fits')


def test_level_stop_does_not_need_calibrations_or_drp(tmp_path):
    path = product(tmp_path, 'input.fits')
    assert reducer.reduce_products(path, stop_after='L1') == [path]
    path2 = product(tmp_path, 'saved.fits', level='L2a')
    assert reducer.reduce_products(path2, stop_after='L2a') == [path2]
    with pytest.raises(ValueError, match='precedes'):
        reducer.reduce_products(path2, stop_after='L1')
    with pytest.raises(ValueError, match='consistent'):
        reducer.reduce_products([path, path2])


def test_reject_pc_duplicates_future_and_occupied_output(tmp_path, stock_walker):
    calls, calibs = stock_walker
    path = product(tmp_path, 'input.fits', pc=1)
    with pytest.raises(ValueError, match='photon-counted'):
        reducer.reduce_products(path)
    with pytest.raises(ValueError, match='Duplicate'):
        reducer.reduce_products([path, path])
    with pytest.raises(ValueError, match='endpoints'):
        reducer.reduce_products(path, stop_after='L4')
    path = product(tmp_path, 'analog.fits')
    destination = tmp_path / 'out/L2a/V0200001001001911001'
    destination.mkdir(parents=True)
    (destination / 'partial.fits').touch()
    with pytest.raises(FileExistsError, match='not empty'):
        reducer.reduce_products(path, tmp_path / 'out', calibration_dir=calibs)
    assert calls == []


def test_configured_cli_and_overrides_use_one_function(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reducer, 'reduce_products', lambda *args: calls.append(args))
    reducer.main([])
    assert calls.pop() == ([reducer.INPUT_PRODUCTS], reducer.OUTPUT_DIR,
                           reducer.STOP_AFTER, reducer.CALIBRATION_DIR, reducer.CROP_SIZE)
    reducer.main(['--inputs', str(tmp_path / 'input'), '--output-dir', str(tmp_path / 'out'),
                  '--stop-after', 'L1', '--calibration-dir', str(tmp_path / 'calibs'), '--crop-size', '101'])
    assert calls.pop() == ([tmp_path / 'input'], tmp_path / 'out', 'L1', tmp_path / 'calibs', 101)


def test_l2b_resume_needs_only_stage_calibrations(tmp_path, stock_walker):
    calls, calibs = stock_walker
    (calibs / 'sim_identity_nonlin.fits').unlink()
    camera = calibs / 'gain_281_exptime_19'
    camera.mkdir()
    for name in ('sim_kgain.fits', 'sim_dark.fits', 'sim_flat.fits', 'sim_bad_pixels.fits'):
        (camera / name).touch()
    path = product(tmp_path, 'saved.fits', level='L2a')
    reducer.reduce_products(path, tmp_path / 'out', 'L2b', calibs)
    assert len(calls) == 1 and calls[0]['name'] == 'l2a_to_l2b'
    assert calls[0]['steps'][1]['calibs'] == {
        'DetectorNoiseMaps': None, 'Dark': str(camera / 'sim_dark.fits')}
    assert calls[0]['steps'][2]['calibs']['TrapCalibration'] is None
    saved = product(tmp_path, 'final.fits', level='L2b')
    assert reducer.reduce_products(saved, stop_after='L2b') == [saved]


def test_later_prerequisite_failure_preserves_lower_outputs(tmp_path, stock_walker):
    calls, calibs = stock_walker
    path = product(tmp_path, 'raw.fits')
    original = path.read_bytes()
    with pytest.raises(FileNotFoundError, match='sim_kgain'):
        reducer.reduce_products(path, tmp_path / 'out', 'L2b', calibs)
    assert len(calls) == 1 and calls[0]['name'] == 'l1_to_l2a_basic'
    assert (tmp_path / 'out/L2a/V0200001001001911001').is_dir()
    assert (tmp_path / 'out/L2a/V0200001001001911001/raw.fits').read_bytes() == b'saved product'
    assert not (tmp_path / 'out/L2b').exists()
    assert path.read_bytes() == original


def test_empty_walker_return_does_not_enter_next_stage(tmp_path, stock_walker, monkeypatch):
    """Handle a mocked empty return; native all-rejected chunks raise instead."""
    calls, calibs = stock_walker
    def no_outputs(recipe):
        calls.append(recipe)
        return []
    monkeypatch.setattr(sys.modules['corgidrp.walker'], 'run_recipe', no_outputs)
    path = product(tmp_path, 'raw.fits')
    assert reducer.reduce_products(path, tmp_path / 'out', 'L2b', calibs) == []
    assert len(calls) == 1


def test_l3_resume_only_needs_astrometry_and_derives_reference(tmp_path, stock_walker):
    calls, calibs = stock_walker
    (calibs / 'sim_identity_nonlin.fits').unlink()
    (calibs / 'sim_detector_params.fits').unlink()
    (calibs / 'sim_astrometry.fits').touch()
    path = product(tmp_path, 'saved.fits', level='L2b')
    reducer.reduce_products(path, tmp_path / 'out', 'L3', calibs, crop_size=101)
    assert len(calls) == 1 and calls[0]['name'] == 'l2b_to_l3'
    assert calls[0]['steps'][0]['keywords']['offset'] == [2, 2]
    assert calls[0]['steps'][2]['keywords']['sizexy'] == 101
    final = product(tmp_path, 'final.fits', level='L3')
    assert reducer.reduce_products(final, stop_after='L3') == [final]
    with pytest.raises(ValueError, match='precedes'):
        reducer.reduce_products(final, stop_after='L2b')


def test_l3_checks_all_references_and_crop_before_writes(tmp_path, stock_walker):
    calls, calibs = stock_walker
    (calibs / 'sim_astrometry.fits').touch()
    a = product(tmp_path, 'a.fits', level='L2b')
    b = product(tmp_path, 'b.fits', level='L2b')
    fits.setval(b, 'EACQ_COL', ext=1, value=511)
    with pytest.raises(ValueError, match='consistent EACQ'):
        reducer.reduce_products([a, b], tmp_path / 'out', 'L3', calibs)
    for size in (0, 1025):
        with pytest.raises(ValueError, match='crop'):
            reducer.reduce_products(a, tmp_path / 'out', 'L3', calibs, crop_size=size)
    assert calls == [] and not (tmp_path / 'out').exists()


def test_simulator_calibration_rejects_nonlinearity_and_absent_truth(tmp_path):
    spec = importlib.util.spec_from_file_location('alt_probe_sim_calibs',
                                                SCRIPT.with_name('alt_probe_sim_calibs.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    path = tmp_path / 'input.fits'
    header = fits.Header()
    header['COMMENT'] = 'nonlin_path : simulated_response.txt'
    fits.PrimaryHDU(header=header).writeto(path)
    with pytest.raises(ValueError, match='nonlin_path=None'):
        helper.create_simulated_calibrations(path, tmp_path / 'calibs')
    assert not (tmp_path / 'calibs').exists()
    header['COMMENT'] = 'nonlin_path : None'
    fits.PrimaryHDU(header=header).writeto(path, overwrite=True)
    with pytest.raises(KeyError):
        helper.create_simulated_calibrations(path, tmp_path / 'calibs')
