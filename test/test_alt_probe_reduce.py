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
                           EMGAIN_C=gain, EMGAIN_A=gain))
    image = np.zeros((1200, 2200), dtype=np.uint16)
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
    calls = []

    def run(recipe):
        calls.append(recipe)
        return [str(Path(recipe['outputdir']) / Path(p).name) for p in recipe['inputs']]

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
        reducer.reduce_products(path, stop_after='L3')
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
                           reducer.STOP_AFTER, reducer.CALIBRATION_DIR)
    reducer.main(['--inputs', str(tmp_path / 'input'), '--output-dir', str(tmp_path / 'out'),
                  '--stop-after', 'L1', '--calibration-dir', str(tmp_path / 'calibs')])
    assert calls.pop() == ([tmp_path / 'input'], tmp_path / 'out', 'L1', tmp_path / 'calibs')


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
