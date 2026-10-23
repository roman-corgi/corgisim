"""Fast tests of campaign validation, planning, safety, and visit metadata.

Propagation and L1 writing are stubbed where needed so the module runs quickly.
"""

import importlib.util
import inspect
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from corgisim import outputs

DRIVER_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'alt_probe_generate_L1_sims.py'


def load_driver():
    """Import the driver from ``examples``."""
    spec = importlib.util.spec_from_file_location('alt_probe_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()


def write_probe(tmp_path, name='probe.fits', shape=(48, 48)):
    """Write a synthetic Gaussian probe and return its path."""
    ny, nx = shape
    y, x = np.indices((ny, nx))
    pattern = 0.8 * np.exp(-((x - 37) ** 2 + (y - 32) ** 2) / 2.0)
    path = tmp_path / name
    fits.writeto(str(path), pattern, overwrite=True)
    return str(path)


def campaign_kwargs(tmp_path, **overrides):
    """Return minimal valid campaign arguments with optional overrides."""
    kwargs = dict(
        scale=1.0,
        probe_files=[write_probe(tmp_path)],
        output_dir=str(tmp_path / 'out'),
        n_frames_per_state=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_default_output_dim_is_255():
    """Verify the constant and function signature default to the 51*5 stamp."""
    assert driver.OUTPUT_DIM == 255
    assert inspect.signature(driver.run_campaign).parameters['output_dim'].default == 255


@pytest.mark.parametrize('frame_time_step_s, exptime', [
    (0.5, 0.1),    # below the 1 s whole-second SCTSRT floor
    (0.999, 0.1),  # just below the floor
    (5.0, 10.0),   # above the floor but shorter than the exposure
    (9.99, 10.0),  # just short of the exposure time
])
def test_frame_time_step_must_clear_exptime_and_one_second(tmp_path, frame_time_step_s, exptime):
    """Reject frame spacing below either the exposure or one-second floor."""
    with pytest.raises(ValueError, match='frame_time_step_s'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, frame_time_step_s=frame_time_step_s, exptime=exptime))


@pytest.mark.parametrize('frame_time_step_s, exptime', [
    (1.0, 0.1),    # exactly the 1 s floor
    (10.0, 10.0),  # exactly the exposure time
    (30.0, 10.0),  # comfortably above both
])
def test_frame_time_step_at_or_above_bound_passes_timing_check(tmp_path, frame_time_step_s, exptime):
    """Accept valid spacing by reaching the later probe-validation stage."""
    bad_probe = write_probe(tmp_path, name='bad_probe.fits', shape=(10, 10))
    with pytest.raises(ValueError, match='does not match DM shape'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, probe_files=[bad_probe],
            frame_time_step_s=frame_time_step_s, exptime=exptime))


def test_planned_filename_matches_outputs_formatter():
    """Verify planned names use the L1 writer's timestamp formatter."""
    visitid = driver.build_visitid('0200', '001', '001', '001', 1, '001')
    for microsecond in (0, 40000, 500000, 950000):
        ftimeutc = datetime(2027, 6, 1, 0, 0, 0, microsecond).isoformat()
        expected = f"cgi_{visitid}_{outputs.isotime_to_yyyymmddThhmmsss(ftimeutc)}_l1_.fits"
        assert driver.planned_l1_filename(visitid, ftimeutc) == expected


def test_filename_planning_detects_sub_tenth_second_collisions():
    """Detect collisions caused by the writer's 0.1-second timestamp rounding."""
    visitid = driver.build_visitid('0200', '001', '001', '001', 1, '001')
    start = datetime(2027, 6, 1, 0, 0, 0)

    collide = [driver.planned_l1_filename(visitid, (start + timedelta(seconds=0.05 * i)).isoformat())
               for i in range(3)]
    assert len(set(collide)) < len(collide)

    spaced = [driver.planned_l1_filename(
        visitid, (start + timedelta(seconds=driver.MIN_FRAME_TIME_STEP_S * i)).isoformat())
        for i in range(3)]
    assert len(set(spaced)) == len(spaced)


def test_invalid_config_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    """Reject invalid input before creating output or copying PROPER files."""
    work_dir = tmp_path / 'cwd'
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with pytest.raises(ValueError, match='frame_time_step_s'):
        driver.run_campaign(**campaign_kwargs(tmp_path, frame_time_step_s=0.5, exptime=0.1))

    assert not (tmp_path / 'out').exists()
    assert os.listdir(work_dir) == []


def test_existing_output_refused_without_overwrite_and_without_side_effects(tmp_path, monkeypatch):
    """Refuse existing products before any PROPER filesystem side effects."""
    work_dir = tmp_path / 'cwd'
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    kwargs = campaign_kwargs(tmp_path)
    visitid = driver.build_visitid(
        driver.VISIT_PROGNUM, driver.VISIT_EXECNUM, driver.VISIT_CAMPAIGN,
        driver.VISIT_SEGMENT, driver.VISIT_OBSNUM_START, driver.VISIT_VISNUM)
    visit_dir = Path(kwargs['output_dir']) / f'V{visitid}'
    visit_dir.mkdir(parents=True)
    (visit_dir / driver.planned_l1_filename(
        visitid, driver.CAMPAIGN_START_TIME_UTC)).write_text('placeholder')

    with pytest.raises(FileExistsError, match='already exist'):
        driver.run_campaign(**kwargs)

    assert os.listdir(work_dir) == []


@pytest.mark.parametrize('start_time', [
    '2027-06-01T00:00:00',
    '2027-06-01T00:00:00Z',
    '2027-06-01T00:00:00+00:00',
    '2027-05-31T20:00:00-04:00',
])
def test_campaign_start_time_is_written_as_naive_utc(tmp_path, monkeypatch, start_time):
    """Equivalent UTC start times reach the writer as identical naive ISO strings."""
    written = []

    class StubOptics:
        def __init__(self, *args, **kwargs):
            pass

        def get_host_star_psf(self, input_scene, **kwargs):
            return object()

        def add_satspot(self, satspot_keywords=None):
            pass

        def remove_satspot(self, satspot_keywords=None):
            pass

    def record_frame(sim_scene, detector, exptime, loc_x, loc_y, outdir, visitid,
                     vistype, ftimeutc, overwrite=False):
        written.append(ftimeutc)

    monkeypatch.setattr(driver.instrument, 'CorgiOptics', StubOptics)
    monkeypatch.setattr(driver.instrument, 'CorgiDetector', lambda *a, **k: object())
    monkeypatch.setattr(driver.scene, 'Scene', lambda properties: object())
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here', lambda: None)
    monkeypatch.setattr(driver, 'load_base_dm',
                        lambda *a, **k: (np.zeros((48, 48)), np.zeros((48, 48))))
    monkeypatch.setattr(driver, 'save_frame', record_frame)

    driver.run_campaign(**campaign_kwargs(
        tmp_path, campaign_start_time_utc=start_time, frame_time_step_s=30.0))

    assert written == ['2027-06-01T00:00:00', '2027-06-01T00:00:30', '2027-06-01T00:01:00']
    for ftimeutc in written:
        assert datetime.fromisoformat(ftimeutc).tzinfo is None
        assert not ftimeutc.endswith('Z') and '+' not in ftimeutc


def test_visit_metadata_tracks_each_probe_execution(tmp_path, monkeypatch):
    """Keep VISITID/VISTYPE synchronized across successive probe executions.

    Stubs capture metadata at scene generation and frame writing, the points
    where it enters simulation COMMENTs and formal L1 headers.
    """
    unset = object()
    scenes = []  # (visit_id, visit_type) seen at each scene generation
    frames = []  # (visitid, vistype) handed to the writer for each frame
    vistype = 'CGIVST_CAL_BORESIGHT'

    class StubOptics:
        def __init__(self, cgi_mode, bandpass, optics_keywords=None, **kwargs):
            self.visit_id = kwargs.get('visit_id', unset)
            self.visit_type = kwargs.get('visit_type', unset)

        def get_host_star_psf(self, input_scene, **kwargs):
            scenes.append((self.visit_id, self.visit_type))
            return object()

        def add_satspot(self, satspot_keywords=None):
            pass

        def remove_satspot(self, satspot_keywords=None):
            pass

    def record_frame(sim_scene, detector, exptime, loc_x, loc_y, outdir, visitid,
                     vistype_, ftimeutc, overwrite=False):
        frames.append((visitid, vistype_))

    monkeypatch.setattr(driver.instrument, 'CorgiOptics', StubOptics)
    monkeypatch.setattr(driver.instrument, 'CorgiDetector', lambda *a, **k: object())
    monkeypatch.setattr(driver.scene, 'Scene', lambda properties: object())
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here', lambda: None)
    monkeypatch.setattr(driver, 'load_base_dm',
                        lambda *a, **k: (np.zeros((48, 48)), np.zeros((48, 48))))
    monkeypatch.setattr(driver, 'save_frame', record_frame)

    driver.run_campaign(**campaign_kwargs(
        tmp_path,
        probe_files=[write_probe(tmp_path, name='probe_a.fits'),
                     write_probe(tmp_path, name='probe_b.fits')],
        vistype=vistype))

    visitids = [driver.build_visitid(
        driver.VISIT_PROGNUM, driver.VISIT_EXECNUM, driver.VISIT_CAMPAIGN,
        driver.VISIT_SEGMENT, driver.VISIT_OBSNUM_START + i, driver.VISIT_VISNUM)
        for i in range(2)]
    assert visitids[0] != visitids[1]

    # One frame per DM state for each probe.
    expected = [(visitid, vistype) for visitid in visitids for _ in range(3)]
    assert scenes == expected
    assert frames == expected
