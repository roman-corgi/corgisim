"""Fast, non-propagating checks of the Alternate Probe L1 campaign driver's
timing constraint and output-filename planning.

These tests never run a PROPER propagation and never write an L1 product:
they only exercise ``run_campaign``'s up-front validation and the pure
filename-planning helper, so the whole module runs in seconds.
"""

import importlib.util
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from corgisim import outputs

DRIVER_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'alt_probe_generate_L1_sims.py'


def load_driver():
    """Import the driver script (which lives in examples/, not the package)."""
    spec = importlib.util.spec_from_file_location('alt_probe_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()


def write_probe(tmp_path, name='probe.fits', shape=(48, 48)):
    """Write a synthetic 48x48 Gaussian DM probe (returns its path), so tests
    never depend on delivered probe files outside this repository."""
    ny, nx = shape
    y, x = np.indices((ny, nx))
    pattern = 0.8 * np.exp(-((x - 37) ** 2 + (y - 32) ** 2) / 2.0)
    path = tmp_path / name
    fits.writeto(str(path), pattern, overwrite=True)
    return str(path)


def campaign_kwargs(tmp_path, **overrides):
    """Build a minimal valid run_campaign keyword set for validation tests.

    Supplies the required ``scale`` and ``probe_files`` plus a temporary
    output directory, so each test only specifies the parameter it exercises;
    ``**overrides`` replaces any of those defaults.
    """
    kwargs = dict(
        scale=1.0,
        probe_files=[write_probe(tmp_path)],
        output_dir=str(tmp_path / 'out'),
        n_frames_per_state=1,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize('frame_time_step_s, exptime', [
    (0.5, 0.1),    # below the 1 s whole-second SCTSRT floor
    (0.999, 0.1),  # just below the floor
    (5.0, 10.0),   # above the floor but shorter than the exposure
    (9.99, 10.0),  # just short of the exposure time
])
def test_frame_time_step_must_clear_exptime_and_one_second(tmp_path, frame_time_step_s, exptime):
    """frame_time_step_s < max(exptime, 1.0) is rejected up front."""
    with pytest.raises(ValueError, match='frame_time_step_s'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, frame_time_step_s=frame_time_step_s, exptime=exptime))


@pytest.mark.parametrize('frame_time_step_s, exptime', [
    (1.0, 0.1),    # exactly the 1 s floor
    (10.0, 10.0),  # exactly the exposure time
    (30.0, 10.0),  # comfortably above both
])
def test_frame_time_step_at_or_above_bound_passes_timing_check(tmp_path, frame_time_step_s, exptime):
    """A step >= max(exptime, 1.0) clears the timing check.

    Acceptance is asserted without propagating: a deliberately malformed
    probe makes the call fail at the *later* probe-validation stage, which
    is only reachable once the timing check has passed.
    """
    bad_probe = write_probe(tmp_path, name='bad_probe.fits', shape=(10, 10))
    with pytest.raises(ValueError, match='does not match DM shape'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, probe_files=[bad_probe],
            frame_time_step_s=frame_time_step_s, exptime=exptime))


def test_planned_filename_matches_outputs_formatter():
    """The planned path must be derived with the writer's own formatter."""
    visitid = driver.build_visitid('0200', '001', '001', '001', 1, '001')
    for microsecond in (0, 40000, 500000, 950000):
        ftimeutc = datetime(2027, 6, 1, 0, 0, 0, microsecond).isoformat()
        expected = f"cgi_{visitid}_{outputs.isotime_to_yyyymmddThhmmsss(ftimeutc)}_l1_.fits"
        assert driver.planned_l1_filename(visitid, ftimeutc) == expected


def test_filename_planning_detects_sub_tenth_second_collisions():
    """Justifies retaining the collision check as a defence in depth.

    The L1 filename timestamp is rounded to 0.1 s, so distinct FTIMEUTC
    values do not by themselves imply distinct filenames. The enforced
    >= max(exptime, 1.0) s spacing keeps real campaigns clear of this, but
    the planner is what actually guarantees it.
    """
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
    """A rejected configuration must not create output or copy PROPER files.

    roman_preflight_proper.copy_here() writes prescription modules into the
    current working directory, so it is deferred until after every
    validation stage.
    """
    work_dir = tmp_path / 'cwd'
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with pytest.raises(ValueError, match='frame_time_step_s'):
        driver.run_campaign(**campaign_kwargs(tmp_path, frame_time_step_s=0.5, exptime=0.1))

    assert not (tmp_path / 'out').exists()
    assert os.listdir(work_dir) == []


def test_existing_output_refused_without_overwrite_and_without_side_effects(tmp_path, monkeypatch):
    """The deepest reachable validation stage also leaves no side effects."""
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


def test_visit_metadata_tracks_each_probe_execution(tmp_path, monkeypatch):
    """Each probe execution's visit metadata matches its own VISITID and the
    configured VISTYPE.

    One CorgiOptics object serves several probe executions, and it snapshots
    visit_id/visit_type into the simulation-metadata block when the scene is
    generated -- that block becomes the L1 primary header's visit_id/visit_type
    COMMENT cards. Overriding only the VISITID keyword at write time would leave
    the COMMENT provenance stale for every probe after the first.

    Optics, detector, scene and frame writing are stubbed, so nothing propagates
    and nothing is written; the assertions cover only the observable contract.
    """
    unset = object()
    scenes = []  # (visit_id, visit_type) seen at each scene generation
    frames = []  # (visitid, vistype) handed to the writer for each frame
    vistype = 'CGIVST_CAL_BORESIGHT'

    class StubOptics:
        def __init__(self, cgi_mode, bandpass, optics_keywords=None, **kwargs):
            # ``unset`` marks metadata the driver never supplied; the real class
            # would silently substitute its own default.
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

    # One scene and (at n_frames_per_state=1) one frame per DM state:
    # unprobed, positive, negative for each probe, in that order.
    expected = [(visitid, vistype) for visitid in visitids for _ in range(3)]
    assert scenes == expected
    assert frames == expected
