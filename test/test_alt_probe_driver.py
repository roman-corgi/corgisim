"""Fast tests of campaign validation, planning, safety, and visit metadata.

Propagation and L1 writing are stubbed where needed so the module runs quickly.
"""

import importlib.util
import inspect
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from corgisim import instrument, observation, outputs

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


@pytest.mark.parametrize('parts', [
    ('02000', '01', '001', '001', 1, '001'),  # compensating malformed widths
    ('0200', '0001', '01', '001', 1, '001'),
    ('0200', '001', '001', '001', 1000, '01'),
    ('０２００', '001', '001', '001', 1, '001'),  # non-ASCII digits
])
def test_visitid_rejects_invalid_components(parts):
    with pytest.raises(ValueError, match='VISITID components'):
        driver.build_visitid(*parts)


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


# Use the real pattern operations while replacing all propagation and detector noise.
add_satspot = instrument.CorgiOptics.add_satspot
remove_satspot = instrument.CorgiOptics.remove_satspot


class MockOptics:
    add_satspot = add_satspot

    def __init__(self, *args, optics_keywords=None, **kwargs):
        self.optics_keywords = dict(optics_keywords or {
            'use_dm1': 1, 'dm1_v': np.zeros((48, 48)),
        })
        self.optics_keywords['dm1_v'] = self.optics_keywords['dm1_v'].copy()
        self.SATSPOTS = 0
        self.satspot_info = None
        self.lam0_um = 0.575
        self.scenes = []
        self.removals = []
        self.injections = 0

    def get_host_star_psf(self, input_scene):
        snapshot = SimpleNamespace(
            dm=self.optics_keywords['dm1_v'].copy(),
            satspots=self.SATSPOTS, info=dict(self.satspot_info or {}),
            visit_id=self.visit_id, visit_type=self.visit_type,
        )
        self.scenes.append(snapshot)
        return snapshot

    def inject_point_sources(self, input_scene, sim_scene):
        self.injections += 1
        sim_scene.injected = True
        return sim_scene

    def remove_satspot(self, satspot_keywords):
        self.removals.append(satspot_keywords['sign'])
        remove_satspot(self, satspot_keywords)


class MockDetector:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def generate_detector_image(self, sim_scene, exptime, **kwargs):
        self.calls.append((sim_scene, exptime, kwargs))
        primary = fits.PrimaryHDU()
        for key, value in dict(visit_id=sim_scene.visit_id,
                               visit_type=sim_scene.visit_type, **sim_scene.info).items():
            primary.header['COMMENT'] = f'{key} : {value}'
        extension = fits.ImageHDU(data=sim_scene.dm.copy())
        extension.header['SATSPOTS'] = sim_scene.satspots
        return SimpleNamespace(image_on_detector=fits.HDUList([primary, extension]))


@pytest.fixture
def mocked_campaign(monkeypatch):
    optics_instances, cameras, written = [], [], []

    def make_optics(*args, **kwargs):
        instance = MockOptics(*args, **kwargs)
        optics_instances.append(instance)
        return instance

    def make_detector(*args, **kwargs):
        instance = MockDetector(*args, **kwargs)
        cameras.append(instance)
        return instance

    def record_frame(hdul, **kwargs):
        written.append((hdul, kwargs))

    monkeypatch.setattr(driver.instrument, 'CorgiOptics', make_optics)
    monkeypatch.setattr(driver.instrument, 'CorgiDetector', make_detector)
    monkeypatch.setattr(driver.scene, 'Scene', lambda properties: object())
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here', lambda: None)
    monkeypatch.setattr(driver, 'load_base_dm',
                        lambda *a, **k: (np.zeros((48, 48)), np.zeros((48, 48))))
    monkeypatch.setattr(outputs, 'save_hdu_to_fits', record_frame)
    return optics_instances, cameras, written


def test_invalid_visitid_fails_before_acquisition(tmp_path, mocked_campaign, monkeypatch):
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here',
                        lambda: pytest.fail('PROPER copy before validation'))
    with pytest.raises(ValueError, match='VISITID components'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, visit_prognum='02000', visit_execnum='01'))
    assert mocked_campaign == ([], [], [])
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('overwrite', [False, True])
@pytest.mark.parametrize('blocked_path', ['second_visit', 'output_ancestor'])
def test_blocked_output_ancestor_fails_before_acquisition(
        tmp_path, mocked_campaign, monkeypatch, overwrite, blocked_path):
    probes = [write_probe(tmp_path, name='probe_a.fits'),
              write_probe(tmp_path, name='probe_b.fits')]
    output_dir = tmp_path / 'out'
    if blocked_path == 'second_visit':
        output_dir.mkdir()
        second_visitid = driver.build_visitid(
            driver.VISIT_PROGNUM, driver.VISIT_EXECNUM, driver.VISIT_CAMPAIGN,
            driver.VISIT_SEGMENT, driver.VISIT_OBSNUM_START + 1, driver.VISIT_VISNUM)
        blocker = output_dir / f'V{second_visitid}'
    else:
        blocker = output_dir
        output_dir = output_dir / 'nested'
    blocker.write_text('blocked')
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here',
                        lambda: pytest.fail('PROPER copy before validation'))

    with pytest.raises(NotADirectoryError, match='output ancestor'):
        driver.run_campaign(**campaign_kwargs(
            tmp_path, probe_files=probes, output_dir=output_dir, overwrite=overwrite))

    assert mocked_campaign == ([], [], [])
    assert blocker.read_text() == 'blocked'
    if blocked_path == 'second_visit':
        assert list(output_dir.iterdir()) == [blocker]


@pytest.mark.parametrize('overwrite', [False, True])
@pytest.mark.parametrize('destination', ['directory', 'symlink'])
def test_invalid_output_destination_fails_before_acquisition(
        tmp_path, mocked_campaign, monkeypatch, overwrite, destination):
    kwargs = campaign_kwargs(tmp_path, overwrite=overwrite)
    visitid = driver.build_visitid(
        driver.VISIT_PROGNUM, driver.VISIT_EXECNUM, driver.VISIT_CAMPAIGN,
        driver.VISIT_SEGMENT, driver.VISIT_OBSNUM_START, driver.VISIT_VISNUM)
    visit_dir = Path(kwargs['output_dir']) / f'V{visitid}'
    visit_dir.mkdir(parents=True)
    path = visit_dir / driver.planned_l1_filename(visitid, driver.CAMPAIGN_START_TIME_UTC)
    if destination == 'directory':
        path.mkdir()
    else:
        path.symlink_to(tmp_path / 'missing.fits')
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here',
                        lambda: pytest.fail('PROPER copy before validation'))

    with pytest.raises(ValueError, match='output path is a directory or symlink'):
        driver.run_campaign(**kwargs)

    assert mocked_campaign == ([], [], [])


@pytest.mark.parametrize('start_time', [
    '2027-06-01T00:00:00',
    '2027-06-01T00:00:00Z',
    '2027-06-01T00:00:00+00:00',
    '2027-05-31T20:00:00-04:00',
])
def test_campaign_start_time_is_written_as_naive_utc(tmp_path, mocked_campaign, start_time):
    """Driver timestamps reach the shared acquisition loop and writer unchanged."""
    _, _, written = mocked_campaign
    driver.run_campaign(**campaign_kwargs(
        tmp_path, campaign_start_time_utc=start_time, frame_time_step_s=30.0,
        n_frames_per_state=2))
    stamps = [kw['overwrite_ext_keywords']['FTIMEUTC'] for _, kw in written]
    assert stamps == [(datetime(2027, 6, 1) + timedelta(seconds=30 * i)).isoformat()
                      for i in range(6)]
    assert all(datetime.fromisoformat(stamp).tzinfo is None for stamp in stamps)


@pytest.mark.parametrize('overwrite', [False, True])
def test_driver_uses_shared_sequence_for_multiple_frames_and_visits(tmp_path, mocked_campaign,
                                                                  monkeypatch, overwrite):
    """Keep group flags, DM states, metadata and write-through across probe visits."""
    optics_instances, cameras, written = mocked_campaign
    vistype = 'CGIVST_CAL_BORESIGHT'
    probes = [write_probe(tmp_path, name='probe_a.fits'),
              write_probe(tmp_path, name='probe_b.fits')]
    # No-retention mode must not make even temporary deep copies of detector images.
    monkeypatch.setattr(observation.copy, 'deepcopy',
                        lambda image: pytest.fail('driver retained a full-frame image'))
    driver.run_campaign(**campaign_kwargs(
        tmp_path, probe_files=probes, n_frames_per_state=2,
        vistype=vistype, overwrite=overwrite))
    optics = optics_instances[0]
    visitids = [driver.build_visitid(
        driver.VISIT_PROGNUM, driver.VISIT_EXECNUM, driver.VISIT_CAMPAIGN,
        driver.VISIT_SEGMENT, driver.VISIT_OBSNUM_START + i, driver.VISIT_VISNUM)
        for i in range(2)]
    assert len(optics.scenes) == 6  # No extra restored-science propagation for n_frames=0.
    assert optics.removals == ['positive', 'negative'] * 2
    assert optics.SATSPOTS == 0
    assert np.allclose(optics.optics_keywords['dm1_v'], 0)
    assert len(cameras[0].calls) == len(written) == 12
    pattern = fits.getdata(probes[0])
    for visit_index, visitid in enumerate(visitids):
        for frame_index, (hdul, kw) in enumerate(written[6 * visit_index:6 * visit_index + 6]):
            sign = (None, 'positive', 'negative')[frame_index // 2]
            expected_dm = 0 if sign is None else pattern * (1 if sign == 'positive' else -1)
            assert np.allclose(hdul[1].data, expected_dm)
            assert hdul[1].header['SATSPOTS'] == 1
            comments = '\n'.join(hdul[0].header['COMMENT'])
            assert f'visit_id : {visitid}' in comments
            assert f'visit_type : {vistype}' in comments
            if sign is None:
                assert 'satspot_sign' not in comments
            else:
                assert f'satspot_sign : {sign}' in comments
                assert f'satspot_pattern_name : probe_{"ab"[visit_index]}' in comments
            assert kw['overwrite_pri_keywords'] == {'VISITID': visitid, 'VISTYPE': vistype}
            expected_time = datetime.fromisoformat(driver.CAMPAIGN_START_TIME_UTC) + timedelta(
                seconds=driver.FRAME_TIME_STEP_S * (6 * visit_index + frame_index))
            assert kw['overwrite_ext_keywords'] == {'FTIMEUTC': expected_time.isoformat()}
            assert kw['overwrite'] is overwrite
            assert kw['write_as_L1'] is True
            assert kw['outdir'] == str(tmp_path / 'out' / f'V{visitid}')
    assert all(kw == {'full_frame': True, 'loc_x': 512, 'loc_y': 512}
               for _, _, kw in cameras[0].calls)


@pytest.mark.parametrize('full_frame', [False, True])
@pytest.mark.parametrize('custom', [False, True])
def test_shared_sequence_preserves_cameras_point_sources_and_retained_list(full_frame, custom):
    """Analytical/custom trios and science use the requested cameras and physical states."""
    optics, satellite_camera, science_camera = MockOptics(), MockDetector(), MockDetector()
    keywords = ({'custom_pattern': np.ones((48, 48)), 'scale': 0.3}
                if custom else {'num_pairs': 2, 'sep_lamD': 7,
                                'angle_deg': [0, 90], 'contrast': 1e-5})
    with pytest.warns(UserWarning, match='save_as_fits'):
        images = observation.generate_observation_sequence(
            SimpleNamespace(point_source_dra=0), optics, science_camera, 20, 2,
            satspots_are_present=True, satspots_detector=satellite_camera,
            satspots_exptime=5, satspots_number_of_frames=2,
            satspot_keywords=keywords, full_frame=full_frame, loc_x=12, loc_y=34)
    assert len(images) == 8
    assert [image.image_on_detector[1].header['SATSPOTS'] for image in images] == [1] * 6 + [0] * 2
    assert [exptime for _, exptime, _ in satellite_camera.calls] == [5] * 6
    assert [exptime for _, exptime, _ in science_camera.calls] == [20] * 2
    expected_options = {'full_frame': True, 'loc_x': 12, 'loc_y': 34} if full_frame else {}
    assert all(kw == expected_options for _, _, kw in satellite_camera.calls + science_camera.calls)
    assert optics.removals == ['positive', 'negative']
    assert optics.injections == len(optics.scenes) == 4
    base, positive, negative, science = optics.scenes
    assert np.allclose(base.dm, 0) and np.allclose(science.dm, 0)
    assert np.any(positive.dm != 0) and np.allclose(negative.dm, -positive.dm)
    if custom:
        assert np.allclose(positive.dm, 0.3)
        assert np.array_equal(keywords['custom_pattern'], np.ones((48, 48)))
    assert 'sign' not in keywords  # Do not mutate the caller's pattern keywords.
    images[0].image_on_detector[1].data[:] = 99
    assert np.allclose(images[1].image_on_detector[1].data, 0)
    assert np.allclose(optics.optics_keywords['dm1_v'], 0, atol=1e-15)


@pytest.mark.parametrize('failure_stage', ['detector', 'writer', 'propagation', 'point_source'])
@pytest.mark.parametrize('sign', [None, 'positive', 'negative'])
def test_shared_sequence_restores_probe_on_error(monkeypatch, failure_stage, sign):
    optics, camera = MockOptics(), MockDetector()
    original_psf, original_inject = optics.get_host_star_psf, optics.inject_point_sources
    original_detector = camera.generate_detector_image

    def maybe_fail_psf(input_scene):
        if failure_stage == 'propagation' and (optics.satspot_info or {}).get('satspot_sign') == sign:
            raise RuntimeError('deliberate failure')
        return original_psf(input_scene)

    def maybe_fail_inject(input_scene, sim_scene):
        if failure_stage == 'point_source' and sim_scene.info.get('satspot_sign') == sign:
            raise RuntimeError('deliberate failure')
        return original_inject(input_scene, sim_scene)

    def maybe_fail_detector(sim_scene, exptime, **kwargs):
        if failure_stage == 'detector' and sim_scene.info.get('satspot_sign') == sign:
            raise RuntimeError('deliberate failure')
        return original_detector(sim_scene, exptime, **kwargs)

    def maybe_fail_writer(hdul, **kwargs):
        if failure_stage == 'writer' and (optics.satspot_info or {}).get('satspot_sign') == sign:
            raise RuntimeError('deliberate failure')

    monkeypatch.setattr(optics, 'get_host_star_psf', maybe_fail_psf)
    monkeypatch.setattr(optics, 'inject_point_sources', maybe_fail_inject)
    monkeypatch.setattr(camera, 'generate_detector_image', maybe_fail_detector)
    monkeypatch.setattr(outputs, 'save_hdu_to_fits', maybe_fail_writer)
    with pytest.raises(RuntimeError, match='deliberate failure'):
        observation.generate_observation_sequence(
            SimpleNamespace(point_source_dra=0), optics, camera, 20, 0,
            satspots_are_present=True, satspots_detector=camera, satspots_exptime=5,
            satspots_number_of_frames=2,
            satspot_keywords={'custom_pattern': np.ones((48, 48)), 'scale': 1.0},
            save_as_fits=True, full_frame=True)
    expected_removals = {None: [], 'positive': ['positive'], 'negative': ['positive', 'negative']}
    assert optics.removals == expected_removals[sign]
    assert optics.SATSPOTS == 0 and optics.satspot_info is None
    assert np.allclose(optics.optics_keywords['dm1_v'], 0)


@pytest.mark.parametrize('options, message', [
    ({'frame_times_utc': []}, 'requires full_frame'),
    ({'frame_times_utc': [], 'full_frame': True, 'save_as_fits': True}, 'contain 2'),
    ({'frame_times_utc': ['2027-06-01T00:00:00Z'] * 2,
      'full_frame': True, 'save_as_fits': True}, 'timezone-naive'),
])
def test_shared_timestamp_validation_precedes_propagation(options, message):
    optics = MockOptics()
    with pytest.raises(ValueError, match=message):
        observation.generate_observation_sequence(object(), optics, MockDetector(), 5, 2, **options)
    assert optics.scenes == []


def test_shared_sequence_missing_probe_keywords_fails_before_propagation():
    optics = MockOptics()
    with pytest.raises(ValueError, match='satspot keywords'):
        observation.generate_observation_sequence(
            object(), optics, MockDetector(), 20, 0,
            satspots_are_present=True, satspots_detector=MockDetector(),
            satspots_exptime=5, satspots_number_of_frames=2)
    assert optics.scenes == [] and optics.SATSPOTS == 0
