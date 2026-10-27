"""Shared acquisition coverage retained from the PR273 driver tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from corgisim import instrument, observation, outputs

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
