"""Fast, non-propagating checks that the Alternate Probe campaign driver keeps the
simulated visit metadata consistent with the VISITID/VISTYPE written to L1.

Background: one ``CorgiOptics`` object serves every probe execution in a campaign,
but each execution has its own VISITID. ``CorgiOptics`` snapshots ``visit_id`` and
``visit_type`` into the simulation-metadata block when the scene is generated
(``corgisim/instrument.py`` -- ``sim_info`` in ``get_host_star_psf``), and that block
becomes the ``visit_id :``/``visit_type :`` COMMENT cards of the L1 primary header
(``corgisim/outputs.py`` -- ``add_comment`` over ``sim_info``). Overriding only the
VISITID keyword at write time therefore leaves the COMMENT provenance pointing at the
``CorgiOptics`` default, which contradicts the header for every probe after the first.

These tests replace the optics, detector and scene with recording stubs, so they never
run a PROPER propagation and never write a FITS file; the whole module runs in
milliseconds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

DRIVER_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'alt_probe_generate_L1_sims.py'

# CorgiOptics defaults, mirrored so the stub misbehaves exactly as the real class would
# if the driver failed to synchronize them (corgisim/instrument.py).
CORGIOPTICS_DEFAULT_VISIT_ID = '0200001001001001001'
CORGIOPTICS_DEFAULT_VISIT_TYPE = 'CGIVST_TDD_OBS'


def load_driver():
    """Import the driver script (which lives in examples/, not the package)."""
    spec = importlib.util.spec_from_file_location('alt_probe_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()


class RecordingOptics:
    """Stand-in for CorgiOptics that records the visit metadata a scene would carry."""

    def __init__(self, cgi_mode=None, bandpass=None, optics_keywords=None, **kwargs):
        self.optics_keywords = dict(optics_keywords or {})
        self.visit_type = kwargs.get('visit_type', CORGIOPTICS_DEFAULT_VISIT_TYPE)
        self.visit_id = kwargs.get('visit_id', CORGIOPTICS_DEFAULT_VISIT_ID)
        self.construction_kwargs = dict(kwargs)
        # (visit_id, visit_type) captured at each scene generation, in order.
        self.scene_metadata = []

    def get_host_star_psf(self, input_scene, **kwargs):
        self.scene_metadata.append((self.visit_id, self.visit_type))
        return object()

    def add_satspot(self, satspot_keywords=None):
        pass

    def remove_satspot(self, satspot_keywords=None):
        pass


class DummyDetector:
    """Stand-in for CorgiDetector; the driver only constructs and passes it on."""

    def __init__(self, emccd_keywords=None, photon_counting=False):
        pass


def write_probe(tmp_path, name):
    """Write a synthetic 48x48 DM probe and return its path."""
    y, x = np.indices((48, 48))
    pattern = 0.8 * np.exp(-((x - 37) ** 2 + (y - 32) ** 2) / 2.0)
    path = tmp_path / name
    fits.writeto(str(path), pattern, overwrite=True)
    return str(path)


@pytest.fixture
def stubbed_driver(monkeypatch):
    """Patch out propagation, DM loading, scene building and frame writing.

    Returns ``(driver, saved, created_optics)``, where ``saved`` collects every
    ``(visitid, vistype)`` pair handed to ``save_frame`` and ``created_optics``
    collects the RecordingOptics instances the driver built.
    """
    saved = []
    created_optics = []

    def optics_factory(*args, **kwargs):
        optics = RecordingOptics(*args, **kwargs)
        created_optics.append(optics)
        return optics

    def fake_save_frame(sim_scene, detector, exptime, loc_x, loc_y, outdir, visitid,
                        vistype, ftimeutc, overwrite=False):
        saved.append((visitid, vistype))

    monkeypatch.setattr(driver.instrument, 'CorgiOptics', optics_factory)
    monkeypatch.setattr(driver.instrument, 'CorgiDetector', DummyDetector)
    monkeypatch.setattr(driver.scene, 'Scene', lambda properties: object())
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here', lambda: None)
    monkeypatch.setattr(driver, 'load_base_dm',
                        lambda *args, **kwargs: (np.zeros((48, 48)), np.zeros((48, 48))))
    monkeypatch.setattr(driver, 'save_frame', fake_save_frame)
    return driver, saved, created_optics


def run_two_probe_campaign(tmp_path, stubbed_driver, **overrides):
    """Run a stubbed two-probe campaign and return (optics, saved, expected_visitids)."""
    drv, saved, created_optics = stubbed_driver

    kwargs = dict(
        scale=1.0,
        probe_files=[write_probe(tmp_path, 'probe_a.fits'),
                     write_probe(tmp_path, 'probe_b.fits')],
        output_dir=str(tmp_path / 'out'),
        n_frames_per_state=1,
    )
    kwargs.update(overrides)
    drv.run_campaign(**kwargs)

    assert len(created_optics) == 1, 'the campaign must reuse a single CorgiOptics'

    expected = [
        drv.build_visitid(kwargs.get('visit_prognum', drv.VISIT_PROGNUM),
                          kwargs.get('visit_execnum', drv.VISIT_EXECNUM),
                          kwargs.get('visit_campaign', drv.VISIT_CAMPAIGN),
                          kwargs.get('visit_segment', drv.VISIT_SEGMENT),
                          kwargs.get('visit_obsnum_start', drv.VISIT_OBSNUM_START) + i,
                          kwargs.get('visit_visnum', drv.VISIT_VISNUM))
        for i in range(2)
    ]
    return created_optics[0], saved, expected


def test_scene_visit_id_matches_visitid_for_every_probe(tmp_path, stubbed_driver):
    """Each probe execution's scene metadata carries that probe's own VISITID."""
    optics, saved, expected = run_two_probe_campaign(tmp_path, stubbed_driver)

    assert expected[0] != expected[1], 'probe executions must have distinct VISITIDs'

    # Three DM states per probe -> three scene generations per probe.
    assert len(optics.scene_metadata) == 6
    assert len(saved) == 6

    scene_visit_ids = [visit_id for visit_id, _ in optics.scene_metadata]
    assert scene_visit_ids == [expected[0]] * 3 + [expected[1]] * 3

    # The visit_id feeding the COMMENT block matches the VISITID keyword written
    # for the same frame, for every frame in the campaign.
    for (scene_visit_id, _), (frame_visitid, _) in zip(optics.scene_metadata, saved):
        assert scene_visit_id == frame_visitid

    # Regression guard: the second probe must not keep the CorgiOptics default.
    assert scene_visit_ids[3:] != [CORGIOPTICS_DEFAULT_VISIT_ID] * 3


def test_scene_visit_type_matches_non_default_vistype(tmp_path, stubbed_driver):
    """A non-default VISTYPE reaches CorgiOptics, so the COMMENT block matches."""
    custom_vistype = 'CGIVST_CAL_BORESIGHT'
    assert custom_vistype != CORGIOPTICS_DEFAULT_VISIT_TYPE

    optics, saved, _ = run_two_probe_campaign(tmp_path, stubbed_driver,
                                              vistype=custom_vistype)

    assert optics.construction_kwargs.get('visit_type') == custom_vistype
    assert [visit_type for _, visit_type in optics.scene_metadata] == [custom_vistype] * 6
    for (_, scene_visit_type), (_, frame_vistype) in zip(optics.scene_metadata, saved):
        assert scene_visit_type == frame_vistype == custom_vistype
