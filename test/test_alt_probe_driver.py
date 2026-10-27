"""Focused mapping and mocked execution of the Alternate Probe example."""

from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from astropy.io import fits
import numpy as np
import pytest

from corgisim import instrument, observation, outputs

add_satspot = instrument.CorgiOptics.add_satspot
remove_satspot = instrument.CorgiOptics.remove_satspot

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/alt_probe_generate_L1_sims.py'
DATA = Path(__file__).parent / 'test_data'
spec = importlib.util.spec_from_file_location('alt_probe_driver', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
CAMERA = 'visit_list/cgi_visit/cgi_excam/'


@pytest.mark.parametrize('full_campaign', [False, True])
def test_cli_selects_default_or_full_campaign_and_maps_input_root(tmp_path, monkeypatch, full_campaign):
    calls = []
    monkeypatch.setattr(driver, 'run_campaign', lambda **kwargs: calls.append(kwargs))
    args = ['--input-root', str(tmp_path / 'inputs'), '--output-dir', str(tmp_path / 'out'),
            '--scale', '0.5', '--overwrite']
    if full_campaign:
        args.append('--full-campaign')
    driver.main(args)
    expected = driver.FULL_CAMPAIGN if full_campaign else driver.OBSERVATIONS
    assert len(expected) == (23 if full_campaign else 2)
    assert [label for label, _, _ in calls[0]['observations']] == [row[0] for row in expected]
    for (_, xml, probe), (_, mapped_xml, mapped_probe) in zip(expected, calls[0]['observations']):
        assert mapped_xml == tmp_path / 'inputs' / xml.relative_to(driver.INPUT_ROOT)
        assert mapped_probe == (tmp_path / 'inputs' / probe.relative_to(driver.INPUT_ROOT) if probe else None)
    assert calls[0]['output_dir'] == tmp_path / 'out'
    assert calls[0]['scale'] == 0.5 and calls[0]['overwrite'] is True


def test_cli_without_flags_preserves_configured_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(driver, 'run_campaign', lambda **kwargs: calls.append(kwargs))
    driver.main([])
    assert calls == [dict(observations=driver.OBSERVATIONS, scale=driver.SCALE,
                          output_dir=driver.DEFAULT_OUTPUT_DIR, overwrite=driver.OVERWRITE)]
    assert [row[0] for row in driver.OBSERVATIONS] == ['A', 'B']


def test_root_override_preserves_manual_xml_and_external_probe_paths(tmp_path, monkeypatch):
    external = tmp_path / 'separate_probes/custom.fits'
    external_xml = tmp_path / 'separate_xml/visit.xml'
    selected = (('manual', None, external), ('custom', external_xml, driver.alternative_probe2))
    monkeypatch.setattr(driver, 'OBSERVATIONS', selected)
    assert driver.select_observations() is selected
    assert driver.select_observations(tmp_path / 'inputs') == (
        ('manual', None, external),
        ('custom', external_xml, tmp_path / 'inputs' / driver.alternative_probe2.relative_to(driver.INPUT_ROOT)),
    )
    assert driver.select_observations(full_campaign=True) is driver.FULL_CAMPAIGN


def inputs(tmp_path):
    entries = []
    for index, letter in enumerate('bcd'):
        probe = tmp_path / f'gaussian{index}.fits'
        fits.writeto(probe, np.arange(48 * 48, dtype=float).reshape(48, 48) / 10000 + index)
        entries.append((letter.upper(), DATA / f'alternative_probe_1A_probe{index + 1}.xml', probe))
    return entries


def edit_xml(tmp_path, changes=(), remove=None):
    root = ET.parse(DATA / 'alternative_probe_1A_probe1.xml').getroot()
    for path, value in changes:
        root.find(path).text = value
    if remove:
        parent, _, leaf = remove.rpartition('/')
        root.find(parent).remove(root.find(remove))
    path = tmp_path / 'edited.xml'
    ET.ElementTree(root).write(path)
    return path


@pytest.fixture
def mocked_generation(monkeypatch):
    frames, optical_objects, hosts, copies = [], [], [], []
    base1, base2 = np.full((48, 48), 8.), np.full((48, 48), 9.)

    class Optics:
        add_satspot = add_satspot

        def __init__(self, mode, bandpass, optics_keywords, **metadata):
            self.optics_keywords = optics_keywords
            self.metadata = metadata
            self.mode, self.bandpass = mode, bandpass
            self.SATSPOTS, self.satspot_info = 0, None
            self.removals = 0
            optical_objects.append(self)

        def get_host_star_psf(self, host):
            return dict(optics=self, dm1=self.optics_keywords['dm1_v'].copy(),
                        tag=self.SATSPOTS, info=self.satspot_info)

        def remove_satspot(self, satspot_keywords):
            remove_satspot(self, satspot_keywords)
            self.removals += 1

    class Detector:
        def __init__(self, keywords, photon_counting):
            self.gain, self.pc = keywords['em_gain'], photon_counting

        def generate_detector_image(self, sim_scene, exptime, **kwargs):
            primary = fits.PrimaryHDU()
            for key, value in (sim_scene['info'] or {}).items():
                primary.header['COMMENT'] = f'{key} : {value}'
            image = fits.ImageHDU(sim_scene['dm1'].copy())
            image.header.update(SATSPOTS=sim_scene['tag'], EXPTIME=exptime,
                                EMGAIN_C=self.gain, ISPC=int(self.pc),
                                EACQ_ROW=kwargs['loc_y'], EACQ_COL=kwargs['loc_x'])
            assert kwargs['full_frame'] is True
            return SimpleNamespace(image_on_detector=fits.HDUList([primary, image]))

    def save(hdul, **kwargs):
        optics = optical_objects[-1]
        header = hdul[1].header
        assert kwargs['write_as_L1'] is True
        frames.append(dict(psf_dm=hdul[1].data.copy(), dm1=optics.optics_keywords['dm1_v'].copy(),
                           dm2=optics.optics_keywords['dm2_v'].copy(), tag=header['SATSPOTS'],
                           info=optics.satspot_info, gain=header['EMGAIN_C'], pc=bool(header['ISPC']),
                           exptime=header['EXPTIME'], visitid=kwargs['overwrite_pri_keywords']['VISITID'],
                           vistype=kwargs['overwrite_pri_keywords']['VISTYPE'],
                           stamp=kwargs['overwrite_ext_keywords']['FTIMEUTC'],
                           outdir=kwargs['outdir'], location=(header['EACQ_COL'], header['EACQ_ROW']),
                           overwrite=kwargs['overwrite']))

    def host(properties, **kwargs):
        hosts.append((properties, kwargs))
        return properties

    monkeypatch.setattr(driver, 'load_base_dm', lambda *a: (base1.copy(), base2.copy()))
    monkeypatch.setattr(driver.roman_preflight_proper, 'copy_here', lambda: copies.append(True))
    monkeypatch.setattr(driver.scene, 'Scene', host)
    monkeypatch.setattr(driver.instrument, 'CorgiOptics', Optics)
    monkeypatch.setattr(driver.instrument, 'CorgiDetector', Detector)
    monkeypatch.setattr(outputs, 'save_hdu_to_fits', save)
    return SimpleNamespace(frames=frames, optics=optical_objects, hosts=hosts, copies=copies, base1=base1, base2=base2)


@pytest.mark.parametrize('probe,obsnum', [(1, 911), (2, 912), (3, 913)])
def test_xml_mapping(probe, obsnum):
    p = driver.read_alt_probe_xml(DATA / f'alternative_probe_1A_probe{probe}.xml')
    assert p['obsnum'] == obsnum and p['visnum'] == 1
    assert p['host'] == dict(target_name='beta Car', RA=138.299875, DEC=-69.71719444, Vmag=1.69, spectral_type='A1')
    assert (p['bandpass'], p['position_angle'], p['vistype']) == ('1A', 149.07, 'CGIVST_TDD_OBS')
    assert p['satellite'] == dict(exptime=19, gain=281, count=10, photon_counting=False)
    assert p['science'] == dict(exptime=28, gain=1181, count=34, photon_counting=False)
    assert p['requested_start'] == '2026-12-20T12:00:00'


@pytest.mark.parametrize('overwrite', [False, True])
def test_all_192_frames_and_state_at_every_frame(tmp_path, mocked_generation, monkeypatch, overwrite):
    entries = inputs(tmp_path)
    source_bytes = {path: path.read_bytes() for _, xml, probe in entries for path in (xml, probe)}
    monkeypatch.setattr(observation.copy, 'deepcopy', lambda image: pytest.fail('retained full-frame image'))
    result = driver.run_campaign(observations=entries, output_dir=tmp_path / 'out', overwrite=overwrite)
    recorded = mocked_generation
    assert len(recorded.frames) == 192 and len(recorded.copies) == 1
    assert len(recorded.optics) == 3
    for index, optics in enumerate(recorded.optics):
        pattern = fits.getdata(entries[index][2])
        frames = recorded.frames[index * 64:(index + 1) * 64]
        assert optics.removals == 2
        assert optics.bandpass == '1A' and optics.metadata['roll_angle'] == 149.07
        assert optics.metadata['visit_id'] == f'0200001001001{911+index}001'
        assert recorded.hosts[index][0]['target_name'] == 'beta Car'
        assert recorded.hosts[index][0]['RA'] == 138.299875
        for n, frame in enumerate(frames):
            sign = 1 if 10 <= n < 20 else -1 if 20 <= n < 30 else 0
            np.testing.assert_allclose(frame['dm1'], recorded.base1 + sign * pattern)
            np.testing.assert_allclose(frame['psf_dm'], frame['dm1'])
            np.testing.assert_array_equal(frame['dm2'], recorded.base2)
            assert frame['tag'] == int(n < 30)
            assert (frame['gain'], frame['exptime']) == ((281, 19) if n < 30 else (1181, 28))
            assert not frame['pc'] and frame['location'] == (512, 512)
            assert frame['vistype'] == 'CGIVST_TDD_OBS'
            assert frame['outdir'].name == 'V' + frame['visitid']
            assert frame['info'] is None if not sign else frame['info']['satspot_sign'] == ('positive' if sign > 0 else 'negative')
            assert frame['overwrite'] is overwrite
        assert optics.SATSPOTS == 0 and optics.satspot_info is None
        np.testing.assert_allclose(optics.optics_keywords['dm1_v'], recorded.base1, rtol=0, atol=1e-12)
    stamps = [datetime.fromisoformat(f['stamp']) for f in recorded.frames]
    assert [stamps[i].isoformat() for i in (0, 64, 128)] == [
        '2026-12-20T12:00:00', '2026-12-20T12:25:22', '2026-12-20T12:50:44']
    assert stamps[-1] + timedelta(seconds=28) == datetime(2026, 12, 20, 13, 16, 6)
    for previous, current in zip(recorded.frames, recorded.frames[1:]):
        assert datetime.fromisoformat(previous['stamp']) + timedelta(seconds=previous['exptime']) == datetime.fromisoformat(current['stamp'])
    assert not list(result.glob('*.json'))
    assert all(path.read_bytes() == data for path, data in source_bytes.items())


def test_no_xml_uses_hardcoded_trio_without_reading_xml(tmp_path, mocked_generation, monkeypatch):
    entries = inputs(tmp_path)
    monkeypatch.setattr(driver, 'read_alt_probe_xml', lambda _: pytest.fail('manual route read XML'))
    monkeypatch.setattr(driver, 'EXPTIME_S', 11.)
    monkeypatch.setattr(driver, 'EM_GAIN', 500.)
    driver.run_campaign(observations=[('manual', None, entries[0][2])], output_dir=tmp_path / 'out', scale=0.3)
    frames = mocked_generation.frames
    assert len(frames) == 9
    assert {f['exptime'] for f in frames} == {11}
    assert {f['gain'] for f in frames} == {500}
    assert frames[0]['stamp'] == driver.CAMPAIGN_START_TIME_UTC
    assert (datetime.fromisoformat(frames[1]['stamp']) - datetime.fromisoformat(frames[0]['stamp'])).total_seconds() == driver.FRAME_TIME_STEP_S
    assert frames[0]['visitid'] == '0200001001001001001'
    assert mocked_generation.optics[0].bandpass == driver.BANDPASS
    assert mocked_generation.hosts[0][0]['spectral_type'] == 'G0V'
    pattern = fits.getdata(entries[0][2])
    np.testing.assert_allclose(frames[3]['dm1'], mocked_generation.base1 + 0.3 * pattern)


@pytest.mark.parametrize('exposure,step', [(0.1, 1), (10, 30)])
def test_manual_exposure_and_spacing_are_separate(tmp_path, mocked_generation, monkeypatch, exposure, step):
    entries = inputs(tmp_path)
    monkeypatch.setattr(driver, 'EXPTIME_S', exposure)
    monkeypatch.setattr(driver, 'FRAME_TIME_STEP_S', step)
    driver.run_campaign(observations=[('manual', None, entries[0][2])], output_dir=tmp_path / 'out')
    first, second = mocked_generation.frames[:2]
    assert first['exptime'] == exposure
    assert (datetime.fromisoformat(second['stamp']) - datetime.fromisoformat(first['stamp'])).total_seconds() == step


@pytest.mark.parametrize('exposure,step', [(10, 9), (0.1, 0.5)])
def test_manual_invalid_spacing_fails_before_copy(tmp_path, mocked_generation, monkeypatch, exposure, step):
    entries = inputs(tmp_path)
    monkeypatch.setattr(driver, 'EXPTIME_S', exposure)
    monkeypatch.setattr(driver, 'FRAME_TIME_STEP_S', step)
    with pytest.raises(ValueError, match='frame spacing'):
        driver.run_campaign(observations=[('manual', None, entries[0][2])], output_dir=tmp_path / 'out')
    assert not mocked_generation.copies and not (tmp_path / 'out').exists()


def test_xml_overrides_hardcoded_camera_star_and_band(tmp_path, mocked_generation, monkeypatch):
    entries = inputs(tmp_path)
    path = edit_xml(tmp_path, [(CAMERA + 'frame_time', '31'), ('cpgs_input/target_exptime', '31')])
    monkeypatch.setattr(driver, 'HOST_STAR_PROPERTIES', {'Vmag': 99})
    monkeypatch.setattr(driver, 'EM_GAIN', 2.)
    monkeypatch.setattr(driver, 'BANDPASS', '4F')
    driver.run_campaign(observations=[('B', path, entries[0][2])], output_dir=tmp_path / 'out')
    assert mocked_generation.frames[-1]['exptime'] == 31
    assert mocked_generation.frames[-1]['gain'] == 1181
    assert mocked_generation.optics[0].bandpass == '1A'
    assert mocked_generation.hosts[0][0]['Vmag'] == 1.69


@pytest.mark.parametrize('field,value', [
    ('cpgs_input/observation_number', '912'), ('cpgs_input/target_gain', '2'),
    ('cpgs_input/satellite_spots_exptime_targ', '28'), ('cpgs_input/satellite_spots_nframes_targ', '5'),
    ('cpgs_input/with_polarization', '1'), (CAMERA + 'auto_gain', 'Y'),
    ('visit_list/cgi_visit/fixed_target/target_id', 'missing'),
])
def test_conflicts_and_unsupported_settings_fail(tmp_path, field, value):
    path = edit_xml(tmp_path, [(field, value)])
    with pytest.raises(ValueError, match=field.split('/')[-1]):
        driver.read_alt_probe_xml(path)


def test_missing_required_value_and_invalid_xml_never_fall_back(tmp_path, mocked_generation):
    entries = inputs(tmp_path)
    path = edit_xml(tmp_path, remove=CAMERA + 'frame_time')
    with pytest.raises(ValueError, match='frame_time'):
        driver.read_alt_probe_xml(path)
    path.write_text('<html>download page</html>')
    with pytest.raises(ValueError, match='obs_specification'):
        driver.run_campaign(observations=[entries[0], ('bad', path, entries[-1][2])], output_dir=tmp_path / 'out')
    path.write_text('<obs_specification>')
    with pytest.raises(ValueError, match='raw observation XML'):
        driver.read_alt_probe_xml(path)
    assert not (tmp_path / 'out').exists() and not mocked_generation.copies


@pytest.mark.parametrize('start', ['2026-12-20T12:00:00Z', '2026-12-20T07:00:00-05:00'])
def test_utc_normalization(tmp_path, mocked_generation, start):
    entries = inputs(tmp_path)
    path = edit_xml(tmp_path, [('cpgs_input/requested_start', start)])
    driver.run_campaign(observations=[('B', path, entries[0][2])], output_dir=tmp_path / 'out')
    assert mocked_generation.frames[0]['stamp'] == '2026-12-20T12:00:00'


def test_invalid_later_probe_and_duplicate_identity_fail_before_writes(tmp_path, mocked_generation):
    entries = inputs(tmp_path)
    with pytest.raises(ValueError, match='duplicate VISITID'):
        driver.run_campaign(observations=[entries[0], ('other label', entries[0][1], entries[-1][2])], output_dir=tmp_path / 'out')
    fits.writeto(entries[-1][2], np.zeros((10, 10)), overwrite=True)
    with pytest.raises(ValueError, match='does not match DM shape'):
        driver.run_campaign(observations=entries, output_dir=tmp_path / 'out')
    assert not mocked_generation.copies and not (tmp_path / 'out').exists()


@pytest.mark.parametrize('scale', [None, True, float('nan')])
def test_invalid_scale_before_outputs(tmp_path, mocked_generation, scale):
    with pytest.raises((ValueError, TypeError)):
        driver.run_campaign(observations=inputs(tmp_path), scale=scale, output_dir=tmp_path / 'out')
    assert not mocked_generation.copies and not (tmp_path / 'out').exists()


def test_existing_last_output_and_invalid_ancestor_fail_before_copy(tmp_path, mocked_generation):
    entries = inputs(tmp_path)
    out = tmp_path / 'out'
    visitid = '0200001001001913001'
    last = out / ('V' + visitid) / driver.planned_l1_filename(visitid, '2026-12-20T13:15:38')
    last.parent.mkdir(parents=True)
    last.write_text('existing')
    with pytest.raises(FileExistsError, match='overwrite=False'):
        driver.run_campaign(observations=entries, output_dir=out)
    assert not (out / 'V0200001001001911001').exists()
    ancestor = tmp_path / 'file'
    ancestor.write_text('not a directory')
    with pytest.raises(NotADirectoryError):
        driver.run_campaign(observations=entries, output_dir=ancestor / 'child')
    assert not mocked_generation.copies


@pytest.mark.parametrize('failed_index', [12, 22])
def test_probe_restoration_on_frame_failure(tmp_path, mocked_generation, monkeypatch, failed_index):
    entries = inputs(tmp_path)
    capture = outputs.save_hdu_to_fits
    def fail(*args, **kwargs):
        if len(mocked_generation.frames) == failed_index:
            raise RuntimeError('injected writer error')
        capture(*args, **kwargs)
    monkeypatch.setattr(outputs, 'save_hdu_to_fits', fail)
    with pytest.raises(RuntimeError, match='injected'):
        driver.run_campaign(observations=entries, output_dir=tmp_path / 'out')
    optics = mocked_generation.optics[0]
    np.testing.assert_allclose(optics.optics_keywords['dm1_v'], mocked_generation.base1, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(optics.optics_keywords['dm2_v'], mocked_generation.base2)
    assert optics.SATSPOTS == 0 and optics.satspot_info is None
    assert not (tmp_path / 'out/run_summary.json').exists()


def test_writer_filename_rounding_and_components():
    visitid = driver.build_visitid('0200', '001', '001', '001', 911, '001')
    assert driver.planned_l1_filename(visitid, '2026-12-20T23:59:59.950000') == f'cgi_{visitid}_20261221t0000000_l1_.fits'
    with pytest.raises(ValueError, match='VISITID'):
        driver.build_visitid('02000', '01', '001', '001', 911, '001')


def setup_xml(tmp_path, band='1A', obsnum=910, changes=()):
    root = ET.parse(DATA / 'alternative_probe_CFAM_1A.xml').getroot()
    for field, value in [
        ('observation_number', str(obsnum)), ('cpgs_input/observation_number', str(obsnum)),
        ('visit_list/cgi_visit/cgi_mechanisms/cfampos', band), ('cpgs_input/adv_set_cfam_pos', band),
        *changes,
    ]:
        root.find(field).text = value
    path = tmp_path / f'synthetic_setup_{obsnum}.xml'
    ET.ElementTree(root).write(path)
    return path


def test_setup_a_then_bcd_keeps_192_frames_and_only_three_visit_folders(tmp_path, mocked_generation, capsys):
    entries = [('A', DATA / 'alternative_probe_CFAM_1A.xml', None), *inputs(tmp_path)]
    out = driver.run_campaign(observations=entries, output_dir=tmp_path / 'out')
    assert len(mocked_generation.frames) == 192
    assert len(mocked_generation.optics) == 3
    assert len(mocked_generation.hosts) == 3
    assert len(list(out.glob('V*'))) == 3
    assert not (out / 'V0200001001001910001').exists()
    frames = mocked_generation.frames
    assert [frames[i]['stamp'] for i in (0, 64, 128)] == [
        '2026-12-20T12:00:00', '2026-12-20T12:25:22', '2026-12-20T12:50:44']
    assert datetime.fromisoformat(frames[-1]['stamp']) + timedelta(seconds=28) == datetime(2026, 12, 20, 13, 16, 6)
    assert all(o.optics_keywords['use_dm1'] == o.optics_keywords['use_dm2'] == 1 for o in mocked_generation.optics)
    console = capsys.readouterr().out
    assert 'CFAM=1A' in console and 'setup duration 70.9s omitted; no images' in console
    assert 'shift=3044s' in console
    assert not list(out.glob('*.json'))


def test_setup_only_needs_no_optical_resources_or_imaging_folder(tmp_path, mocked_generation, monkeypatch, capsys):
    monkeypatch.setattr(driver, 'load_base_dm', lambda *a: pytest.fail('setup read base DMs'))
    monkeypatch.setattr(driver, 'load_probe_pattern', lambda *a: pytest.fail('setup read probe'))
    out = driver.run_campaign(observations=[('A', DATA / 'alternative_probe_CFAM_1A.xml', None)], output_dir=tmp_path / 'out')
    assert not mocked_generation.frames and not mocked_generation.optics and not mocked_generation.copies
    assert not out.exists()
    assert 'CFAM=1A' in capsys.readouterr().out


def test_setup_time_does_not_anchor_or_shift_imaging(tmp_path, mocked_generation):
    early = setup_xml(tmp_path, changes=[('cpgs_input/requested_start', '2026-12-20T11:00:00')])
    driver.run_campaign(observations=[('A', early, None), *inputs(tmp_path)], output_dir=tmp_path / 'out')
    assert mocked_generation.frames[0]['stamp'] == '2026-12-20T12:00:00'


def test_setup_band_mismatch_is_error_before_any_writes(tmp_path, mocked_generation):
    different_band = setup_xml(tmp_path, band='1B')
    with pytest.raises(ValueError, match='conflicts with setup band'):
        driver.run_campaign(observations=[('A', different_band, None), *inputs(tmp_path)], output_dir=tmp_path / 'out')
    assert not (tmp_path / 'out').exists() and not mocked_generation.copies and not mocked_generation.optics


def test_later_synthetic_setup_and_matching_imaging(tmp_path, mocked_generation, capsys):
    entries = [('A', DATA / 'alternative_probe_CFAM_1A.xml', None), *inputs(tmp_path)]
    later_setup = setup_xml(tmp_path, band='1B', obsnum=914)
    later_image = edit_xml(tmp_path, [
        ('observation_number', '915'), ('cpgs_input/observation_number', '915'),
        ('cpgs_input/howfsc_cfam_pos', '1B'),
    ])
    entries += [('synthetic E', later_setup, None), ('synthetic F', later_image, entries[1][2])]
    out = driver.run_campaign(observations=entries, output_dir=tmp_path / 'out')
    assert len(mocked_generation.frames) == 256 and len(mocked_generation.optics) == 4
    assert mocked_generation.optics[-1].bandpass == '1B'
    assert mocked_generation.frames[192]['stamp'] == '2026-12-20T13:16:06'
    console = capsys.readouterr().out
    assert 'CFAM=1A' in console and 'CFAM=1B' in console
    assert len(list(out.glob('V*'))) == 4


def test_historical_json_is_untouched_and_not_an_output_dependency(tmp_path, mocked_generation):
    entries = inputs(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    historical = out / 'run_summary.json'
    historical.write_text('historical artifact, not read by generation')
    driver.run_campaign(observations=[entries[0]], output_dir=out)
    assert historical.read_text() == 'historical artifact, not read by generation'
    assert len(mocked_generation.frames) == 64


@pytest.mark.parametrize('field,value', [
    ('visit_list/cgi_visit/cgi_mechanisms/set_dms', 'Y'),
    ('visit_list/cgi_visit/cgi_mechanisms/spampos', 'OPEN'),
    ('visit_list/cgi_visit/cgi_advance_setup/float_bin_1', '1'),
    ('cpgs_input/adv_set_excam_on', '1'),
    ('cpgs_input/adv_set_cfam_pos', '1B'),
    ('visit_list/cgi_visit/cgi_excam/exposure_duration', '1'),
])
def test_additional_active_setup_commands_are_rejected(tmp_path, mocked_generation, field, value):
    path = setup_xml(tmp_path, changes=[(field, value)])
    with pytest.raises(ValueError, match=field.split('/')[-1]):
        driver.run_campaign(observations=[('A', path, None), *inputs(tmp_path)], output_dir=tmp_path / 'out')
    assert not mocked_generation.copies and not (tmp_path / 'out').exists()
