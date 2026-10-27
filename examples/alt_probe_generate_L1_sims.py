#!/usr/bin/env python
"""Generate Alternate Probe L1 trios, followed by science when XML requests it."""

import argparse
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import proper
import roman_preflight_proper
from astropy.io import fits

from corgisim import instrument, observation, outputs, scene
from corgisim.sat_spots import add_custom_pattern_dm


# --- Configuration: paths -------------------------------------------------
# Paths: edit these placeholders. Relative paths resolve from the working directory.
ALTERNATIVE_XML_ROOT = Path('/path/to/alternative/xml')
STANDARD_XML_ROOT = Path('/path/to/standard/xml')
PROBE_ROOT = Path('/path/to/probes')
DEFAULT_OUTPUT_DIR = Path('/path/to/output')
DM_ROOTNAME = 'hlc_ni_3e-8'
DM1_PATH = None  # Optional explicit paths; None uses the installed DM_ROOTNAME maps.
DM2_PATH = None

# --- Campaign observations ------------------------------------------------
probe1 = PROBE_ROOT / 'nfov_dm_dmrel_5.0e-07_gaussian0.fits'
probe2 = PROBE_ROOT / 'nfov_dm_dmrel_5.0e-07_gaussian1.fits'
probe3 = PROBE_ROOT / 'nfov_dm_dmrel_5.0e-07_gaussian2.fits'

# HLC_B1_C1: alternative probes (A-L). Temporary observation numbers/bands show the fixed order;
# executable identities/settings come from XML. Set XML to None for hard-coded imaging.
# Commented <...> paths are placeholders for XML files not yet supplied.
# No setup XML handy? Comment out A (and any other setup rows).
# Imaging XML supplies its own bandpass; without imaging XML, set BANDPASS below.
# Omit setup rows entirely rather than setting their XML path to None.
ALTERNATIVE_OBSERVATIONS = (
    ('A', ALTERNATIVE_XML_ROOT / 'observation_a/alternative_probe_CFAM_1A.xml', None),  # 910 setup 1A
    ('B', ALTERNATIVE_XML_ROOT / 'observation_b/alternative_probe_1A_probe1.xml', probe1),  # 911 1A
    # ('C', ALTERNATIVE_XML_ROOT / 'observation_c/alternative_probe_1A_probe2.xml', probe2),  # 912 1A
    # ('D', ALTERNATIVE_XML_ROOT / 'observation_d/alternative_probe_1A_probe3.xml', probe3),  # 913 1A
    # ('E', ALTERNATIVE_XML_ROOT / '<E_914_setup_1B.xml>', None),
    # ('F', ALTERNATIVE_XML_ROOT / '<F_915_probe1_1B.xml>', probe1),
    # ('G', ALTERNATIVE_XML_ROOT / '<G_916_probe2_1B.xml>', probe2),
    # ('H', ALTERNATIVE_XML_ROOT / '<H_917_probe3_1B.xml>', probe3),
    # ('I', ALTERNATIVE_XML_ROOT / '<I_918_setup_1C.xml>', None),
    # ('J', ALTERNATIVE_XML_ROOT / '<J_919_probe1_1C.xml>', probe1),
    # ('K', ALTERNATIVE_XML_ROOT / '<K_920_probe2_1C.xml>', probe2),
    # ('L', ALTERNATIVE_XML_ROOT / '<L_921_probe3_1C.xml>', probe3),
)
# HLC_B1_C1: standard probes (A-K). All XML filenames below are unsupplied placeholders;
# uncomment only after supplying and checking the actual inputs. No initial setup is assumed.
# Numbers are temporary observation numbers, not executable identity overrides.
STANDARD_OBSERVATIONS = (
    # ('A', STANDARD_XML_ROOT / '<A_922_probe1_1C.xml>', probe1),
    # ('B', STANDARD_XML_ROOT / '<B_923_probe2_1C.xml>', probe2),
    # ('C', STANDARD_XML_ROOT / '<C_924_probe3_1C.xml>', probe3),
    # ('D', STANDARD_XML_ROOT / '<D_925_setup_1A.xml>', None),
    # ('E', STANDARD_XML_ROOT / '<E_926_probe1_1A.xml>', probe1),
    # ('F', STANDARD_XML_ROOT / '<F_927_probe2_1A.xml>', probe2),
    # ('G', STANDARD_XML_ROOT / '<G_928_probe3_1A.xml>', probe3),
    # ('H', STANDARD_XML_ROOT / '<H_929_setup_1B.xml>', None),
    # ('I', STANDARD_XML_ROOT / '<I_981_probe1_1B.xml>', probe1),
    # ('J', STANDARD_XML_ROOT / '<J_982_probe2_1B.xml>', probe2),
    # ('K', STANDARD_XML_ROOT / '<K_983_probe3_1B.xml>', probe3),
)
OBSERVATIONS = ALTERNATIVE_OBSERVATIONS + STANDARD_OBSERVATIONS
# Without XML, replace the selection above with this tuple; it excludes setup entries.
# OBSERVATIONS = (
#     ('B', None, probe1),
#     # ('C', None, probe2),
#     # ('D', None, probe3),
# )

# --- Simulation controls --------------------------------------------------
SCALE = 1.0  # Amplitude belongs in probe files; multiplier is for exceptional adjustments.
OVERWRITE = False

# --- Hard-coded observation settings (used only without XML) ---------------
# The original trio has no target-science block.
HOST_STAR_PROPERTIES = {'Vmag': 5, 'spectral_type': 'G0V', 'target_name': 'example', 'RA': 0., 'DEC': 0.}
CGI_MODE = 'excam'
COR_TYPE = 'hlc_band1'
BANDPASS = '1A'  # Used when an imaging entry's XML path is None; imaging XML takes precedence.
POSITION_ANGLE = 0.
EXPTIME_S, EM_GAIN, PHOTON_COUNTING, N_FRAMES_PER_STATE = 10., 1000., False, 3
VISTYPE, VISIT_OBSNUM_START, VISIT_VISNUM = 'CGIVST_TDD_OBS', 1, '001'
CAMPAIGN_START_TIME_UTC = '2027-06-01T00:00:00'
FRAME_TIME_STEP_S = 30.  # Manual spacing must clear exposure and whole-second SCTSRT precision.

# --- Shared simulation parameters (also used with XML) ---------------------
# Voltage arrays are not transformed.
VISIT_PROGNUM, VISIT_EXECNUM, VISIT_CAMPAIGN, VISIT_SEGMENT = '0200', '001', '001', '001'
POLAXIS, OUTPUT_DIM, LOC_X, LOC_Y = 10, 255, 512, 512
MAGTYPE, SPECTRUM_METHOD = 'vegamag', 'bpgs'
SATELLITE_PHOTON_COUNTING = False  # Template convention; no separate XML field.
HOST_STAR_DEFAULTS = dict(magtype=MAGTYPE, ref_flag=False, stellar_diam_mas=None)
OPTICS_KEYWORDS = dict(use_errors=1, polaxis=POLAXIS, output_dim=OUTPUT_DIM, use_dm1=1,
                       use_dm2=1, use_fpm=1, use_lyot_stop=1, use_field_stop=1)
STELLAR_DIAM_AND_JITTER = None
DETECTOR_KEYWORDS = {}  # Resolved detector settings are already recorded in the FITS COMMENT cards.
IF_QUIET, FULL_FRAME = True, True
# First requested start is UTC; XML spacing is exposure duration, with no added gaps.
# No flight overheads, LOCAM control, proper-motion evolution, jitter or finite extent.
# Other optics/detector/writer defaults are inherited from Corgisim, not audited here.


def read_alt_probe_xml(path):
    """Extract one HLC imaging visit or a CFAM-only advance setup."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f'{path}: expected raw observation XML: {exc}') from exc
    if root.tag != 'obs_specification' or len(root.findall('visit_list/cgi_visit')) != 1:
        raise ValueError(f'{path}: expected obs_specification with one cgi_visit')

    def value(field, cast=str):
        nodes = root.findall(field)
        if len(nodes) != 1 or not nodes[0].text or not nodes[0].text.strip():
            raise ValueError(f'{path}: missing or repeated {field}')
        try:
            return cast(nodes[0].text.strip())
        except (ValueError, KeyError) as exc:
            raise ValueError(f'{path}: invalid {field}') from exc

    def agree(field, expected, cast=str):
        if value(field, cast) != expected:
            raise ValueError(f'{path}: conflicting or unsupported {field}; expected {expected!r}')

    visit, camera, cpgs = 'visit_list/cgi_visit/', 'visit_list/cgi_visit/cgi_excam/', 'cpgs_input/'
    obsnum = value('observation_number', int)
    agree(cpgs + 'observation_number', obsnum, int)
    try:
        visnum = int(root.find('visit_list/cgi_visit').attrib['number'])
    except (KeyError, ValueError) as exc:
        raise ValueError(f'{path}: invalid cgi_visit/@number') from exc
    if value(visit + 'cgi_visit_type') == 'CGIVST_UTI_ADVANCE_SETUP':
        mechanism = visit + 'cgi_mechanisms/'
        agree(visit + 'observation_type', 'CGI_UTIL')
        agree(visit + 'cgi_howfsc/is_howfsc_visit', 'N')
        agree(camera + 'exposure_duration', 0, float)
        agree(camera + 'observe_satellite_spots', 'N')
        for name in ('reset_fsm', 'reset_fine_fcm', 'set_dms'):
            agree(mechanism + name, 'N')
            agree(cpgs + 'adv_set_' + name, '0')
        for name in ('spam', 'fpam', 'fsam', 'lsam', 'dpam', 'fcm'):
            agree(mechanism + name + 'pos', 'NONE')
            agree(cpgs + 'adv_set_' + name + '_pos', 'NONE')
        for name in ('dm1', 'dm2', 'excam', 'fcm', 'fsm', 'locam'):
            agree(visit + 'cgi_states/' + name + '_on', 'N')
            agree(cpgs + 'adv_set_' + name + '_on', '0')
        agree(visit + 'cgi_states/cam_cts_htr_state', 'NO_CHANGE')
        agree(cpgs + 'adv_set_cam_decon_state', 'NO_CHANGE')
        agree(visit + 'cgi_advance_setup/star_ratio_lds_name', 'NONE')
        agree(cpgs + 'adv_set_lowfsc_sr_type', 'NONE')
        agree(cpgs + 'adv_set_lowfsc_sr_value', 0, float)
        agree(cpgs + 'adv_set_time_gap', 0, float)
        for prefix in ('float_bin_', 'crc_bin_', 'checksum_'):
            for index in range(1, 5):
                agree(visit + 'cgi_advance_setup/' + prefix + str(index), 0, float)
        band = value(mechanism + 'cfampos')
        agree(cpgs + 'adv_set_cfam_pos', band)
        if band not in ('1A', '1B', '1C'):
            raise ValueError(f'{path}: unsupported setup cfampos {band!r}')
        duration = value(visit + 'initial_overhead_duration', float) + value(visit + 'science_duration', float)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f'{path}: invalid setup duration')
        return dict(setup_band=band, obsnum=obsnum, visnum=visnum, vistype='CGIVST_UTI_ADVANCE_SETUP',
                    omitted_overhead_s=duration, requested_start=value(cpgs + 'requested_start'))
    for field, expected in (
        (visit + 'cgi_visit_type', 'CGIVST_TDD_OBS'), (visit + 'coronagraph_configuration', 'NFOV'),
        (visit + 'fixed_target/reference_target', 'N'), (visit + 'cgi_howfsc/is_howfsc_visit', 'N'),
        (visit + 'is_spectroscopy', 'N'), (camera + 'auto_gain', 'N'),
        (camera + 'observe_satellite_spots', 'Y'), (cpgs + 'target_autogain', '0'),
        (cpgs + 'observe_target_only', '1'), (cpgs + 'with_polarization', '0'),
        (cpgs + 'is_spectroscopy', '0'), (cpgs + 'dig_dark_hole', '0'),
        (cpgs + 'coronagraph_mask', '1'), (cpgs + 'filter', '1'),
    ):
        agree(field, expected)
    target_id = value(visit + 'fixed_target/target_id')
    targets = [t for t in root.findall('target_list/target') if t.findtext('target_id') == target_id]
    if len(targets) != 1:
        raise ValueError(f'{path}: target_id {target_id!r} must resolve to exactly one target')
    target_index = list(root.findall('target_list/target')).index(targets[0]) + 1
    target = f'target_list/target[{target_index}]/'
    host = {'target_name': value(target + 'target_name'), 'RA': value(target + 'ra_computed', float),
            'DEC': value(target + 'dec_computed', float), 'Vmag': value(cpgs + 'target_v_mag', float),
            'spectral_type': value(cpgs + 'target_spec_type') + f"{value(cpgs + 'target_sub_type', float):g}"}
    agree(cpgs + 'target_name', host['target_name'])
    satellite, science = {}, {}
    for settings, prefix, duplicate, suffix in (
        (satellite, 'satellite_spots_', 'satellite_spots_', '_targ'), (science, '', 'target_', ''),
    ):
        settings.update(exptime=value(camera + prefix + 'frame_time', float),
                        gain=value(camera + prefix + 'gain', float),
                        count=value(camera + prefix + 'number_of_frames', int))
        agree(cpgs + duplicate + 'exptime' + suffix, settings['exptime'], float)
        agree(cpgs + duplicate + 'gain' + suffix, settings['gain'], float)
    agree(cpgs + 'satellite_spots_nframes_targ', satellite['count'], int)
    agree(camera + 'satellite_spots_pair_id', 11, int)
    agree(cpgs + 'satellite_spots_pair_id', 11, int)
    science['photon_counting'] = value(camera + 'photon_counting', lambda s: {'Y': True, 'N': False}[s])
    agree(cpgs + 'target_pcounting', int(science['photon_counting']), int)
    satellite['photon_counting'] = SATELLITE_PHOTON_COUNTING
    bandpass = value(cpgs + 'howfsc_cfam_pos')
    if bandpass not in ('1A', '1B', '1C'):
        raise ValueError(f'{path}: unsupported howfsc_cfam_pos {bandpass!r}')
    return dict(host=host, bandpass=bandpass, cor_type='hlc_band1', cgi_mode='excam',
                position_angle=value(visit + 'position_angle', float), vistype='CGIVST_TDD_OBS',
                obsnum=obsnum, visnum=visnum, satellite=satellite, science=science,
                requested_start=value(cpgs + 'requested_start'))


def build_visitid(prognum, execnum, campaign, segment, obsnum, visnum):
    """Assemble a validated 19-digit VISITID."""
    parts = (prognum, execnum, campaign, segment, f'{obsnum:03d}', visnum)
    if any(not isinstance(p, str) or len(p) != n or not p.isascii() or not p.isdigit()
           for p, n in zip(parts, (4, 3, 3, 3, 3, 3))):
        raise ValueError('VISITID components require 4/3/3/3/3/3 digits')
    return ''.join(parts)


def load_base_dm(dm_rootname=DM_ROOTNAME, dm1_path=None, dm2_path=None):
    """Load base voltage maps from explicit or installed example paths."""
    dm1_path = dm1_path or f'{roman_preflight_proper.lib_dir}/examples/{dm_rootname}_dm1_v.fits'
    dm2_path = dm2_path or f'{roman_preflight_proper.lib_dir}/examples/{dm_rootname}_dm2_v.fits'
    return proper.prop_fits_read(dm1_path), proper.prop_fits_read(dm2_path)


def load_probe_pattern(path):
    return fits.getdata(path)


def probe_label(path):
    return Path(path).stem


def planned_l1_filename(visitid, ftimeutc):
    return f'cgi_{visitid}_{outputs.isotime_to_yyyymmddThhmmsss(ftimeutc)}_l1_.fits'


def run_campaign(*, observations=OBSERVATIONS, scale=SCALE, output_dir=DEFAULT_OUTPUT_DIR, overwrite=OVERWRITE):
    """Read/check all selected inputs, then acquire each visit in list order.

    Entries are (label, optional XML path, probe FITS path). None uses the
    hard-coded settings. Invalid XML always fails; it never selects defaults.
    """
    dm1 = dm2 = None
    visits, identities, planned_paths = [], set(), []
    setup_seen, current_band = False, None
    output_dir = Path(output_dir)
    time = None
    for index, (label, xml_path, probe_path) in enumerate(observations):
        p = read_alt_probe_xml(xml_path) if xml_path is not None else dict(
            host=HOST_STAR_PROPERTIES.copy(), cgi_mode=CGI_MODE, cor_type=COR_TYPE, bandpass=BANDPASS,
            position_angle=POSITION_ANGLE, vistype=VISTYPE, obsnum=VISIT_OBSNUM_START + index,
            visnum=int(VISIT_VISNUM), requested_start=CAMPAIGN_START_TIME_UTC, science=None,
            frame_time_step_s=FRAME_TIME_STEP_S,
            satellite=dict(exptime=EXPTIME_S, gain=EM_GAIN, count=N_FRAMES_PER_STATE, photon_counting=PHOTON_COUNTING))
        context = str(xml_path) if xml_path is not None else f'{label}: hard-coded settings'
        if not 1 <= p['obsnum'] <= 999 or not 1 <= p['visnum'] <= 999:
            raise ValueError(f'{context}: observation/visit number must be in 1..999')
        visitid = build_visitid(VISIT_PROGNUM, VISIT_EXECNUM, VISIT_CAMPAIGN, VISIT_SEGMENT,
                               p['obsnum'], f"{p['visnum']:03d}")
        if visitid in identities:
            raise ValueError(f'{context}: duplicate VISITID {visitid}')
        identities.add(visitid)
        try:
            if 'T' not in p['requested_start']:
                raise ValueError
            requested = datetime.fromisoformat(p['requested_start'])
        except ValueError as exc:
            raise ValueError(f'{context}: invalid requested_start') from exc
        if requested.tzinfo is not None:
            requested = requested.astimezone(timezone.utc).replace(tzinfo=None)
        if 'setup_band' in p:
            if probe_path is not None:
                raise ValueError(f'{context}: CFAM setup must not have a probe binding')
            current_band = p['setup_band']
            setup_seen = True
            print(f'{label} VISITID={visitid}: CFAM={current_band}, '
                  f'setup duration {p["omitted_overhead_s"]:g}s omitted; no images')
            continue
        if current_band is not None and p['bandpass'] != current_band:
            raise ValueError(f'{context}: bandpass {p["bandpass"]} conflicts with setup band {current_band}')
        if not 0 <= p['host']['RA'] < 360 or not -90 <= p['host']['DEC'] <= 90 \
                or not math.isfinite(p['host']['Vmag']) or not math.isfinite(p['position_angle']) \
                or not scene.is_valid_spectral_type(p['host']['spectral_type']):
            raise ValueError(f'{context}: invalid target coordinates, magnitude, spectral type or position angle')
        if time is None:
            time = requested
        if dm1 is None:
            dm1, dm2 = load_base_dm(DM_ROOTNAME, DM1_PATH, DM2_PATH)
            for dm in (dm1, dm2):
                add_custom_pattern_dm(np.zeros((48, 48)), dm, scale)
        if probe_path is None:
            raise ValueError(f'{context}: imaging requires a probe binding')
        pattern = load_probe_pattern(probe_path)
        for sign in ('positive', 'negative'):
            command = add_custom_pattern_dm(dm1, pattern, scale, sign)
            if not np.all(np.isfinite(command)):
                raise ValueError(f'{probe_path}: non-finite scaled DM command')
        outdir = output_dir / f'V{visitid}'
        frame_times = []
        cameras = [p['satellite']] * 3
        if p['science'] is not None:
            cameras.append(p['science'])
        for camera in cameras:
            step = p.get('frame_time_step_s', camera['exptime'])
            if isinstance(camera['count'], bool) or not isinstance(camera['count'], int) or camera['count'] <= 0 \
                    or not math.isfinite(camera['exptime']) or camera['exptime'] <= 0 \
                    or not math.isfinite(camera['gain']) or camera['gain'] <= 0 \
                    or not math.isfinite(step) or step < max(camera['exptime'], 1):
                raise ValueError(f'{context}: invalid count, exposure, gain or frame spacing')
            for _ in range(camera['count']):
                frame_times.append(time.isoformat())
                planned_paths.append(outdir / planned_l1_filename(visitid, frame_times[-1]))
                time += timedelta(seconds=step)
        visits.append((label, probe_path, p, pattern, visitid, outdir, frame_times, requested))
    if not visits and not setup_seen:
        raise ValueError('observations must not be empty')
    if not visits:
        return output_dir
    for path in planned_paths:
        if path.is_dir() or path.is_symlink():
            raise ValueError(f'output path is a directory or symlink: {path}')
        if path.exists() and not overwrite:
            raise FileExistsError(f'output already exists and overwrite=False: {path}')
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise NotADirectoryError(f'output ancestor is not a directory: {parent}')

    roman_preflight_proper.copy_here()
    # Spawned PROPER workers must import the prescriptions copied into this working directory.
    prescription_dir = str(Path.cwd())
    if prescription_dir not in sys.path:
        sys.path.insert(0, prescription_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Scale={scale}; base DM1={DM1_PATH or DM_ROOTNAME}, DM2={DM2_PATH or DM_ROOTNAME}. '
          'UTC timing: XML exposure steps, zero added overhead/gaps; manual FRAME_TIME_STEP_S.')
    for label, probe_path, p, pattern, visitid, outdir, frame_times, requested in visits:
        outdir.mkdir(exist_ok=True)
        host = dict(p['host'], **HOST_STAR_DEFAULTS)
        base_scene = scene.Scene(host, spmethod=SPECTRUM_METHOD)
        optics_keywords = dict(OPTICS_KEYWORDS, cor_type=p['cor_type'], dm1_v=dm1.copy(), dm2_v=dm2.copy())
        optics = instrument.CorgiOptics(p['cgi_mode'], p['bandpass'], optics_keywords=optics_keywords,
                                        roll_angle=p['position_angle'], if_quiet=IF_QUIET,
                                        stellar_diam_and_jitter_keywords=STELLAR_DIAM_AND_JITTER,
                                        visit_id=visitid, visit_type=p['vistype'])
        satellite, science = p['satellite'], p['science']
        satellite_detector = instrument.CorgiDetector(dict(DETECTOR_KEYWORDS, em_gain=satellite['gain']),
                                                     photon_counting=satellite['photon_counting'])
        detector = (instrument.CorgiDetector(dict(DETECTOR_KEYWORDS, em_gain=science['gain']),
                                            photon_counting=science['photon_counting'])
                    if science is not None else satellite_detector)
        observation.generate_observation_sequence(
            base_scene, optics, detector, science['exptime'] if science else satellite['exptime'],
            n_frames=science['count'] if science else 0,
            satspots_are_present=True, satspots_detector=satellite_detector,
            satspots_exptime=satellite['exptime'], satspots_number_of_frames=satellite['count'],
            satspot_keywords=dict(custom_pattern=pattern, scale=scale, pattern_name=probe_label(probe_path)),
            visit_id=visitid, vistype=p['vistype'], save_as_fits=True, output_dir=outdir,
            full_frame=FULL_FRAME, loc_x=LOC_X, loc_y=LOC_Y, frame_times_utc=frame_times,
            retain_images=False, overwrite=overwrite,
        )
        effective = datetime.fromisoformat(frame_times[0])
        counts = [satellite['count']] * 3 + ([science['count']] if science else [])
        print(f'{label} VISITID={visitid} counts={counts} '
              f'start={effective.isoformat()} UTC, requested={p["requested_start"]}, '
              f'shift={(effective-requested).total_seconds():g}s')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=float, default=SCALE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true', default=OVERWRITE)
    run_campaign(**vars(parser.parse_args()))
