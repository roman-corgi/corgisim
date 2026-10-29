#!/usr/bin/env python
"""Generate flight-like Alternate Probe L1 campaigns.

For each custom DM probe, the driver writes unprobed, positive, and negative
frames, then restores the base DM. ``scale`` is required; normal operation
uses ``1.0``. Intended amplitudes and orientation belong in the probe files;
the multiplier allows exceptional amplitude adjustments.

All three template states carry the group flag ``SATSPOTS=1``. Probed frames
also carry pattern/scale/sign COMMENT provenance; the unprobed background has
no sign. Subsequent target science frames carry ``SATSPOTS=0``.
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import proper
import roman_preflight_proper
from astropy.io import fits

from corgisim import instrument, observation, outputs, scene
from corgisim.sat_spots import add_custom_pattern_dm

# Defaults are overridable through the CLI or ``run_campaign``.

# --- Host star ---------------------------------------------------------
HOST_STAR_PROPERTIES = {'Vmag': 5, 'spectral_type': 'G0V', 'magtype': 'vegamag'}

# --- Coronagraph / optics -----------------------------------------------
CGI_MODE = 'excam'
COR_TYPE = 'hlc_band1'
# Standalone default is the central subband; use ``1F`` for full Band 1.
BANDPASS = '1B'
DM_ROOTNAME = 'hlc_ni_3e-8'  # roman_preflight_proper HLC Band-1 example dark-hole solution
# Explicit paths override the example DM solution.
DM1_PATH = None
DM2_PATH = None
POLAXIS = 10
OUTPUT_DIM = 51 * 5  # 255x255: standard-observation stamp dimension chosen for this standalone simulation

# --- Detector placement on the 1024x1024 clean frame --------------------
LOC_X = 512
LOC_Y = 512

# --- Detector / exposure -------------------------------------------------
EXPTIME_S = 10.0
EM_GAIN = 1000.0
PHOTON_COUNTING = False
N_FRAMES_PER_STATE = 3  # unprobed / +probe / -probe each get this many frames

# --- Visit metadata --------------------------------------------------------
# Confirmed VISTYPE for the Alternate Probe satellite-spot observation.
VISTYPE = 'CGIVST_TDD_OBS'

# VISITID fields; OBSNUM increments once per probe.
VISIT_PROGNUM = '0200'
VISIT_EXECNUM = '001'
VISIT_CAMPAIGN = '001'
VISIT_SEGMENT = '001'
VISIT_OBSNUM_START = 1
VISIT_VISNUM = '001'

# --- Timestamps ------------------------------------------------------------
# Spacing must clear the exposure time and whole-second SCTSRT precision.
CAMPAIGN_START_TIME_UTC = '2027-06-01T00:00:00'
FRAME_TIME_STEP_S = 30.0
MIN_FRAME_TIME_STEP_S = 1.0

# --- Output ------------------------------------------------------------
# ``None`` writes under ``<cwd>/alt_probe_l1_output``.
DEFAULT_OUTPUT_DIR = None


def build_visitid(prognum, execnum, campaign, segment, obsnum, visnum):
    """Assemble a validated 19-digit VISITID."""
    parts = (prognum, execnum, campaign, segment, f'{obsnum:03d}', visnum)
    if any(not isinstance(p, str) or len(p) != n or not p.isascii() or not p.isdigit()
           for p, n in zip(parts, (4, 3, 3, 3, 3, 3))):
        raise ValueError('VISITID components require 4/3/3/3/3/3 digits')
    return ''.join(parts)


def load_base_dm(dm_rootname=DM_ROOTNAME, dm1_path=None, dm2_path=None):
    """Load the base DM1/DM2 voltage maps from explicit or example paths."""
    if dm1_path is None:
        dm1_path = roman_preflight_proper.lib_dir + '/examples/' + dm_rootname + '_dm1_v.fits'
    if dm2_path is None:
        dm2_path = roman_preflight_proper.lib_dir + '/examples/' + dm_rootname + '_dm2_v.fits'
    dm1 = proper.prop_fits_read(dm1_path)
    dm2 = proper.prop_fits_read(dm2_path)
    return dm1, dm2


def load_probe_pattern(path):
    """Read a relative-DM probe FITS array in volts."""
    return fits.getdata(path)


def probe_label(path):
    """Return the probe filename stem for provenance."""
    return os.path.splitext(os.path.basename(path))[0]


def planned_l1_filename(visitid, ftimeutc):
    """Return the L1 filename derived by the writer."""
    return f"cgi_{visitid}_{outputs.isotime_to_yyyymmddThhmmsss(ftimeutc)}_l1_.fits"


def run_campaign(
    *,
    scale,
    probe_files,
    n_frames_per_state=N_FRAMES_PER_STATE,
    exptime=EXPTIME_S,
    em_gain=EM_GAIN,
    photon_counting=PHOTON_COUNTING,
    output_dim=OUTPUT_DIM,
    loc_x=LOC_X,
    loc_y=LOC_Y,
    bandpass=BANDPASS,
    cor_type=COR_TYPE,
    cgi_mode=CGI_MODE,
    dm_rootname=DM_ROOTNAME,
    dm1_path=DM1_PATH,
    dm2_path=DM2_PATH,
    vistype=VISTYPE,
    visit_prognum=VISIT_PROGNUM,
    visit_execnum=VISIT_EXECNUM,
    visit_campaign=VISIT_CAMPAIGN,
    visit_segment=VISIT_SEGMENT,
    visit_obsnum_start=VISIT_OBSNUM_START,
    visit_visnum=VISIT_VISNUM,
    frame_time_step_s=FRAME_TIME_STEP_S,
    campaign_start_time_utc=CAMPAIGN_START_TIME_UTC,
    output_dir=DEFAULT_OUTPUT_DIR,
    overwrite=False,
):
    """Generate an unprobed/positive/negative L1 sequence for each probe.

    Inputs and output paths are validated before any files are written.
    ``frame_time_step_s`` must be at least ``max(exptime, 1.0)``.

    Parameters
    ----------
    scale : numbers.Real
        Required probe amplitude scale; normally 1.0.
    probe_files : iterable of path-like
        Relative-DM probe FITS files, one sequence per file.
    n_frames_per_state : int, optional
        Frames acquired in each DM state.
    exptime : float, optional
        Exposure time in seconds.
    em_gain : float, optional
        EM gain passed to CorgiDetector.
    photon_counting : bool, optional
        Enable photon-counting mode.
    output_dim : int, optional
        Simulated stamp size in pixels.
    loc_x, loc_y : int, optional
        Stamp center in the 1024x1024 science area.
    bandpass : str, optional
        CGI bandpass name.
    cor_type, cgi_mode : str, optional
        CorgiOptics configuration.
    dm_rootname : str, optional
        Example DM solution rootname.
    dm1_path, dm2_path : str or None, optional
        Explicit base DM paths, overriding ``dm_rootname``.
    vistype : str, optional
        Primary-header VISTYPE.
    visit_prognum, visit_execnum, visit_campaign, visit_segment, visit_visnum : str, optional
        Fixed VISITID fields.
    visit_obsnum_start : int, optional
        First observation number; increments per probe.
    frame_time_step_s : float, optional
        Seconds between frame starts.
    campaign_start_time_utc : str, optional
        ISO UTC start time. Timezone-naive values are taken as UTC; aware
        values (e.g. ``Z`` or ``+00:00``) are converted to UTC.
    output_dir : str, pathlib.Path or None, optional
        Output root; defaults to ``<cwd>/alt_probe_l1_output``.
    overwrite : bool, optional
        Replace existing products. Default is False.

    Returns
    -------
    str or pathlib.Path
        Output root containing per-visit directories.

    Raises
    ------
    ValueError, TypeError
        If campaign or probe inputs are invalid.
    FileExistsError
        If a product exists and ``overwrite`` is False.
    """
    # Validate before creating files or copying PROPER prescriptions.
    if not isinstance(frame_time_step_s, (int, float)) or isinstance(frame_time_step_s, bool) \
            or not math.isfinite(frame_time_step_s) or frame_time_step_s <= 0:
        raise ValueError(
            f"frame_time_step_s must be a finite positive number of seconds, got {frame_time_step_s!r}")

    if not isinstance(n_frames_per_state, int) or isinstance(n_frames_per_state, bool) \
            or n_frames_per_state <= 0:
        raise ValueError(
            f"n_frames_per_state must be a positive integer, got {n_frames_per_state!r}")

    if not isinstance(exptime, (int, float)) or isinstance(exptime, bool) \
            or not math.isfinite(exptime) or exptime <= 0:
        raise ValueError(f"exptime must be a finite positive number of seconds, got {exptime!r}")

    # Prevent overlapping exposures and duplicate whole-second SCTSRT values.
    min_step = max(exptime, MIN_FRAME_TIME_STEP_S)
    if frame_time_step_s < min_step:
        raise ValueError(
            f"frame_time_step_s ({frame_time_step_s} s) must be >= "
            f"max(exptime, {MIN_FRAME_TIME_STEP_S}) = {min_step} s, so that sequential "
            "detector exposures do not overlap and SCTSRT (stored by outputs.py only to "
            "whole-second precision) remains strictly increasing across the campaign"
        )

    probe_files = list(probe_files)
    if not probe_files:
        raise ValueError("probe_files must contain at least one probe FITS path")

    dm1, dm2 = load_base_dm(dm_rootname, dm1_path=dm1_path, dm2_path=dm2_path)

    # Validate every probe before writing a partial campaign.
    probes = []
    for probe_path in probe_files:
        pattern = load_probe_pattern(probe_path)
        add_custom_pattern_dm(dm1, pattern, scale)
        probes.append((probe_path, pattern))

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'alt_probe_l1_output')

    # Plan every visit and timestamp before writing.
    campaign_time = datetime.fromisoformat(campaign_start_time_utc)
    # The L1 writer expects timezone-naive UTC strings.
    if campaign_time.tzinfo is not None:
        campaign_time = campaign_time.astimezone(timezone.utc).replace(tzinfo=None)
    plan = []
    for probe_index, (probe_path, pattern) in enumerate(probes):
        visitid = build_visitid(
            visit_prognum, visit_execnum, visit_campaign, visit_segment,
            visit_obsnum_start + probe_index, visit_visnum,
        )
        visit_outdir = os.path.join(output_dir, f'V{visitid}')
        frame_times = []
        for _ in range(3 * n_frames_per_state):
            frame_times.append(campaign_time.isoformat())
            campaign_time += timedelta(seconds=frame_time_step_s)
        plan.append((probe_path, pattern, visitid, visit_outdir, frame_times))

    planned_paths = [
        os.path.join(visit_outdir, planned_l1_filename(visitid, ftimeutc))
        for _, _, visitid, visit_outdir, frame_times in plan
        for ftimeutc in frame_times
    ]

    for planned_path in planned_paths:
        path = Path(planned_path)
        if path.is_dir() or path.is_symlink():
            raise ValueError(f'output path is a directory or symlink: {path}')
        if path.exists() and not overwrite:
            raise FileExistsError(
                f'output already exists and overwrite=False: {path}'
            )
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise NotADirectoryError(f'output ancestor is not a directory: {parent}')

    # copy_here() has filesystem side effects, so call it after validation.
    roman_preflight_proper.copy_here()
    # PROPER workers import the prescriptions copied into the working directory.
    prescription_dir = os.getcwd()
    if prescription_dir not in sys.path:
        sys.path.insert(0, prescription_dir)

    os.makedirs(output_dir, exist_ok=True)

    optics_keywords = {
        'cor_type': cor_type, 'use_errors': 1, 'polaxis': POLAXIS, 'output_dim': output_dim,
        'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
        'use_fpm': 1, 'use_lyot_stop': 1, 'use_field_stop': 1,
    }

    base_scene = scene.Scene(HOST_STAR_PROPERTIES)
    # Reuse one optics/detector pair; update visit_id before each probe.
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords,
                                    if_quiet=True, visit_type=vistype)
    detector = instrument.CorgiDetector({'em_gain': em_gain}, photon_counting=photon_counting)

    for probe_index, (probe_path, pattern, visitid, visit_outdir, frame_times) in enumerate(plan):
        pattern_name = probe_label(probe_path)
        os.makedirs(visit_outdir, exist_ok=True)

        print(f"[{probe_index + 1}/{len(plan)}] probe={pattern_name} VISITID={visitid} scale={scale}")

        observation.generate_observation_sequence(
            base_scene, optics, detector, exptime, n_frames=0,
            satspots_are_present=True, satspots_detector=detector,
            satspots_exptime=exptime, satspots_number_of_frames=n_frames_per_state,
            satspot_keywords={'custom_pattern': pattern, 'scale': scale, 'pattern_name': pattern_name},
            visit_id=visitid, vistype=vistype, save_as_fits=True,
            output_dir=visit_outdir, full_frame=True, loc_x=loc_x, loc_y=loc_y,
            frame_times_utc=frame_times,
            retain_images=False, overwrite=overwrite,
        )

    return output_dir


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Alternate Probe L1 campaign driver: generates unprobed/+probe/-probe "
            "L1 FITS trios for one or more custom DM probe files, via the "
            "satellite-spot custom_pattern path."
        ),
    )
    parser.add_argument('--scale', type=float, required=True,
                         help="Required probe amplitude multiplier; normally 1.0. "
                              "Use other values only for exceptional amplitude adjustments.")
    parser.add_argument('--probe-files', nargs='+', required=True,
                         help="Paths to 48x48 dmrel probe FITS files, one campaign trio per file. "
                              "REQUIRED -- no default (no user-specific paths are embedded here).")
    parser.add_argument('--n-frames-per-state', type=int, default=N_FRAMES_PER_STATE)
    parser.add_argument('--exptime', type=float, default=EXPTIME_S)
    parser.add_argument('--em-gain', type=float, default=EM_GAIN)
    parser.add_argument('--photon-counting', action='store_true', default=PHOTON_COUNTING)
    parser.add_argument('--output-dim', type=int, default=OUTPUT_DIM)
    parser.add_argument('--loc-x', type=int, default=LOC_X)
    parser.add_argument('--loc-y', type=int, default=LOC_Y)
    parser.add_argument('--bandpass', default=BANDPASS,
                         help="Standalone default is '1B' (Band 1 central subband); use '1F' for full Band 1.")
    parser.add_argument('--cor-type', default=COR_TYPE)
    parser.add_argument('--cgi-mode', default=CGI_MODE)
    parser.add_argument('--dm-rootname', default=DM_ROOTNAME,
                         help="roman_preflight_proper example rootname; ignored if --dm1-path/--dm2-path given.")
    parser.add_argument('--dm1-path', default=DM1_PATH, help="Arbitrary DM1 FITS path, overrides --dm-rootname.")
    parser.add_argument('--dm2-path', default=DM2_PATH, help="Arbitrary DM2 FITS path, overrides --dm-rootname.")
    parser.add_argument('--vistype', default=VISTYPE,
                         help="VISTYPE written to the L1 primary header; defaults to CGIVST_TDD_OBS.")
    parser.add_argument('--visit-prognum', default=VISIT_PROGNUM)
    parser.add_argument('--visit-execnum', default=VISIT_EXECNUM)
    parser.add_argument('--visit-campaign', default=VISIT_CAMPAIGN)
    parser.add_argument('--visit-segment', default=VISIT_SEGMENT)
    parser.add_argument('--visit-obsnum-start', type=int, default=VISIT_OBSNUM_START)
    parser.add_argument('--visit-visnum', default=VISIT_VISNUM)
    parser.add_argument('--campaign-start-time-utc', default=CAMPAIGN_START_TIME_UTC)
    parser.add_argument('--frame-time-step-s', type=float, default=FRAME_TIME_STEP_S,
                        help="Spacing between consecutive frame start times. Must be >= "
                             f"max(--exptime, {MIN_FRAME_TIME_STEP_S}) so exposures do not "
                             "overlap and whole-second SCTSRT stays strictly increasing.")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true', default=False,
                         help="Replace existing L1 output files. Default is to refuse and abort.")
    return parser.parse_args()


def main():
    """Command-line entry point: parse arguments and run one campaign."""
    args = parse_args()
    run_campaign(
        scale=args.scale,
        probe_files=args.probe_files,
        n_frames_per_state=args.n_frames_per_state,
        exptime=args.exptime,
        em_gain=args.em_gain,
        photon_counting=args.photon_counting,
        output_dim=args.output_dim,
        loc_x=args.loc_x,
        loc_y=args.loc_y,
        bandpass=args.bandpass,
        cor_type=args.cor_type,
        cgi_mode=args.cgi_mode,
        dm_rootname=args.dm_rootname,
        dm1_path=args.dm1_path,
        dm2_path=args.dm2_path,
        vistype=args.vistype,
        visit_prognum=args.visit_prognum,
        visit_execnum=args.visit_execnum,
        visit_campaign=args.visit_campaign,
        visit_segment=args.visit_segment,
        visit_obsnum_start=args.visit_obsnum_start,
        visit_visnum=args.visit_visnum,
        frame_time_step_s=args.frame_time_step_s,
        campaign_start_time_utc=args.campaign_start_time_utc,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
