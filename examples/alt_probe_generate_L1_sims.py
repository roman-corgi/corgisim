#!/usr/bin/env python
"""Alternate Probe L1 campaign driver.

This is a direct, scripted repetition of the existing satellite-spot
observing template demonstrated in ``examples/Satellite_spots_demo.ipynb``:

    unprobed -> add_satspot(+probe) -> remove_satspot -> add_satspot(-probe)
    -> remove_satspot (restore)

repeated once per configured probe file, using the custom-pattern satellite
spot path added to ``corgisim.sat_spots.add_custom_pattern_dm`` /
``instrument.CorgiOptics.add_satspot``/``remove_satspot``. It intentionally
contains NO new sequencing classes/framework: a single ``CorgiOptics`` object
and a single ``CorgiDetector`` object are built once and reused for the whole
campaign, exactly as in the demo notebook.

Probe scale provenance (see also corgisim.sat_spots.add_custom_pattern_dm)
------------------------------------------------------------------------
``scale`` has NO default anywhere in this script -- it must be supplied
explicitly (``--scale`` on the command line, or as a keyword argument to
``run_campaign``). Two conflicting conventions exist, and they differ in
applied probe intensity by roughly a factor of 11, so silently choosing one
would materially change the simulated data. The correct value for a given
probe delivery must be established by measuring the resulting probe normalized
intensity against the probe's design target:

  * ``scale = 1.0``: applies the delivered relative-DM array unmodified.
  * ``scale = 0.3``: the legacy convention from
    ``corgihowfsc/corgihowfsc/sensing/GettingProbes.py::get_dm_probes``
    (``dm1 = dm10 + scale * dmrel``, default ``scale = +/-0.3``). That 0.3
    was a generic default of the HOWFSC probing helper and may not apply to
    this delivery.

SATSPOTS semantics / Phase 2 note
------------------------------------------------------------------------
This driver leaves the SATSPOTS flag exactly as the existing corgisim
machinery produces it: unprobed frames get SATSPOTS=0, both +probe and
-probe frames get SATSPOTS=1. No re-tagging is performed here.

NOTE (Phase 2 concern, not addressed by this driver): corgidrp's
``l3_to_l4.find_star`` (with its default ``subtract_no_offset_frames=True``)
expects satellite-spot data in **three equal-sized groups that are ALL
tagged SATSPOTS=1** (no-offset / +offset / -offset), ordered by SCTSRT. That
convention differs from the SATSPOTS=0/1 split produced by this driver
(unprobed=0, probed=1) and will need to be reconciled -- either by a
corgidrp recipe change or a different find_star call -- when the downstream
processing recipe is developed. Do not attempt to fix this here.

Probe provenance in the written L1 headers (known asymmetry)
------------------------------------------------------------------------
The existing corgisim machinery records probe provenance only while a
pattern is actually applied to the DM: ``CorgiOptics.add_satspot`` populates
``satspot_info``, which becomes the ``satspot_pattern_name`` /
``satspot_scale`` / ``satspot_sign`` COMMENT cards in the primary header, and
``remove_satspot`` clears it. Consequently, in the products written here:

  * positive- and negative-probe frames carry the probe filename stem, the
    applied scale, and the sign as COMMENTs (plus ``SATSPOTS=1``);
  * unprobed frames carry ``SATSPOTS=0`` and **no** probe pattern/scale/sign
    COMMENTs at all.

Unprobed frames are therefore associated with their probe execution only
indirectly: every frame of one probe's trio shares that probe's VISITID (each
probe execution gets a distinct VISITID), and within a VISITID the acquisition
order is recoverable from the strictly ascending FTIMEUTC/SCTSRT values
(unprobed frames first, then positive, then negative). That is sufficient to
reconstruct the trio downstream. Attaching explicit probe provenance to
unprobed frames as well would require changing ``instrument.py`` (so that
``satspot_info`` can describe a pending/inactive probe), which is deliberately
out of scope for this driver; it is documented here rather than worked around.
"""

import argparse
import math
import os
from collections import Counter
from datetime import datetime, timedelta

import proper
import roman_preflight_proper
from astropy.io import fits

from corgisim import instrument, outputs, scene
from corgisim.sat_spots import add_custom_pattern_dm

# ============================================================================
# Configuration defaults (all overridable via CLI flags / run_campaign kwargs)
# ============================================================================

# --- Host star ---------------------------------------------------------
HOST_STAR_PROPERTIES = {'Vmag': 5, 'spectral_type': 'G0V', 'magtype': 'vegamag'}

# --- Coronagraph / optics -----------------------------------------------
CGI_MODE = 'excam'
COR_TYPE = 'hlc_band1'
# '1B' = Band 1 central subband; '1F' = full Band 1 imaging filter is the
# alternative.
# UNCONFIRMED -- the correct choice for this observing template should be
# confirmed before production runs.
BANDPASS = '1B'
DM_ROOTNAME = 'hlc_ni_3e-8'  # roman_preflight_proper HLC Band-1 example dark-hole solution
# DM1_PATH / DM2_PATH: set to arbitrary FITS file paths to override
# DM_ROOTNAME entirely (see load_base_dm); left as None to use the
# roman_preflight_proper example above by default.
DM1_PATH = None
DM2_PATH = None
POLAXIS = 10
OUTPUT_DIM = 153  # matches the 153x153 GITL crop geometry used by corgihowfsc

# --- Detector placement on the 1024x1024 clean frame --------------------
LOC_X = 512
LOC_Y = 512

# --- Detector / exposure -------------------------------------------------
EXPTIME_S = 10.0
EM_GAIN = 1000.0
PHOTON_COUNTING = False
N_FRAMES_PER_STATE = 3  # unprobed / +probe / -probe each get this many frames

# --- Visit metadata --------------------------------------------------------
# VISTYPE for the satellite-spot imaging template is UNCONFIRMED (corgisim's
# tutorial1 uses 'CGIVST_TDD_OBS'; other campaign scripts use a CAL VISTYPE)
# -- kept configurable, and should be confirmed before production runs.
VISTYPE = 'CGIVST_TDD_OBS'

# VISITID scheme: 19-digit string PROGNUM(4)+EXECNUM(3)+CAMPAIGN(3)+
# SEGMENT(3)+OBSNUM(3)+VISNUM(3), matching outputs.save_hdu_to_fits's
# slicing convention. OBSNUM is incremented once per probe execution so that
# each probe file gets a distinct VISITID; the other fields are fixed
# placeholders, configurable below.
VISIT_PROGNUM = '0200'
VISIT_EXECNUM = '001'
VISIT_CAMPAIGN = '001'
VISIT_SEGMENT = '001'
VISIT_OBSNUM_START = 1
VISIT_VISNUM = '001'

# --- Probe files -----------------------------------------------------------
# No default here on purpose: this driver must not embed any user-specific
# absolute paths. `probe_files` is REQUIRED (--probe-files on the CLI); pass
# a list of 48x48 dmrel probe FITS files, one campaign trio per file (e.g.
# the delivered Gaussian Alternate Probe arrays, which are kept outside this
# repository).

# --- Timestamps ------------------------------------------------------------
# Arbitrary fixed campaign start time; FTIMEUTC increases strictly by
# FRAME_TIME_STEP_S for every frame, in acquisition order, across the whole
# campaign (all states of probe 1, then all states of probe 2, ...), so that
# time-sorting downstream reconstructs the acquisition sequence.
#
# run_campaign REQUIRES frame_time_step_s >= max(exptime, 1.0), for two
# independent reasons:
#   * >= exptime: consecutive frames are separate detector exposures, so
#     their start times must be separated by at least one exposure time or
#     the simulated acquisition would describe overlapping integrations.
#   * >= 1.0 s: outputs.py derives SCTSRT/SCTEND from FTIMEUTC but stores
#     them only to whole-second precision, so sub-second frame spacing can
#     collapse to a non-increasing SCTSRT sequence and destroy the
#     acquisition ordering that downstream time-sorting depends on.
#
# That constraint also implies frames are at least 1 s apart, which is
# comfortably above the 0.1 s rounding that
# outputs.isotime_to_yyyymmddThhmmsss applies to FTIMEUTC when deriving the
# L1 filename. run_campaign nevertheless still builds the complete planned
# VISITID/FTIMEUTC sequence up front and verifies with that same formatter
# that every output path is unique, as a defensive check that cannot drift
# from the writer if the filename convention or this constraint changes.
CAMPAIGN_START_TIME_UTC = '2027-06-01T00:00:00'
FRAME_TIME_STEP_S = 30.0

# Absolute floor on frame spacing, imposed by whole-second SCTSRT storage in
# outputs.py (see the note above).
MIN_FRAME_TIME_STEP_S = 1.0

# --- Output ------------------------------------------------------------
# Default output directory mirrors outputs.save_hdu_to_fits's own
# outdir=None convention (current working directory) so nothing is written
# into the tracked examples/ source directory unless the user opts in.
DEFAULT_OUTPUT_DIR = None


def build_visitid(prognum, execnum, campaign, segment, obsnum, visnum):
    """Assemble and validate a 19-digit VISITID string for one probe execution.

    Parameters
    ----------
    prognum, execnum, campaign, segment, visnum : str
        Fixed VISITID fields.
    obsnum : int
        Observation number, zero-padded to three digits; incremented once per
        probe so each template execution gets its own VISITID.

    Returns
    -------
    str
        The concatenated 19-digit VISITID.

    Raises
    ------
    ValueError
        If the concatenated fields are not exactly 19 numeric digits.
    """
    visitid = f"{prognum}{execnum}{campaign}{segment}{obsnum:03d}{visnum}"
    if len(visitid) != 19 or not visitid.isdigit():
        raise ValueError(f"Constructed VISITID '{visitid}' is not a 19-digit numeric string")
    return visitid


def load_base_dm(dm_rootname=DM_ROOTNAME, dm1_path=None, dm2_path=None):
    """Load the DM1/DM2 base (unprobed) solution.

    Parameters
    ----------
    dm_rootname : str, optional
        Rootname of a roman_preflight_proper example DM solution.
    dm1_path, dm2_path : str or None, optional
        Explicit DM FITS paths; when given, they override ``dm_rootname``.

    Returns
    -------
    tuple of numpy.ndarray
        The (dm1, dm2) 48x48 voltage maps.
    """
    if dm1_path is None:
        dm1_path = roman_preflight_proper.lib_dir + '/examples/' + dm_rootname + '_dm1_v.fits'
    if dm2_path is None:
        dm2_path = roman_preflight_proper.lib_dir + '/examples/' + dm_rootname + '_dm2_v.fits'
    dm1 = proper.prop_fits_read(dm1_path)
    dm2 = proper.prop_fits_read(dm2_path)
    return dm1, dm2


def load_probe_pattern(path):
    """Read a 48x48 relative-DM probe FITS array (volts).

    Validation is left to sat_spots.add_custom_pattern_dm, which run_campaign
    calls on every probe before any output is written.
    """
    return fits.getdata(path)


def probe_label(path):
    """Return a short probe label (the filename stem) for ``pattern_name``
    provenance, so the value fits intact in a FITS COMMENT card.

    The stem is not inherently unique: probe files sharing a basename in
    different directories produce the same label. Probe executions are
    distinguished in the products by their distinct VISITIDs.
    """
    return os.path.splitext(os.path.basename(path))[0]


def planned_l1_filename(visitid, ftimeutc):
    """Return the L1 filename outputs.save_hdu_to_fits will derive for this frame.

    Reuses that writer's timestamp formatter so the prediction cannot drift
    from the writer.
    """
    return f"cgi_{visitid}_{outputs.isotime_to_yyyymmddThhmmsss(ftimeutc)}_l1_.fits"


def save_frame(sim_scene, detector, exptime, loc_x, loc_y, outdir, visitid, vistype, ftimeutc,
               overwrite=False):
    """Propagate one exposure onto the detector as a full-frame L1 product and save it.

    Parameters
    ----------
    sim_scene : corgisim.scene.SimulatedScene
        Noise-free scene for the current DM state.
    detector : corgisim.instrument.CorgiDetector
        Detector model used to generate the full-frame image.
    exptime : float
        Exposure time in seconds.
    loc_x, loc_y : int
        Column and row of the stamp centre within the 1024x1024 science area.
    outdir : str
        Directory the L1 file is written to.
    visitid, vistype : str
        Primary-header values, overridden per frame.
    ftimeutc : str
        Frame start time (ISO UTC); also determines the output filename.
    overwrite : bool, optional
        If True, replace an existing output file. Default False, so an
        existing file is never silently clobbered.

    Returns
    -------
    None
    """
    sim_scene = detector.generate_detector_image(sim_scene, exptime, full_frame=True, loc_x=loc_x, loc_y=loc_y)
    outputs.save_hdu_to_fits(
        sim_scene.image_on_detector,
        outdir=outdir,
        write_as_L1=True,
        overwrite=overwrite,
        overwrite_pri_keywords={'VISITID': visitid, 'VISTYPE': vistype},
        overwrite_ext_keywords={'FTIMEUTC': ftimeutc},
    )


def run_campaign(
    *,
    scale,  # REQUIRED -- no default; see the scale-provenance note in the module docstring.
    probe_files,  # REQUIRED -- no default, so no user-specific paths are embedded here.
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
    """Generate flight-like Alternate Probe L1 datasets for a list of probe files.

    For each probe file, runs one unprobed/+probe/-probe satellite-spot
    template trio (direct scripted repetition of
    examples/Satellite_spots_demo.ipynb) on a single, persistent CorgiOptics
    object shared across the whole campaign; only add_satspot/remove_satspot
    mutate its DM state between states/probes. Returns the output directory
    used.

    ``overwrite=False`` (the default) refuses to clobber existing output
    files; pass ``overwrite=True`` to replace them.

    ``frame_time_step_s`` must be at least ``max(exptime, 1.0)`` seconds so
    that consecutive exposures do not overlap and the whole-second SCTSRT
    values written by outputs.py stay strictly increasing.

    All configuration is validated -- and every probe loaded, every VISITID
    built and every output path planned -- before anything is written and
    before PROPER's prescription files are copied into the working
    directory, so an invalid configuration has no filesystem side effects.

    Parameters
    ----------
    scale : numbers.Real
        Probe amplitude scale factor. REQUIRED, with no default, because the
        convention for the delivered probe arrays is unresolved (see the
        module docstring).
    probe_files : list of str
        Paths to 48x48 relative-DM probe FITS files. REQUIRED, with no
        default, so no user-specific paths are embedded here. One
        satellite-spot template trio is executed per file.
    n_frames_per_state : int, optional
        Number of detector frames acquired in each of the three DM states.
    exptime : float, optional
        Exposure time in seconds.
    em_gain : float, optional
        EM gain passed to CorgiDetector.
    photon_counting : bool, optional
        Whether the detector model runs in photon-counting mode.
    output_dim : int, optional
        Size of the simulated stamp in pixels.
    loc_x, loc_y : int, optional
        Column and row of the stamp centre within the 1024x1024 science area.
    bandpass : str, optional
        CGI bandpass name (UNCONFIRMED default; see the module docstring).
    cor_type, cgi_mode : str, optional
        Coronagraph configuration and observing mode passed to CorgiOptics.
    dm_rootname : str, optional
        roman_preflight_proper example DM rootname; ignored when
        ``dm1_path``/``dm2_path`` are given.
    dm1_path, dm2_path : str or None, optional
        Explicit base DM1/DM2 FITS paths, overriding ``dm_rootname``.
    vistype : str, optional
        VISTYPE written to the primary header (UNCONFIRMED default).
    visit_prognum, visit_execnum, visit_campaign, visit_segment, visit_visnum : str, optional
        Fixed VISITID fields shared by every probe execution.
    visit_obsnum_start : int, optional
        Observation number of the first probe; incremented by one per probe so
        each execution gets a distinct VISITID.
    frame_time_step_s : float, optional
        Spacing between consecutive frame start times, in seconds. Must be at
        least ``max(exptime, MIN_FRAME_TIME_STEP_S)``.
    campaign_start_time_utc : str, optional
        ISO UTC start time of the first frame of the campaign.
    output_dir : str, pathlib.Path or None, optional
        Directory the per-visit subdirectories are written into. If None,
        ``<cwd>/alt_probe_l1_output`` is used.
    overwrite : bool, optional
        If True, replace existing L1 output files. Default is False, which
        aborts before writing if any planned output already exists.

    Returns
    -------
    str or pathlib.Path
        The output directory containing the per-visit subdirectories: the
        ``output_dir`` value as passed in, or the default path when None.

    Raises
    ------
    ValueError
        If any campaign input is invalid (non-finite or non-positive timing or
        exposure values, a non-positive frame count, an empty probe list, a
        frame spacing below ``max(exptime, MIN_FRAME_TIME_STEP_S)``, an
        invalid VISITID, or a collision between planned output filenames), or
        if a probe array fails the sat_spots.add_custom_pattern_dm checks.
    TypeError
        If ``scale`` is not a real scalar number
        (from sat_spots.add_custom_pattern_dm).
    FileExistsError
        If a planned output file already exists and ``overwrite`` is False.
    """
    # ---- Validate campaign inputs before doing any work -------------------
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

    # Frames are separate detector exposures written with whole-second SCTSRT
    # precision (see the FRAME_TIME_STEP_S note above), so the step must clear
    # both the exposure time and the 1 s SCTSRT resolution.
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

    # load_base_dm only reads FITS arrays out of roman_preflight_proper.lib_dir,
    # so it needs no PROPER prescription in the working directory; copy_here()
    # is deferred until all validation has passed (see below).
    dm1, dm2 = load_base_dm(dm_rootname, dm1_path=dm1_path, dm2_path=dm2_path)

    # Load and validate every probe file up front, before any output product
    # is created, so that a malformed probe later in the list cannot leave
    # partial campaign output on disk. add_custom_pattern_dm performs the
    # actual shape/finiteness/complex/scale checks; it is reused here purely
    # for validation and its result is discarded.
    probes = []
    for probe_path in probe_files:
        pattern = load_probe_pattern(probe_path)
        add_custom_pattern_dm(dm1, pattern, scale)
        probes.append((probe_path, pattern))

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'alt_probe_l1_output')

    # ---- Build the complete planned acquisition sequence ------------------
    # Every VISITID is constructed (and thereby validated) here, and every
    # FTIMEUTC is assigned here, so the full set of output paths is known
    # before a single file is written.
    campaign_time = datetime.fromisoformat(campaign_start_time_utc)
    plan = []
    for probe_index, (probe_path, pattern) in enumerate(probes):
        visitid = build_visitid(
            visit_prognum, visit_execnum, visit_campaign, visit_segment,
            visit_obsnum_start + probe_index, visit_visnum,
        )
        visit_outdir = os.path.join(output_dir, f'V{visitid}')
        states = []
        for state in ('unprobed', 'positive', 'negative'):
            stamps = []
            for _ in range(n_frames_per_state):
                stamps.append(campaign_time.isoformat())
                campaign_time += timedelta(seconds=frame_time_step_s)
            states.append((state, stamps))
        plan.append((probe_path, pattern, visitid, visit_outdir, states))

    # ---- Verify the planned output paths are unique -----------------------
    # The L1 filename is derived from VISITID plus FTIMEUTC rounded to 0.1 s,
    # so distinct timestamps do not guarantee distinct filenames.
    planned_paths = [
        os.path.join(visit_outdir, planned_l1_filename(visitid, ftimeutc))
        for _, _, visitid, visit_outdir, states in plan
        for _, stamps in states
        for ftimeutc in stamps
    ]
    collisions = sorted(path for path, count in Counter(planned_paths).items() if count > 1)
    if collisions:
        raise ValueError(
            f"{len(collisions)} planned L1 output path(s) would collide because the "
            "filename timestamp is rounded to 0.1 s; increase frame_time_step_s or "
            "use distinct VISITIDs. First collision: " + collisions[0]
        )

    if not overwrite:
        existing = [path for path in planned_paths if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                f"{len(existing)} planned L1 output file(s) already exist and overwrite=False. "
                "First: " + existing[0]
            )

    # ---- Execute the campaign ---------------------------------------------
    # roman_preflight_proper.copy_here() has filesystem side effects (it copies
    # PROPER prescription files into the current working directory), so it is
    # deferred until every probe, VISITID, timestamp, collision and
    # existing-file check above has passed: an invalid configuration must
    # leave no trace on the filesystem.
    roman_preflight_proper.copy_here()

    os.makedirs(output_dir, exist_ok=True)

    optics_keywords = {
        'cor_type': cor_type, 'use_errors': 1, 'polaxis': POLAXIS, 'output_dim': output_dim,
        'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
        'use_fpm': 1, 'use_lyot_stop': 1, 'use_field_stop': 1,
    }

    base_scene = scene.Scene(HOST_STAR_PROPERTIES)
    # A single, persistent CorgiOptics/CorgiDetector pair is reused across all
    # probes and all trio states, exactly mirroring Satellite_spots_demo.ipynb.
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, if_quiet=True)
    detector = instrument.CorgiDetector({'em_gain': em_gain}, photon_counting=photon_counting)

    def capture(stamps, outdir, visitid):
        """Acquire and save the frames of the current DM state.

        ``stamps`` holds the frame start times (ISO UTC) for this state, in
        acquisition order.
        """
        # The noise-free scene is generated once per DM state and reused for
        # all frames in that state (matching
        # observation.generate_observation_sequence); only the detector noise
        # realization differs between frames.
        sim_scene = optics.get_host_star_psf(base_scene)
        for ftimeutc in stamps:
            save_frame(sim_scene, detector, exptime, loc_x, loc_y, outdir, visitid, vistype,
                       ftimeutc, overwrite=overwrite)

    for probe_index, (probe_path, pattern, visitid, visit_outdir, states) in enumerate(plan):
        pattern_name = probe_label(probe_path)
        stamps_by_state = dict(states)
        os.makedirs(visit_outdir, exist_ok=True)

        print(f"[{probe_index + 1}/{len(plan)}] probe={pattern_name} VISITID={visitid} scale={scale}")

        satspot_keywords = {'custom_pattern': pattern, 'scale': scale, 'pattern_name': pattern_name}

        # 1) Unprobed frames (SATSPOTS=0) -- DM is pristine base here, either
        #    because this is the first probe or because step 4 below restored
        #    it after the previous probe.
        capture(stamps_by_state['unprobed'], visit_outdir, visitid)

        # 2) Positive-probe frames (SATSPOTS=1).
        optics.add_satspot(satspot_keywords=satspot_keywords)
        capture(stamps_by_state['positive'], visit_outdir, visitid)

        # 3) Negative-probe frames (SATSPOTS=1): undo the positive pattern,
        #    then apply the negative (sign-flipped) pattern.
        optics.remove_satspot(satspot_keywords=satspot_keywords)
        satspot_keywords['sign'] = 'negative'
        optics.add_satspot(satspot_keywords=satspot_keywords)
        capture(stamps_by_state['negative'], visit_outdir, visitid)

        # 4) Restore the DM to the pristine base state before the next probe.
        optics.remove_satspot(satspot_keywords=satspot_keywords)

    return output_dir


def parse_args():
    """Build the command-line parser and parse ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, with one attribute per run_campaign keyword.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Alternate Probe L1 campaign driver: generates unprobed/+probe/-probe "
            "L1 FITS trios for one or more custom DM probe files, via the "
            "satellite-spot custom_pattern path."
        ),
    )
    parser.add_argument('--scale', type=float, required=True,
                         help="Probe amplitude scale factor. REQUIRED, no default -- "
                              "see module docstring for the 0.3-vs-1.0 provenance discussion.")
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
                         help="UNCONFIRMED default '1B' (Band 1 central subband); '1F' is the alternative.")
    parser.add_argument('--cor-type', default=COR_TYPE)
    parser.add_argument('--cgi-mode', default=CGI_MODE)
    parser.add_argument('--dm-rootname', default=DM_ROOTNAME,
                         help="roman_preflight_proper example rootname; ignored if --dm1-path/--dm2-path given.")
    parser.add_argument('--dm1-path', default=DM1_PATH, help="Arbitrary DM1 FITS path, overrides --dm-rootname.")
    parser.add_argument('--dm2-path', default=DM2_PATH, help="Arbitrary DM2 FITS path, overrides --dm-rootname.")
    parser.add_argument('--vistype', default=VISTYPE,
                         help="UNCONFIRMED for this template; see module docstring.")
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
