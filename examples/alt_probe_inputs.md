# Alternate and Standard Probe inputs

The generation script reads local XML and probe FITS files from
`roman-corgi/corgiobsspecs`. Download these four pinned snapshots in a browser
signed in to GitHub with access to that repository, then unzip them:

| Inputs | Source branch/folder | Pinned ZIP download |
|---|---|---|
| Alternate XML (12 observations) | [HLC_B1_C1_alternative_probes](https://github.com/roman-corgi/corgiobsspecs/tree/HLC_B1_C1_alternative_probes/obs_specs/HLC_B1_C1/alternative_probes) | [2c5f340846db5f00f4f8e634619d32725fe8249f](https://github.com/roman-corgi/corgiobsspecs/archive/2c5f340846db5f00f4f8e634619d32725fe8249f.zip) |
| Standard XML (11 observations) | [HLC_B1_C1_standard_probes](https://github.com/roman-corgi/corgiobsspecs/tree/HLC_B1_C1_standard_probes/obs_specs/HLC_B1_C1/standard_probes) | [aa4688822d32bc3c1e34184bca0df9de457cd423](https://github.com/roman-corgi/corgiobsspecs/archive/aa4688822d32bc3c1e34184bca0df9de457cd423.zip) |
| Alternate probes (3 FITS files) | [alternative_probes_DM_commands](https://github.com/roman-corgi/corgiobsspecs/tree/alternative_probes_DM_commands/obs_specs/HLC_B1_C1/alternative_probes/aux_files) | [c6a6a9386d8b3a01f48abd7bc808a16a0e2fe936](https://github.com/roman-corgi/corgiobsspecs/archive/c6a6a9386d8b3a01f48abd7bc808a16a0e2fe936.zip) |
| Standard probes (3 FITS files) | [standard_probes_DM_commands](https://github.com/roman-corgi/corgiobsspecs/tree/standard_probes_DM_commands/obs_specs/HLC_B1_C1/standard_probes/aux_files) | [b66e9153eae8ce82583cf0ffec268b15f29f68b2](https://github.com/roman-corgi/corgiobsspecs/archive/b66e9153eae8ce82583cf0ffec268b15f29f68b2.zip) |

Each ZIP selects the listed branch's exact commit without switching branches.
Cloning is useful when editing observation specifications; running this example
only requires the downloaded files.
Each archive expands into `corgiobsspecs-<commit>`; copy only the folders listed below.

## Combine the inputs

Keep the inputs outside the Corgisim checkout, preserving the original filenames:

1. From the alternate XML archive, copy `obs_specs/HLC_B1_C1/alternative_probes`
   into `/path/to/inputs/HLC_B1_C1`.
2. From the standard XML archive, copy `obs_specs/HLC_B1_C1/standard_probes`
   alongside it.
3. From each probe archive, copy its matching activity's `aux_files` folder into
   the activity folder. Add it without replacing the XML observation folders.

```text
/path/to/inputs/HLC_B1_C1/
├── alternative_probes/
│   ├── observation_a/alternative_probe_CFAM_1A.xml
│   ├── observation_b/alternative_probe_1A_probe1.xml
│   ├── ... (remaining observation folders through observation_l)
│   └── aux_files/
│       ├── nfov_dm_dmrel_5.0e-07_gaussian0.fits
│       ├── nfov_dm_dmrel_5.0e-07_gaussian1.fits
│       └── nfov_dm_dmrel_5.0e-07_gaussian2.fits
└── standard_probes/
    ├── observation_a/standard_probe_1C_probe1.xml
    ├── ... (remaining observation folders through observation_k)
    └── aux_files/
        ├── nfov_dm_dmrel_5.0e-07_standard0.fits
        ├── nfov_dm_dmrel_5.0e-07_standard1_unmod.fits
        └── nfov_dm_dmrel_5.0e-07_standard2.fits
```

## Configure and run

At the top of [alt_probe_generate_L1_sims.py](alt_probe_generate_L1_sims.py), configure
the default input/output roots, or override them with `--input-root`/`--output-dir`:

```python
INPUT_ROOT = Path('/path/to/inputs/HLC_B1_C1')
DEFAULT_OUTPUT_DIR = Path('/path/to/data/new_campaign')
```

`INPUT_ROOT` contains both activity folders. The script derives their XML and
`aux_files` paths; edit the derived probe roots only if using another layout.
`--input-root` relocates only paths beneath the configured `INPUT_ROOT`; explicit
XML or probe paths outside it remain unchanged. This preserves separately stored
probe arrays when overriding the campaign root.
Optional `DM1_PATH`/`DM2_PATH` override the installed `hlc_ni_3e-8` base maps.
Normal `SCALE = 1.0` applies the supplied probe arrays directly.

Only alternate A setup and B imaging are selected by default. `--full-campaign`
selects all 12 alternate and 11 standard rows in order, including five setup rows
and 18 imaging visits. It supersedes the editable `OBSERVATIONS` selection.
To omit setup A, set `OBSERVATIONS = ALTERNATIVE_OBSERVATIONS[1:2]`: B's XML supplies
its bandpass. For XML-free imaging, use the script's commented `OBSERVATIONS`
example with XML paths `None`, omit setup rows, and edit the hard-coded settings,
including `BANDPASS = '1A'`. Supplied XML takes precedence; invalid XML raises an error.

Choose a new output directory and run from scratch, where PROPER copies helper files:

```bash
conda activate corgisim
mkdir -p /path/to/scratch
cd /path/to/scratch
python /path/to/corgisim/examples/alt_probe_generate_L1_sims.py \
  --input-root /path/to/inputs/HLC_B1_C1 \
  --output-dir /path/to/data/new_campaign
```

Append `--full-campaign` for the full sequential campaign. `--scale` and
`--overwrite` remain available; no flags use the configured roots and A+B selection.
Visits and frames run sequentially; PROPER may use worker processes internally.

The script creates L1 visit folders and prints setup/timing context. XML timing
starts at the first imaging request in UTC and advances by exposure duration,
with no added gaps or setup overhead.

Open [Alt_probe_L1_demo.ipynb](Alt_probe_L1_demo.ipynb), set
`PROGRAM_DIR = Path('/path/to/data/new_campaign')`, `VISIT_INDEX = 0` and
`FRAME_INDEX_WITHIN_STATE = 0`, then run it to inspect existing FITS products.
Set `REDUCED_DIR` to the existing reduction root if it is stored separately.
The notebook generates no data and displays all available levels together.

## Reduce saved products through L3

The adapter uses stock CorgiDRP recipes, tested with **CorgiDRP 5.1.1** and
**EMCCDDetect 2.6.2**. Select the tested DRP checkout through process `PYTHONPATH`
or use its installed environment. Normal DRP configuration must already exist.
No package installation or user-configuration change is performed by these scripts.

Create explicit nominal simulator calibrations, then reduce existing L1:

```bash
python /path/to/corgisim/examples/alt_probe_sim_calibs.py \
  /path/to/data/new_campaign /path/to/calibrations --stop-after L3
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign \
  --output-dir /path/to/data/new_campaign/reduced \
  --calibration-dir /path/to/calibrations --stop-after L3 --crop-size 255
```

Choose the endpoint needed in both commands; L1 needs no calibration helper.
The helper defaults to L2a and also accepts a single representative FITS file.
For L2b/L3 from L1, directory input supplies every exposure/gain camera set.

| Endpoint | Stock recipe | Saved data | Required calibration |
|---|---|---|---|
| L1 | no reduction | full 1200×2200, DN | none |
| L2a | `l1_to_l2a_basic` | individual 1024×1024, DN; ERR/DQ/BIAS | detector parameters, identity nonlinearity |
| L2b | `l2a_to_l2b` | individual 1024×1024, photoelectron; ERR/DQ/BIAS | camera KGain/PTC and direct dark; identity flat, zero defect map |
| L3 | `l2b_to_l3` | individual crop, photoelectron/s; ERR/DQ/WCS | nominal astrometry only |

Resume directly from a saved level into a new destination:

```bash
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign/reduced/L2b \
  --output-dir /path/to/data/resumed/reduced \
  --calibration-dir /path/to/calibrations --stop-after L3 --crop-size 255
```

Resuming L2b→L3 requires only `sim_astrometry.fits`. The helper with L2b inputs
and `--stop-after L3` can create that file alone, without earlier-stage truth or
calibrations. Requested endpoints below the input level are rejected; matching
levels return existing paths without processing. Each stage checks only its own
prerequisites. A later-stage failure preserves lower saved FITS; resume those
products into a fresh destination.

Editable settings are at the top of the scripts; CLI values override them.
Generation remains L1-only by default. Add `--stop-after L2a|L2b|L3`, the
calibration directory and optional crop size to generate and reduce through the
same function, using an empty campaign output directory. Existing L1 generation
options remain available. The standalone reducer defaults to L2a.

Directory inputs select FITS directly inside the directory and immediate `V*`
subfolders; explicit files are also accepted. Select one common data level and
analog SCI frames. Calls are grouped by visit, exposure and commanded/effective
gain; L3 also checks every frame has a consistent EACQ reference and shape within
each group. Native recipes save under `<output>/<level>/V<visit>` with processing
history and recipe JSON. Counts report filtered frames from completed calls.
Native filtering raises and stops the run if an entire four-frame chunk is
rejected, even if later inputs are usable. Previously saved products remain;
use a new destination after a partial failure. Occupied visit outputs,
duplicate inputs and crops extending beyond the input image are rejected.

The adapter disables automatic caldb initialization before DRP imports. Every
calibration path is explicit, JIT lookup and missing-step skipping are disabled,
and any recipe database use is confined to the output root. It does not call
`autogen_recipe` or change the real DRP configuration.

### Optional checkout selection and logging

When working with separate branches or worktrees, select the intended source
checkouts before running the commands above:

```bash
SIM=/path/to/corgisim-checkout
DRP=/path/to/corgidrp-checkout
export PYTHONPATH="$SIM:$DRP${PYTHONPATH:+:$PYTHONPATH}"
```

This applies to the current terminal and its child processes. Omit it when your
environment already imports the intended versions; remove stale worktree paths
from `PYTHONPATH` when returning to your usual checkouts. The scripts themselves
disable automatic calibration-database initialization; no extra export is needed.

For long runs, optionally display progress while saving it to a log (Bash/Zsh):

```bash
set -o pipefail
python -u /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign \
  --output-dir /path/to/data/new_campaign/reduced \
  --calibration-dir /path/to/calibrations --stop-after L3 --crop-size 255 \
  2>&1 | tee reduction.log
```

`-u` prints progress promptly; `2>&1` includes errors; `pipefail` preserves a
failed exit status if Python fails. Use a new log filename for each run. This
also works for L1 generation, but does not prevent sleep or keep a job running
after the terminal closes. On a cluster, use the scheduler's job submission and
logging facilities instead.

### Nominal simulator calibrations and limits

These products describe model parameters, not measured detector calibration or
uncertainty. The helper reads detector truth from primary COMMENT cards and
rejects missing/unsupported truth. Inputs sharing a calibration directory must
have matching detector parameters. L2a assumes no extra image/prescan bias offset
and requires `nonlin_path=None`. L2b additionally requires `flat_path=None`, no
traps, `fast_gain_mode=auto` and `gain_CIC_Q=roman`. Camera files are stored under
`gain_<g>_exptime_<t>`; the zero defect map adds no invented bad pixels, while
native frame DQ continues to mask cosmic rays and saturation.

Direct dark means include exposure-scaled dark current, entrance CIC and partial
CIC from installed gain-register stages 1..N−1. Auto mode disables partial CIC at
gain≤200. The nominal model assumes prescan medians center on amplifier bias;
stochastic CIC skew can leave a small background offset. Zero calibration
parameter errors describe fixed model inputs, not measured accuracy.

The PTC is in input electrons at the native post-gain-division step. After ENF
multiplication its variance is
`ENF² max(signal − partial_CIC_mean, 0) + (read_noise/gain)² + partial_CIC_variance`.
A constant negative branch keeps uncertainties finite for negative observations.
Native DRP separately carries the prescan estimator error. The clipped
signal-based estimate is approximate; desmear changes SCI without full ERR or
pixel-covariance propagation. DQ-masked SCI may be NaN while ERR remains finite.
Stock DQ is not an independent guarantee of cosmic-ray completeness.
The activity recipe explicitly uses native `cosm_tail='auto'`, matching the
simulator's Roman gain-dependent tail length, `round(gain * 40/1000)` pixels;
homogeneous gain groups keep that setting consistent within each call.

Nominal astrometry uses the simulator's 0.0218 arcsec/pixel, north angle 0,
calibration PA_APER 0 and zero boresight correction; each image uses its commanded
RA/DEC and PA_APER. This describes the model convention, not measured pointing.
EACQ is a NumPy pixel coordinate. The adapter supplies the existing WCS offset
`EACQ + 1 − (size−1)//2` to express that same reference as FITS CRPIX. For a
1024-pixel image with EACQ=512, the offset is 2 and CRPIX is 513; a centered
255-pixel crop gives EACQ=127 and CRPIX=128. The 255 default retains the optical
stamp. Crop is pixel extraction, not registration or distortion resampling.
Native cropping retains SCI/ERR/DQ; the lower-level BIAS extension stays in L2b.

### Folder layout and program inspection

The original L1 visit folders remain directly under the program directory.
The reducer's chosen output directory contains the level folders. For example,
when that output is `program/reduced`, the existing layout is:

```text
program/                              # PROGRAM_DIR: original L1 inputs
├── V<visitid>/
│   └── cgi_<visitid>_<time>_l1_.fits
├── V<another_visitid>/...
└── reduced/                          # REDUCED_DIR: chosen --output-dir
    ├── L2a/V<visitid>/
    │   ├── cgi_<visitid>_<time>_l2a.fits
    │   └── l1_to_l2a_basic_*_recipe.json
    ├── L2b/V<visitid>/
    │   ├── cgi_<visitid>_<time>_l2b.fits
    │   └── l2a_to_l2b_*_recipe.json
    └── L3/V<visitid>/
        ├── cgi_<visitid>_<time>_l3_.fits
        └── l2b_to_l3_*_recipe.json
```

Existing reductions may be outside the program directory: keep them there and
override `REDUCED_DIR`. No `L1/` subfolder, file moves, copies or symlinks are
required. Calibrations also remain at the explicitly chosen calibration path.
Native recipe JSON is saved per level/visit, with separate camera calls when needed.
The folder-level reduction command above processes all immediate `V*` visits;
it does not require listing individual exposures.

The notebook configuration matches these two existing roots:

```python
PROGRAM_DIR = Path('/path/to/program')
REDUCED_DIR = PROGRAM_DIR / 'reduced'  # Or Path('/path/to/existing/reductions').
VISIT_INDEX = 0
FRAME_INDEX_WITHIN_STATE = 0
```

The viewer inventories every available level and visit, then shows the selected
VISITID in four explicit **L1, L2a, L2b and L3** sections. Missing levels are
reported. Each state's selected exposure is identified by SCTSRT in the lowest
available level and matched across levels. A filtered counterpart is shown as
missing, rather than replaced with another indexed exposure.

Each section compares a single template trio with its arithmetic means, retains
the matched-template difference panels, and shows single/mean science separately
when present. Means stay in memory, with input N and per-pixel valid coverage;
no combined FITS products are created. Samples must be finite and, for calibrated
data, have DQ=0. Zero-coverage pixels are masked. Combinations never cross levels,
visits, states/probe signs, bands, exposure/gain/PC settings, pointing/roll or
reference geometry. There is no alignment, interpolation, weighting or clipping.
Raw L1 has no DQ, so its mean can retain cosmic contamination. Later means also
do not guarantee complete cosmic-ray removal. White indicates excluded samples,
missing selected exposures or zero valid coverage. ERR summaries remain per-frame
native uncertainties; the viewer does not infer combined uncertainty.

### Inspect an individual frame in DS9

The primary HDU (index0) contains metadata, with no image. The science array is
**image extension1**, which currently has no EXTNAME, even in L3. Do not require
an extension named SCI. L1 has only primary and image HDUs; calibrated files add
ERR (extension2) and DQ (extension3). L2a/L2b also retain BIAS.

Open a saved science array explicitly, substituting its actual filename:

```bash
ds9 "/path/to/reduced/L3/V<visitid>/cgi_<visitid>_<time>_l3_.fits[1]"
```

Use `[2]` or `[ERR]` for ERR and `[3]` or `[DQ]` for DQ in calibrated files.
ERR is a one-plane cube; display its first plane if prompted. Nonzero DQ means
flagged; L2b/L3 science pixels may already be NaN there. Image units are DN for
L1/L2a, photoelectron for L2b and photoelectron/s for L3. DS9's
[qualified filename syntax](https://ds9.si.edu/doc/ref/file.html) accepts extension
numbers or names. The notebook adds its DQ mask for display; a DS9 view of the
science HDU alone does not automatically apply a separate DQ extension.
