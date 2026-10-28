# Alternate and Standard Probe workflow

## 1. Get the inputs

Download these four pinned ZIPs in a browser signed in to GitHub with access to `roman-corgi/corgiobsspecs`, then unzip them:

| Inputs | Source branch/folder | Pinned ZIP download |
|---|---|---|
| Alternate XML (12 observations) | [HLC_B1_C1_alternative_probes](https://github.com/roman-corgi/corgiobsspecs/tree/HLC_B1_C1_alternative_probes/obs_specs/HLC_B1_C1/alternative_probes) | [2c5f340846db5f00f4f8e634619d32725fe8249f](https://github.com/roman-corgi/corgiobsspecs/archive/2c5f340846db5f00f4f8e634619d32725fe8249f.zip) |
| Standard XML (11 observations) | [HLC_B1_C1_standard_probes](https://github.com/roman-corgi/corgiobsspecs/tree/HLC_B1_C1_standard_probes/obs_specs/HLC_B1_C1/standard_probes) | [aa4688822d32bc3c1e34184bca0df9de457cd423](https://github.com/roman-corgi/corgiobsspecs/archive/aa4688822d32bc3c1e34184bca0df9de457cd423.zip) |
| Alternate probes (3 FITS files) | [alternative_probes_DM_commands](https://github.com/roman-corgi/corgiobsspecs/tree/alternative_probes_DM_commands/obs_specs/HLC_B1_C1/alternative_probes/aux_files) | [c6a6a9386d8b3a01f48abd7bc808a16a0e2fe936](https://github.com/roman-corgi/corgiobsspecs/archive/c6a6a9386d8b3a01f48abd7bc808a16a0e2fe936.zip) |
| Standard probes (3 FITS files) | [standard_probes_DM_commands](https://github.com/roman-corgi/corgiobsspecs/tree/standard_probes_DM_commands/obs_specs/HLC_B1_C1/standard_probes/aux_files) | [b66e9153eae8ce82583cf0ffec268b15f29f68b2](https://github.com/roman-corgi/corgiobsspecs/archive/b66e9153eae8ce82583cf0ffec268b15f29f68b2.zip) |

Each ZIP contains the whole repository at the listed branch's pinned commit and expands into
`corgiobsspecs-<commit>`; it does not include all branches. Cloning is useful when editing observation
specifications; running this example only requires the downloaded files. Keep inputs outside the Corgisim checkout.
Copy each XML archive's `obs_specs/HLC_B1_C1/<activity>` folder into your input root, then add the matching
DM-command archive's `aux_files` inside that activity folder without replacing its XML observation folders.
The XML branches supply observation settings; the separate DM-command branches supply the relative-DM
voltage arrays in `aux_files`. Preserve the original filenames:

```text
/path/to/inputs/HLC_B1_C1/
├── alternative_probes/
│   ├── observation_a/alternative_probe_CFAM_1A.xml
│   ├── observation_b/alternative_probe_1A_probe1.xml
│   ├── ... (through observation_l)
│   └── aux_files/
│       ├── nfov_dm_dmrel_5.0e-07_gaussian0.fits
│       ├── nfov_dm_dmrel_5.0e-07_gaussian1.fits
│       └── nfov_dm_dmrel_5.0e-07_gaussian2.fits
└── standard_probes/
    ├── observation_a/standard_probe_1C_probe1.xml
    ├── ... (through observation_k)
    └── aux_files/
        ├── nfov_dm_dmrel_5.0e-07_standard0.fits
        ├── nfov_dm_dmrel_5.0e-07_standard1_unmod.fits
        └── nfov_dm_dmrel_5.0e-07_standard2.fits
```

## 2. Configure and generate L1

At the top of [alt_probe_generate_L1_sims.py](alt_probe_generate_L1_sims.py), set the default roots:

```python
INPUT_ROOT = Path('/path/to/inputs/HLC_B1_C1')
DEFAULT_OUTPUT_DIR = Path('/path/to/data/new_campaign')
```

`INPUT_ROOT` is the parent of both activity folders shown above. The script derives
`ALTERNATIVE_XML_ROOT` and `STANDARD_XML_ROOT` from it, then derives each `*_PROBE_ROOT` by appending
`aux_files`. Those folders hold the three Gaussian alternative probe arrays and three standard arrays
listed above; the observation rows bind them to the corresponding XML visits. For another layout,
edit the derived XML/probe roots. Probe arrays can remain separately stored.

CLI values override the configured defaults. `--input-root` relocates only configured paths beneath
`INPUT_ROOT`; explicit XML or probe paths outside it stay unchanged. Activate the project environment
and run from a scratch directory, where PROPER copies helper files. Choose a new output directory:

```bash
conda activate corgisim
mkdir -p /path/to/scratch
cd /path/to/scratch
python /path/to/corgisim/examples/alt_probe_generate_L1_sims.py \
  --input-root /path/to/inputs/HLC_B1_C1 \
  --output-dir /path/to/data/new_campaign
```

Default selection is alternate A setup + B imaging. `--full-campaign` supersedes the editable `OBSERVATIONS`
selection and runs all 12 alternate + 11 standard entries: five no-image setup entries and 18 imaging visits.
Visits and frames run sequentially; PROPER may use workers internally. XML timing starts at the first imaging
request in UTC and advances by exposure duration without added gaps or setup overhead.
To skip setup A, set `OBSERVATIONS = ALTERNATIVE_OBSERVATIONS[1:2]`; B's XML supplies its bandpass.

Keep `SCALE=1.0` (or `--scale 1.0`) to apply supplied arrays directly. Optional `DM1_PATH`/`DM2_PATH` replace
the installed `hlc_ni_3e-8` base maps. `--overwrite` allows replacement of existing L1 products;
otherwise existing products are refused. Use `--help` for all options.

For simple XML-free imaging, use the commented `OBSERVATIONS` example with XML paths `None`, omit setup rows and edit the hard-coded
settings, including `BANDPASS='1A'` for Band 1A (choose the intended band for other manual visits).
Supplied imaging XML takes precedence for the bandpass and observation settings; invalid XML raises an error.

## 3. Reduce to the required level

The adapter uses stock recipes, tested with CorgiDRP 5.1.1 and EMCCDDetect 2.6.2. Use an environment with normal DRP configuration
already present. Inputs must share one data level and contain analog (`ISPC=0`) SCI arrays; L1 must be full 1200×2200 frames.
A directory input selects FITS directly inside it and its immediate `V*` folders; individual file paths are also accepted.

Create nominal simulator calibrations in an empty calibration directory, then reduce saved L1.
The helper also accepts a representative FITS file, but directory input is needed to collect every
exposure/gain camera set for L2b/L3 from L1:

```bash
python /path/to/corgisim/examples/alt_probe_sim_calibs.py \
  /path/to/data/new_campaign /path/to/calibrations --stop-after L3
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign \
  --output-dir /path/to/data/new_campaign/reduced \
  --calibration-dir /path/to/calibrations --stop-after L3 --crop-size 255
```

| Endpoint | Stock recipe | Saved data | Required calibration |
|---|---|---|---|
| L1 | no reduction | full 1200×2200, DN | none |
| L2a | `l1_to_l2a_basic` | individual 1024×1024, DN; ERR/DQ/BIAS | detector parameters, identity nonlinearity |
| L2b | `l2a_to_l2b` | individual 1024×1024, photoelectron; ERR/DQ/BIAS | camera KGain/PTC and direct dark; identity flat, zero defect map |
| L3 | `l2b_to_l3` | individual crop, photoelectron/s; ERR/DQ/WCS | nominal astrometry only |

L3 defaults to a 255×255 crop, set with `--crop-size`. Choose `--stop-after L2a`, `L2b` or `L3` in both
commands; both default to L2a. L1 generation needs no calibrations and remains L1-only by default. Its `--stop-after`,
`--calibration-dir` and `--crop-size` flags can also request reduction after generation into a new empty campaign directory,
using prepared matching calibrations.

Resume from a saved level into a fresh destination:

```bash
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign/reduced/L2b \
  --output-dir /path/to/data/resumed/reduced \
  --calibration-dir /path/to/calibrations --stop-after L3
```

L2b→L3 needs only `sim_astrometry.fits`; the calibration helper can create it from L2b inputs with `--stop-after L3`. Native
filtering stops the run if an entire four-frame chunk is rejected, even if later inputs are usable. Saved products remain; use a new
destination after a partial failure. An endpoint below the input level is rejected; a matching level returns existing paths.
Each stage checks only its own prerequisites. Occupied visit outputs, duplicate inputs and crops outside the input image
are rejected. Calls are grouped by visit, exposure and commanded/effective gain; L3 requires consistent EACQ/shape within a group.
The scripts disable automatic caldb initialization, use explicit calibration paths and confine recipe database use to the output root;
they do not modify the real DRP configuration.

### Optional checkout selection and logging

For separate source checkouts, set these before running commands; omit them when your environment already imports the intended
versions:

```bash
SIM=/path/to/corgisim-checkout
DRP=/path/to/corgidrp-checkout
export PYTHONPATH="$SIM:$DRP${PYTHONPATH:+:$PYTHONPATH}"
```

Remove stale worktree paths when returning to usual checkouts. For Bash/Zsh logging, run `set -o pipefail`, add `-u` after `python`
and append `2>&1 | tee reduction.log` to your command. `-u` prints progress promptly; `2>&1` includes errors;
`pipefail` preserves a failed exit status if Python fails. Use a new log for each run. Logging does not prevent sleep
or keep jobs alive after the terminal closes. On clusters, use scheduler logging.

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

## 4. Inspect the saved products

The output layout keeps original L1 and each derived level separate:

```text
new_campaign/                       # PROGRAM_DIR
├── V<visitid>/*.fits                # L1
└── reduced/                        # REDUCED_DIR (or an existing external root)
    ├── L2a/V<visitid>/*.fits
    ├── L2b/V<visitid>/*.fits
    └── L3/V<visitid>/*.fits
```

Native recipe JSON is saved alongside derived products. Open [Alt_probe_L1_demo.ipynb](Alt_probe_L1_demo.ipynb) and set:

```python
PROGRAM_DIR = Path('/path/to/data/new_campaign')
REDUCED_DIR = PROGRAM_DIR / 'reduced'  # Override for existing reductions elsewhere.
VISIT_INDEX = 0
FRAME_INDEX_WITHIN_STATE = 0
```

The read-only notebook inventories all available levels/visits and displays the selected visit's individual template/science frames,
differences and arithmetic means in explicit L1/L2a/L2b/L3 sections. Counts describe files present, not XML completeness.
Each state's exposure is identified by SCTSRT in the lowest available level and matched across levels; missing levels or
filtered counterparts are reported without substituting a different indexed exposure.

Means show input N and per-pixel valid coverage. They combine only matching level, visit, state/probe sign, band,
exposure/gain/PC, pointing/roll and reference geometry, using finite samples and DQ=0 where available.
White marks excluded samples, missing counterparts or zero coverage. Means stay in memory without alignment,
interpolation, weighting, clipping, combined uncertainty or new combined FITS. Raw L1 lacks DQ;
means at any level can retain cosmic contamination. ERR summaries describe native per-frame uncertainties.

For DS9, select science extension 1 explicitly:

```bash
ds9 "/path/to/data/new_campaign/reduced/L3/V<visitid>/cgi_<visitid>_<time>_l3_.fits[1]"
```

Primary HDU 0 is metadata only. Science extension 1 has no EXTNAME, even in L3, so do not require an extension named SCI.
Calibrated files have ERR at extension 2 (`[ERR]`, a one-plane cube) and DQ at 3 (`[DQ]`); choose the first ERR plane if prompted.
Nonzero DQ flags samples; L2b/L3 science may already be NaN there. DS9 does not automatically apply DQ to science.
L1 has neither ERR nor DQ. Units are given in the table above.
