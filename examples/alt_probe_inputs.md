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
`CAMPAIGN_DIR = Path('/path/to/data/new_campaign')`, `VISIT_INDEX = 0` and
`FRAME_INDEX_WITHIN_STATE = 0`, then run it to inspect existing FITS products.
The notebook generates no data.

## Reduce saved frames through L2b

[alt_probe_reduce.py](alt_probe_reduce.py) uses CorgiDRP's stock
`l1_to_l2a_basic.json`: prescan bias subtraction, acquisition-frame filtering,
cosmic-ray flags, nonlinearity correction and individual L2a saves in DN.
Configure its top processing settings or supply CLI overrides:

```bash
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign/V0200001001001911001 \
  --output-dir /path/to/data/new_campaign/reduced \
  --calibration-dir /path/to/calibrations --stop-after L2a
```

This path was tested with CorgiDRP5.1.1 and EMCCDDetect2.6.2.
Use the tested CorgiDRP checkout on `PYTHONPATH` (or its installed environment).
Normal DRP configuration must already exist. The adapter disables automatic
calibration database initialization before importing DRP, uses explicit
calibration files, and confines any recipe database use to the output root.
It does not change user configuration. No automatic calibration lookup is used.

For Corgisim L1 generated **without simulated nonlinearity**, create the two
required model-truth files explicitly:

```bash
python /path/to/corgisim/examples/alt_probe_sim_calibs.py \
  /path/to/data/new_campaign/V0200001001001911001/representative_l1.fits \
  /path/to/calibrations
```

The helper reads full wells, e-/DN, read noise, row time and gain-register length
from primary COMMENT cards; check that these values match every input camera.
`sim_detector_params.fits` and `sim_identity_nonlin.fits` describe the simulator,
not measured calibrations. The identity table is valid only when `nonlin_path`
is `None`; the helper rejects other values or absent truth. The recipe uses
DetectorParams' gain conversion, with no separate KGain product, and assumes
zero additional image/prescan bias offset (`DetectorNoiseMaps=None`). Replace
the two calibration files with matching native DRP products for other inputs.

For analog L2b, generate a new calibration directory with all required cameras:

```bash
python /path/to/corgisim/examples/alt_probe_sim_calibs.py \
  /path/to/data/new_campaign /path/to/l2b_calibrations --stop-after L2b
python /path/to/corgisim/examples/alt_probe_reduce.py \
  --inputs /path/to/data/new_campaign/reduced/L2a \
  --output-dir /path/to/data/new_campaign/reduced \
  --calibration-dir /path/to/l2b_calibrations --stop-after L2b
```

The helper accepts a directory or individual FITS paths. Its default single-file
L2a use remains available. L2b additionally requires simulator metadata for an
identity flat, absent traps and `fast_gain_mode=auto`, `gain_CIC_Q=roman`.
Unsupported or missing truth raises an error. Each `gain_<g>_exptime_<t>` folder
contains a native KGain/PTC, direct Dark, identity FlatField and zero BadPixelMap.
The zero map represents the simulator's absence of additional detector defects;
the recipe still masks cosmic rays and saturation using each frame's DQ.
TrapCalibration is None; no CTI correction is inferred or simulated.

Dark means include exposure-scaled dark current, entrance CIC and installed
EMCCDDetect's partial CIC from gain-register stages1..N-1. Auto gain mode disables
partial CIC at gain≤200. The mean model assumes that prescan medians center on
the amplifier bias; a stochastic skew from CIC can leave a small background
offset. These analytic calibrations support pipeline checks, not precision
detector calibration. Their zero parameter errors describe fixed model inputs,
not measured calibration accuracy or the absence of observation noise.

The PTC is tabulated in input electrons at the point where the stock recipe
uses it, after gain division. After native ENF multiplication, its variance is
`ENF² max(signal − partial_CIC_mean, 0) + (read_noise/gain)² + partial_CIC_variance`.
A constant negative branch keeps the uncertainty finite for negative data.
Native DRP carries prescan uncertainty and adds the shot/read contribution.
The signal-dependent estimate and native desmear/error handling are
approximations: desmear adjusts SCI without propagating pixel covariance or
changing ERR. DQ flags identify the pixels replaced by NaN; their ERR may remain
finite. The nominal dark has no measured-stack uncertainty.

Directory input selects FITS directly in it and in immediate `V*` subfolders;
explicit FITS paths can also follow `--inputs`. Every selected input must have
one common level, with analog SCI arrays. L1 requires full 1200×2200 frames.
Calls are separated by visit, exposure, commanded and effective gain. Frames remain
individual and retain their state metadata. Outputs are saved under
`<level>/V<visit>` with native DRP recipe/history. Counts printed per call include
any frames filtered by the recipe. Use a fresh destination; occupied visit
folders are rejected so failed or stale products cannot be mistaken for a
complete run. Saved input at its requested endpoint is returned unchanged.
L1 can run through L2a to L2b, or processing can resume directly from saved L2a.
Each stage checks only its own calibration dependencies. A later-stage failure
preserves the lower saved FITS; resume that level into a fresh destination.

Generation stays L1-only by default. To reduce immediately after generation,
add `--stop-after L2b --calibration-dir /path/to/l2b_calibrations` to the generation
command, using a new empty campaign directory. Both entry points use the same
reduction function. L3 and crop controls are not implemented at this checkpoint.
