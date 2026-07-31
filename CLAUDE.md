# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project runs on the BYU HPC cluster (SLURM). The conda environment is `tomoMono`:

```bash
conda activate tomoMono
# or use the full path directly:
/home/ljh79/.conda/envs/tomoMono/bin/python
```

Create/update the environment from scratch:
```bash
conda env create -f environment.yml
```

Key dependencies: Python 3.12, TomoPy, SVMBIR, ASTRA Toolbox 2.1 (CUDA 11.0), CuPy (cuda12x via pip), PyTorch 2.4.1, OpenCV, scikit-image, scipy, numpy 1.26, tifffile, h5py. GANrec is only needed by the archived GANrec scripts.

There is no test suite, linter config, or build step — this is a research codebase driven by scripts and notebooks.

## Running Scripts

Only three Python entry points live at the root; everything else is a notebook or archived.

**Staged alignment pipeline (GPU — the production alignment path):**
```bash
python align.py                          # runs the 4x → 2x → 1x staged pipeline
sbatch runGPUAlign.sh                    # cluster (6 h, 1 GPU, 4 CPUs x 100 GB = 400 GB)
```
`align.py` has no CLI — the input HDF5 path, dropped angle indices, output dirs, and per-stage
alignment parameters are edited inside its `if __name__ == '__main__':` block.

**Single reconstruction from pre-aligned projections:**
```bash
python main.py
```
Also no CLI. Edit the configuration block at the top of [main.py](main.py): `TIFF_FILE`,
`OUTPUT_DIR`, `ALGORITHM`, `NUM_ITER`, `SAVE`, `RAW_HDF5` (angles are read from the raw HDF5),
`DROP_ANGLES`.

**Reconstruction algorithm / hyperparameter search:**
```bash
python recon_param_search.py --tiff-file <path> [--y-start N] [--y-end N] [--width N] [--output-dir <dir>] [--no-save]
sbatch runTomopyParamSearch.sh           # cluster (6 h, 1 GPU, 4 CPUs x 125 GB = 500 GB)
```
The only script with argparse. It runs two sweeps back to back: an algorithm comparison
(SIRT_CUDA / ART_CUDA / FBP_CUDA / gridrec / tv) and a SIRT_CUDA hyperparameter sweep
(100/200/400/600 iterations plus a positivity-constrained run), scoring each config with RCS and
FSC and saving a TIFF plus an orthogonal-slice PNG.

Logs go to `logs/`, SLURM stdout/err goes to `sbatch_output/<jobname>/` (create that subdirectory
before submitting — not every sbatch script does `mkdir -p`), parameter-search CSVs go to
`hyperparam_results/`.

## Architecture

Root-level Python package with three subpackages: `alignment/`, `metrics/`, `filters/`.

### Core class: `tomoData` ([tomoDataClass.py](tomoDataClass.py))

All state lives in a `tomoData` instance:
- `data` — original raw projections (never modified after `jitter()`)
- `workingProjections` — scratch copy the alignment routines shift repeatedly
- `finalProjections` — the buffer that gets reconstructed; receives committed shifts
- `tracked_shifts` — per-projection (y, x) shift accumulator, zeroed by `make_updates_shift()`
- `tracked_rotations` — per-projection rotation accumulator, applied by `make_updates_rotate()`
- `ang` — projection angles in radians
- `rotation_center` / `center_offset` — set by `center_projections()` and `reconstruct()`
- `recon` — 3D volume after `reconstruct()`; also cached as `_recon_pre_kovacik` before `kovacik_filter()`
- `finalReprojections` — cached reprojections used by RCS; set to `None` to force recomputation

**Two-buffer alignment pattern**: alignment methods update `workingProjections` and accumulate
into `tracked_shifts`. `make_updates_shift()` commits those shifts to `finalProjections` in a
single subpixel interpolation pass (avoids stacking interpolation blur across rounds).
`reconstruct()` always runs on `finalProjections`.

**Delegate pattern** (bottom of the module): the free functions in `alignment/`, `metrics/`, and
`filters/` are auto-attached to the class by `_attach_delegate`, so `tomo.cross_correlate_align(...)`
and `cross_correlate_align(tomo, ...)` are equivalent. To add a new alignment/metric/filter
function, write it as a free function taking `tomo` first, export it from that subpackage's
`__init__.py`, and add it to the `for _fn in (...)` tuple at the bottom of `tomoDataClass.py`.

Other notable methods: `crop(new_y, new_x, anchor='center'|'bottom')` (`crop_center` /
`crop_bottom_center` are deprecated shims), `reset_workingProjections()`, `normalize(isPhaseData)`,
`standardize(isPhaseData)`, `center_projections()` (iterative `tomopy.find_center_vo` centering,
max 3 passes), the `shift_envelope` / `shift_envelope_idx` properties (how many edge pixels
accumulated shifts have exposed, and which projection is responsible), and the
`makeNotebook*Movie` / `makeScript*Movie` / `displayReconOrthogonalSlices` viewers.

`simulate_projections(recon, angles, ...)` is a module-level function (also exposed as the
`simulateProjections` method) that forward-projects with ASTRA `FP3D_CUDA` and falls back to
`tomopy.project`; it auto-disables ASTRA when no GPU is present.

### GPU backend ([gpu.py](gpu.py))

Centralizes GPU detection so other modules don't run their own try/except ladders. Probes run once
at import time and print a one-line backend banner. It also raises `NUMEXPR_MAX_THREADS` before
tomopy imports (login nodes otherwise abort). Exports:
- `xp` — CuPy if a working GPU is available, else NumPy
- `cp` — CuPy module or `None`
- `torch` — PyTorch module when a CUDA/MPS device is present, or `None` (used as a feature flag)
- `svmbir` — SVMBIR module or `None`
- `ndimage_shift`, `gaussian_filter`, `fourier_shift` — GPU-aware drop-ins for scipy.ndimage equivalents
- `to_numpy(arr)` — convert xp array to numpy without copying if already numpy

### Alignment package ([alignment/](alignment/))

Standalone functions that take a `tomoData` as first argument (the class delegates to them). Import
from the `alignment` package directly (e.g. `from alignment import cross_correlate_align`).

| Module | Functions |
|---|---|
| [alignment/cross_correlate.py](alignment/cross_correlate.py) | `cross_correlate_align`, `compute_grad_image` |
| [alignment/pma.py](alignment/pma.py) | `projection_matching_alignment` |
| [alignment/vmf.py](alignment/vmf.py) | `vertical_mass_fluctuation_align` |
| [alignment/legacy.py](alignment/legacy.py) | `tomopy_align`, `optical_flow_align`, `rotate_correlate_align`, `find_optimal_rotation`, `bilateralFilter`, `shift_min_to_middle`, `unrotate` |

Key functions:
- `cross_correlate_align` — sequential XC between adjacent projections; ROI, downsampling pyramid, gradient mode, rolling median reference
- `projection_matching_alignment` — reconstruct → forward-project → measure shift (phase-XC or Lucas-Kanade optical flow) → apply; multi-scale via `levels`/`scale`
- `vertical_mass_fluctuation_align` — registers per-row mass profiles against the mean profile across angles; corrects vertical drift only
- `rotate_correlate_align` — corrects rotational misalignment

Note: `shift_method='optical_flow'` inside PMA solves for a single rigid (dy, dx) per projection.
The standalone `optical_flow_align` in `legacy.py` is a different operation — it warps images
non-rigidly and does *not* update `tracked_shifts`, so it cannot be composed with the two-buffer
pattern.

### Metrics package ([metrics/](metrics/))

| Module | Function | Use for |
|---|---|---|
| [metrics/fsc.py](metrics/fsc.py) | `fourier_shell_correlation` | Reconstruction quality — half-set FSC resolution estimate (most trustworthy) |
| [metrics/reprojection_consistency.py](metrics/reprojection_consistency.py) | `reprojection_consistency_score` | Alignment quality — per-angle NRMSE of measured vs. reprojected (most reliable for alignment) |
| [metrics/sinogram_consistency.py](metrics/sinogram_consistency.py) | `sinogram_consistency_score` | Rough gauge only — Helgason-Ludwig CoM check; useful for spotting outlier projections |

The sharpness metric was removed — don't reintroduce references to `metrics/sharpness.py`
(an old copy still sits in `Archive/sharpness.py`).

### Filters package ([filters/](filters/))

| Module | Function |
|---|---|
| [filters/kovacik.py](filters/kovacik.py) | `kovacik_filter` — post-reconstruction soft Fourier angular filter for missing-wedge ray artifacts (Kovacik et al. 2014); edits `tomo.recon` in place |

### Helper utilities ([helperFunctions.py](helperFunctions.py))

- `subpixel_shift` — Fourier-domain subpixel shift (GPU-dispatched)
- `convert_to_numpy` / `convert_to_tiff` / `convert_to_2Dtiff` — TIFF I/O carrying scale metadata (`convert_to_numpy` returns `(array, scale_info)`)
- `DualLogger` — tees stdout to both console and log file
- `MoviePlotter` / `runwidget` — interactive projection/slice viewers for Jupyter and scripts
- `degree_to_positiveRadians` — angle unit conversion
- `FFT` / `FFT2` / `IFFT` / `IFFT2`, `show`, `add_noise` — small numerical helpers

### Data flow for a typical run

```
Load HDF5 → tomoData(projs, angles)
  → normalize(isPhaseData=True)
  → cross_correlate_align(...)           # updates workingProjections + tracked_shifts
  → center_projections()
  → make_updates_shift()                 # commits to finalProjections
  → [crop() if needed]
  → projection_matching_alignment(...)   # updates workingProjections + tracked_shifts
  → make_updates_shift()
  → reconstruct(algorithm='SIRT_CUDA')   # runs on finalProjections → self.recon
  → kovacik_filter()                     # refines self.recon in-place
  → convert_to_tiff(...)
```

`align.py` runs this at three resolutions, carrying `tracked_shifts` forward between stages by
scaling them by 2 and seeding the next stage before its PMA pass.

### Reconstruction algorithms

`tomoData.reconstruct(algorithm, snr_db=None, num_iter=400, extra_options=None, crop_ratio=0.98)`
dispatches on the string:
- `'*_CUDA'` (`SIRT_CUDA`, `ART_CUDA`, `FBP_CUDA`) — ASTRA GPU via `tomopy.astra`; raises if no GPU
- `'svmbir'` — SVMBIR MBIR (CPU, slow, high quality); output remapped to tomopy geometry by `_correct_svmbir_geometry`
- anything else — `tomopy.recon(algorithm=...)` on CPU (`'gridrec'`, `'sirt'`, `'art'`, `'tv'`)

`extra_options` is passed straight to ASTRA (e.g. `{'MinConstraint': 0}` for positivity). A
circular mask of `crop_ratio` is applied to the finished volume. Note that `GRIDREC_CUDA` and
`TV_CUDA` do not exist — those algorithms are CPU-only.

### Notebooks

Root-level (active workflows):
- [tomoMono_demo.ipynb](tomoMono_demo.ipynb) — end-to-end walkthrough on a simulated Shepp-Logan phantom; the best reference for API usage
- [weddingCake_roughDraft.ipynb](weddingCake_roughDraft.ipynb) — same sequence as the demo but on real measured phase projections (TPP foam "wedding cake" sample)
- [densityConversion.ipynb](densityConversion.ipynb) — scale a reconstruction and map voxel intensity to mass density (histograms, region segmentation → `massDensity/*.tif`, `figures/*.pdf`)

[notebooks/](notebooks/) (reference/experimental, mostly unmaintained):
`debug_FSC_resolution.ipynb`, `lookAtRecons.ipynb`, `recon_algorithm_comparison.ipynb`,
`FourierRingCorrelation.ipynb`, `featureSizeFFT.ipynb`, `projectXRFonVolume.ipynb`,
`test_notebook_phanton.ipynb`, `alignmentExperimenting.ipynb`, `sinogramMaker.ipynb`,
`centroidPlotter.ipynb`, `SeedingPtychographyRecons.ipynb`, `shiftWithPhaseRamp.ipynb`,
`testingFilters.ipynb`, `Drop worst contrast images.ipynb`, `Astra_project_test.ipynb`,
`makeJungFrauFigure.ipynb`. These reach the root package via a `sys.path.insert` preamble.

### Archive

[Archive/](Archive/) holds retired code and old outputs — do not treat it as live:
- `Archive/svmbir/` — `runSVMBIRrec.py`, `runSVMBIRparamSearch.py` and their sbatch scripts
- `Archive/ganrec/` — `runGANrec.py`, `runGPUGANrec.sh`, the GANrec walkthrough notebooks, and a vendored `ganrec_pkg`
- `Archive/hyperparameter_search.py` + `runHyperparamSearch.sh` — XCA/PMA alignment config search
- `Archive/taylor_tomo_align.py`, `taylor_tomo_recon.py`, `sharpness.py`, `test_notebook_realData.ipynb`, and archived aligned/recon TIFFs

If a task needs SVMBIR or GANrec runs, these archived scripts are the starting point, but their
paths and imports need checking against the current package layout.

## Data Locations

- Raw experimental data: `/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/` (cluster, not in repo). `tomo_data_run_final_2.hdf5` is the Oct 2025 tilt series; angle indices 19 and 26 are dropped as bad in every script that reads it.
- Aligned projections (TIFF): `alignedProjections/`, with `sinograms/` and `shifts/` subdirs written by `align.py`
- Reconstructions (TIFF): `reconstructions/`
- Small test phantoms and angle files (HDF5/npy): `data/`
- Mass density outputs (TIFF): `massDensity/`
- Plots and figures: `figures/`
- XRF scan data (HDF5): `XRF_Data/`
- Parameter-search CSVs: `hyperparam_results/`

`.gitignore` excludes all `*.tif`/`*.tiff` plus `data/`, `figures/`, `reconstructions/`,
`alignedProjections/`, `logs/`, `hyperparam_results/`, and `sbatch_output/` — data products stay
local; only code and notebooks are tracked.

## HPC Notes

The cluster uses SLURM. GPU jobs need `--gpus=1` and enough RAM for the data at the chosen
downsampling level (full-res Oct25 data requires ~160 GB, so the sbatch scripts request 400–500 GB
total across 4 CPUs; 4× downsample fits in ~12 GB/CPU). Compute directories are not backed up —
only home directories are.
