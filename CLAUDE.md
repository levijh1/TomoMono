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

Key dependencies: Python 3.12, TomoPy, SVMBIR, ASTRA Toolbox 2.1 (CUDA 11.0), CuPy (cuda12x via pip), PyTorch 2.4.1, GANrec, OpenCV, scikit-image, scipy, numpy 1.26, tifffile, h5py.

## Running Scripts

**Alignment pipeline (GPU-intensive):**
```bash
python align.py                          # interactive / local run
sbatch runGPUAlign.sh                    # submit to SLURM cluster (48h, 1 GPU, 200 GB RAM)
```

**Single SIRT_CUDA reconstruction from pre-aligned projections:**
```bash
python main.py --tiff-file <path> [--y-start N] [--y-end N] [--width N] [--output-dir <dir>] [--no-save]
```

**TomoPy algorithm/hyperparameter search (multi-algorithm comparison):**
```bash
python recon_param_search.py --tiff-file <path> [--y-start N] [--y-end N] [--width N] [--output-dir <dir>] [--no-save]
sbatch runTomopyParamSearch.sh           # cluster submission (1 GPU, 500 GB RAM)
```

**Hyperparameter search over XCA + PMA alignment configs:**
```bash
python hyperparameter_search.py
python hyperparameter_search.py --resume --logfile hyperparam_results/<name>.csv
sbatch runHyperparamSearch.sh            # cluster submission
```

**SVMBIR reconstruction from pre-aligned projections (CPU-only, high quality):**
```bash
python runSVMBIRrec.py --tiff-file <path> --y-start N --y-end N --width N
sbatch runSVMBIRrec_fullres.sh           # full-res (32 CPUs, 208 GB, 72h)
```

**SVMBIR hyperparameter search (scans qGGMRF/solver settings):**
```bash
python runSVMBIRparamSearch.py --tiff-file <path> [--fsc] [--num-workers N]
sbatch runSVMBIRparamSearch.sh           # cluster submission (32 CPUs, 4 workers)
```

**GANrec reconstruction:**
```bash
python runGANrec.py
sbatch runGPUGANrec.sh                   # GPU job
```

Logs go to `logs/`, SLURM stdout/err goes to `sbatch_output/`, hyperparam results go to `hyperparam_results/`.

## Architecture

The codebase is organized as a root-level Python package with three subpackages for alignment, metrics, and filters.

### Core class: `tomoData` ([tomoDataClass.py](tomoDataClass.py))

All state lives in a `tomoData` instance:
- `data` — original raw projections (never modified after `jitter()`)
- `workingProjections` — scratch copy used during iterative alignment
- `finalProjections` — accumulates committed shifts for final reconstruction
- `tracked_shifts` — per-projection (y, x) shift accumulator, reset by `make_updates_shift()`
- `recon` — 3D volume after `reconstruct()`; also cached as `_recon_pre_kovacik` before `kovacik_filter()`

**Two-buffer alignment pattern**: alignment methods update `workingProjections` and accumulate into `tracked_shifts`. Call `make_updates_shift()` to commit those shifts to `finalProjections` via a single subpixel interpolation pass (avoids stacking interpolation error). `reconstruct()` always runs on `finalProjections`.

GPU acceleration is centralized in `gpu.py` (see below). ASTRA is used for GPU forward projection inside `simulate_projections()` (also exported as a standalone function from `tomoDataClass`) with a tomopy fallback.

### GPU backend ([gpu.py](gpu.py))

Centralizes GPU detection so other modules don't run their own try/except ladders. Probes run once at import time. Exports:
- `xp` — CuPy if a working GPU is available, else NumPy
- `cp` — CuPy module or `None`
- `torch` — PyTorch module when CUDA/MPS device is present, or `None` (used as feature flag)
- `svmbir` — SVMBIR module or `None`
- `ndimage_shift`, `gaussian_filter`, `fourier_shift` — GPU-aware drop-ins for scipy.ndimage equivalents
- `to_numpy(arr)` — convert xp array to numpy without copying if already numpy

### Alignment package ([alignment/](alignment/))

Standalone functions that take a `tomoData` as first argument (the class delegates to them). Import from `alignment` directly; `alignment_methods.py` is a backwards-compatibility shim that re-exports everything.

| Module | Functions |
|---|---|
| [alignment/cross_correlate.py](alignment/cross_correlate.py) | `cross_correlate_align`, `compute_grad_image` |
| [alignment/pma.py](alignment/pma.py) | `projection_matching_alignment` |
| [alignment/vmf.py](alignment/vmf.py) | `vertical_mass_fluctuation_align` |
| [alignment/legacy.py](alignment/legacy.py) | `tomopy_align`, `optical_flow_align`, `rotate_correlate_align`, `find_optimal_rotation`, `bilateralFilter`, `shift_min_to_middle`, `unrotate` |

Key functions:
- `cross_correlate_align` — sequential XC between adjacent projections; ROI, downsampling pyramid, gradient mode, rolling median reference
- `projection_matching_alignment` — Projection Matching Alignment: reconstruct → forward-project → measure shift (phase-XC or optical flow) → apply; multi-scale via `levels` param
- `rotate_correlate_align` — corrects rotational misalignment
- `vertical_mass_fluctuation_align` — aligns opposite-angle pairs by vertical CoM

### Metrics package ([metrics/](metrics/))

| Module | Function |
|---|---|
| [metrics/sinogram_consistency.py](metrics/sinogram_consistency.py) | `sinogram_consistency_score` — row-to-row consistency in sinograms |
| [metrics/reprojection_consistency.py](metrics/reprojection_consistency.py) | `reprojection_consistency_score` — L2 between measured and re-projected projections |
| [metrics/fsc.py](metrics/fsc.py) | `fourier_shell_correlation` — splits data into half-sets and computes FSC resolution |
| [metrics/sharpness.py](metrics/sharpness.py) | `reconstruction_sharpness_score` — sharpness metric for reconstruction quality |

### Filters package ([filters/](filters/))

| Module | Function |
|---|---|
| [filters/kovacik.py](filters/kovacik.py) | `kovacik_filter` — post-reconstruction Fourier angular filter for missing-wedge artifacts |

### Helper utilities ([helperFunctions.py](helperFunctions.py))

- `subpixel_shift` — Fourier-domain subpixel shift (GPU-dispatched)
- `convert_to_tiff` / `convert_to_numpy` / `convert_to_2Dtiff` — TIFF I/O with scale metadata
- `DualLogger` — tees stdout to both console and log file
- `MoviePlotter` / `runwidget` — interactive projection/slice viewers for Jupyter and scripts
- `degree_to_positiveRadians` — angle unit conversion

### Data flow for a typical run

```
Load HDF5 → tomoData(projs, angles)
  → normalize(isPhaseData=True)
  → cross_correlate_align(...)           # updates workingProjections + tracked_shifts
  → make_updates_shift()                 # commits to finalProjections
  → [crop_center() if needed]
  → projection_matching_alignment(...)   # updates workingProjections + tracked_shifts
  → make_updates_shift()
  → reconstruct(algorithm='SIRT_CUDA')   # runs on finalProjections → self.recon
  → kovacik_filter()                     # refines self.recon in-place
  → convert_to_tiff(...)
```

### Reconstruction algorithms

`tomoData.reconstruct(algorithm)` dispatches on the string:
- `'SIRT_CUDA'` / `'*_CUDA'` — ASTRA GPU via `tomopy.astra`, 400 iterations
- `'svmbir'` — SVMBIR MBIR (high quality, slow)
- anything else — `tomopy.recon(algorithm=algorithm)` (CPU: `'sirt'`, `'art'`, `'gridrec'`, `'tv'`)

### Notebooks

Root-level (main workflows):
- [tomoMono_demo.ipynb](tomoMono_demo.ipynb) — end-to-end walkthrough with simulated phantom
- [test_notebook_realData.ipynb](test_notebook_realData.ipynb) — alignment method comparison on real APS beamtime data
- [ganrec_realData_walkthrough.ipynb](ganrec_realData_walkthrough.ipynb) — GANrec reconstruction walkthrough on real data
- [lookAtRecons.ipynb](lookAtRecons.ipynb) — reconstruction viewing/comparison
- [densityConversion.ipynb](densityConversion.ipynb) — mass density mapping from reconstructed volumes (histogram analysis, region segmentation, outputs massDensity*.tif)
- [debug_FSC_resolution.ipynb](debug_FSC_resolution.ipynb) — FSC resolution analysis and debugging for SIRT_CUDA_positivity reconstructions

Archived to [notebooks/](notebooks/) (reference/experimental):
- `notebooks/test_notebook_phanton.ipynb` — alignment method comparison on phantom data
- `notebooks/ganrec_phantom_walkthrough.ipynb` — GANrec on phantom data
- `notebooks/recon_algorithm_comparison.ipynb` — reconstruction algorithm benchmark
- `notebooks/FourierRingCorrelation.ipynb` — FRC resolution analysis

## Data Locations

- Raw experimental data: `/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/` (cluster, not in repo)
- Aligned projections (TIFF): `alignedProjections/`
- Reconstructions (TIFF): `reconstructions/`
- Small test phantoms (HDF5): `data/`
- Mass density outputs (TIFF): root directory (`massDensity*.tif`)

## HPC Notes

The cluster uses SLURM. GPU jobs need `--gpus=1` and enough RAM for the data at the chosen downsampling level (full-res Oct25 data requires ~160 GB; 4× downsample fits in ~12 GB/CPU). Compute directories are not backed up — only home directories are.
