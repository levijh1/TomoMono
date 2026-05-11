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

Key dependencies: Python 3.12, TomoPy, SVMBIR, ASTRA Toolbox, CuPy (CUDA 11.8), PyTorch (cu118), OpenCV, scikit-image, scipy, numpy 1.26, tifffile, h5py.

## Running Scripts

**Alignment pipeline (GPU-intensive):**
```bash
python align.py                          # interactive / local run
sbatch runGPUAlign.sh                    # submit to SLURM cluster (48h, 1 GPU, 200 GB RAM)
```

**3D reconstruction from pre-aligned projections:**
```bash
python main.py
```

**Hyperparameter search over XCA + PMA alignment configs:**
```bash
python hyperparameter_search.py
python hyperparameter_search.py --resume --logfile hyperparam_results/<name>.csv
sbatch runHyperparamSearch.sh            # cluster submission
```

Logs go to `logs/`, SLURM stdout/err goes to `sbatch_output/`, hyperparam results go to `hyperparam_results/`.

## Architecture

The codebase is a single-package Python library; there are no submodules or imports from subdirectories.

### Core class: `tomoData` ([tomoDataClass.py](tomoDataClass.py))

All state lives in a `tomoData` instance:
- `data` — original raw projections (never modified after `jitter()`)
- `workingProjections` — scratch copy used during iterative alignment
- `finalProjections` — accumulates committed shifts for final reconstruction
- `tracked_shifts` — per-projection (y, x) shift accumulator, reset by `make_updates_shift()`
- `recon` — 3D volume after `reconstruct()`; also cached as `_recon_pre_kovacik` before `kovacik_filter()`

**Two-buffer alignment pattern**: alignment methods update `workingProjections` and accumulate into `tracked_shifts`. Call `make_updates_shift()` to commit those shifts to `finalProjections` via a single subpixel interpolation pass (avoids stacking interpolation error). `reconstruct()` always runs on `finalProjections`.

GPU acceleration is transparent: CuPy is used for array ops if available, otherwise falls back to NumPy/SciPy. ASTRA is used for GPU forward projection inside `simulate_projections()` with a tomopy fallback.

### Alignment methods ([alignment_methods.py](alignment_methods.py))

Standalone functions that take a `tomoData` as first argument (the class just delegates to them):

| Function | What it does |
|---|---|
| `cross_correlate_align` | Sequential cross-correlation between adjacent projections; supports ROI, downsampling pyramid, gradient mode, rolling median reference |
| `PMA` | Projection Matching Alignment — iterates: reconstruct → forward-project → measure shift (phase-XC or optical flow) → apply; multi-scale via `levels` param |
| `rotate_correlate_align` | Corrects rotational misalignment |
| `vertical_mass_fluctuation_align` | Aligns opposite-angle pairs by vertical CoM |
| `tomopy_align` | Wraps TomoPy joint reprojection |
| `optical_flow_align` | Dense TV-L1 optical flow |
| `sinogram_consistency_score` | Quality metric: measures row-to-row consistency in sinograms |
| `reprojection_consistency_score` | Quality metric (RCS): L2 between measured and re-projected projections |
| `fourier_shell_correlation` | Resolution metric: splits data into two half-sets and computes FSC |
| `kovacik_filter` | Post-reconstruction Fourier angular filter for missing-wedge artifacts (called on `tomoData`, lives in `tomoDataClass.py`) |

### Helper utilities ([helperFunctions.py](helperFunctions.py))

- `subpixel_shift` — Fourier-domain subpixel shift (GPU-dispatched)
- `convert_to_tiff` / `convert_to_numpy` — TIFF I/O with scale metadata
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
  → PMA(...)                             # updates workingProjections + tracked_shifts
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

- [tomoMono_demo.ipynb](tomoMono_demo.ipynb) — end-to-end walkthrough with simulated phantom
- [test_taylor_alignment_methods_phanton.ipynb](test_taylor_alignment_methods_phanton.ipynb) — alignment method comparison on phantom data
- [test_taylor_alignment_methods_realData.ipynb](test_taylor_alignment_methods_realData.ipynb) — same comparison on real APS beamtime data
- [recon_algorithm_comparison.ipynb](recon_algorithm_comparison.ipynb) — reconstruction algorithm benchmark

## Data Locations

- Raw experimental data: `/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/` (cluster, not in repo)
- Aligned projections (TIFF): `alignedProjections/`
- Reconstructions (TIFF): `reconstructions/`
- Small test phantoms (HDF5): `data/`

## HPC Notes

The cluster uses SLURM. GPU jobs need `--gpus=1` and enough RAM for the data at the chosen downsampling level (full-res Oct25 data requires ~160 GB; 4× downsample fits in ~12 GB/CPU). Compute directories are not backed up — only home directories are.
