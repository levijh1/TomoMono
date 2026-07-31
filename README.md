# TomoMono

**TomoMono** is a Python toolkit for tomographic alignment, reconstruction, and analysis. It wraps TomoPy, SVMBIR, and ASTRA into a single `tomoData` class that manages the full pipeline—from raw projections through alignment, reconstruction, and density analysis—with GPU acceleration throughout.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Key Files](#key-files)
- [Alignment Algorithms](#alignment-algorithms)
- [Reconstruction Algorithms](#reconstruction-algorithms)
- [Post-Reconstruction Filtering](#post-reconstruction-filtering)
- [Quality Metrics](#quality-metrics)
- [Notebooks](#notebooks)
- [Running on the HPC Cluster](#running-on-the-hpc-cluster)
- [Contact](#contact)

---

## Quick Start

```python
from tomoDataClass import tomoData
import tomopy

# Simulate a phantom dataset
obj    = tomopy.shepp3d(size=128)
angles = tomopy.angles(nang=180, ang1=0, ang2=360)
projs  = tomopy.project(obj, angles, pad=False)

# Create a tomoData object and add realistic misalignment
tomo = tomoData(projs, angles)
tomo.jitter(maxShift=5)

# Align — coarse-to-fine
tomo.cross_correlate_align(max_iterations=10)
tomo.projection_matching_alignment(iterations_per_level=[5], algorithm='SIRT_CUDA')
tomo.make_updates_shift()   # commit shifts (avoids accumulated interpolation error)

# Reconstruct and view
tomo.reconstruct(algorithm='SIRT_CUDA')
tomo.makeNotebookReconMovie()
```

See [tomoMono_demo.ipynb](tomoMono_demo.ipynb) for a full interactive walkthrough.

---

## Installation

```bash
git clone https://github.com/levijh1/TomoMono.git
cd TomoMono

# Create the conda environment (takes a few minutes)
conda env create -f environment.yml
conda activate tomoMono
```

Key dependencies: Python 3.12, TomoPy, SVMBIR, ASTRA Toolbox 2.1 (CUDA), CuPy, PyTorch 2.4, scikit-image, scipy, numpy 1.26, tifffile, h5py.

---

## Repository Structure

```
TomoMono/
├── tomoDataClass.py        # Core class — all state, alignment, and reconstruction
├── helperFunctions.py      # Utilities: file I/O, subpixel shift, visualization
├── gpu.py                  # GPU detection; exports xp (CuPy or NumPy), torch, svmbir
├── align.py                # Script: staged 4x → 2x → 1x alignment pipeline
├── main.py                 # Script: one reconstruction from pre-aligned projections
├── recon_param_search.py   # Script: compare SIRT/ART/FBP/gridrec/tv side by side
├── runGPUAlign.sh          # SLURM submission for align.py
├── runTomopyParamSearch.sh # SLURM submission for recon_param_search.py
│
├── alignment/              # Alignment algorithms (import from here)
│   ├── cross_correlate.py  #   Cross-correlation alignment (XCA)
│   ├── pma.py              #   Projection Matching Alignment (PMA)
│   ├── vmf.py              #   Vertical Mass Fluctuation alignment
│   └── legacy.py           #   Optical flow, rotation correction, tomopy_align
│
├── metrics/                # Alignment & reconstruction quality metrics
│   ├── fsc.py              #   Fourier Shell Correlation — best reconstruction-quality metric
│   ├── reprojection_consistency.py  # RCS — best alignment-quality metric
│   └── sinogram_consistency.py      # Helgason-Ludwig CoM check — rough alignment gauge
│
├── filters/
│   └── kovacik.py          # Post-reconstruction Fourier angular filter
│
├── tomoMono_demo.ipynb           # Interactive demo on a phantom — start here
├── weddingCake_roughDraft.ipynb  # Same pipeline on real measured phase data
├── densityConversion.ipynb       # Convert a reconstructed volume to mass density
├── notebooks/              # Reference / experimental notebooks (FSC, XRF, filters, ...)
│
├── data/                   # Small phantoms and angle files
├── alignedProjections/     # Aligned projection TIFFs (output of align.py)
├── reconstructions/        # Reconstructed 3D volumes (TIFFs)
├── massDensity/            # Mass density maps (output of densityConversion.ipynb)
├── figures/                # Plots and paper figures
├── XRF_Data/               # XRF scans used by notebooks/projectXRFonVolume.ipynb
├── hyperparam_results/     # CSV results from parameter searches
├── logs/                   # Script run logs
├── sbatch_output/          # SLURM stdout/stderr
└── Archive/                # Retired code: GANrec, SVMBIR scripts, alignment param search
```

Data products (`*.tif`, `data/`, `figures/`, `reconstructions/`, `alignedProjections/`, `logs/`,
`hyperparam_results/`, `sbatch_output/`) are gitignored — only code and notebooks are tracked.

---

## Key Files

### [tomoDataClass.py](tomoDataClass.py) — the core class

All state lives in a `tomoData` instance. The main internal buffers are:

| Attribute | Description |
|---|---|
| `data` | Original raw projections — never modified after `jitter()` |
| `workingProjections` | Scratch copy updated by alignment methods |
| `finalProjections` | Accumulates committed shifts; used for reconstruction |
| `tracked_shifts` | Per-projection (y, x) shift accumulator |
| `ang` | Projection angles, in radians |
| `rotation_center` | Rotation-axis position found by `center_projections()` / `reconstruct()` |
| `recon` | 3D volume after `reconstruct()` |

**The two-buffer pattern**: alignment methods write shifts to `workingProjections` and accumulate them in `tracked_shifts`. Calling `make_updates_shift()` applies all accumulated shifts to `finalProjections` in a single subpixel-interpolation pass — this avoids stacking interpolation error across multiple alignment rounds. `reconstruct()` always runs on `finalProjections`.

Typical method sequence:

```python
tomo = tomoData(projections, angles)
tomo.normalize(isPhaseData=True)        # or isPhaseData=False for absorption
tomo.cross_correlate_align(...)
tomo.center_projections()
tomo.projection_matching_alignment(...)
tomo.make_updates_shift()               # commit before reconstructing
tomo.reconstruct(algorithm='SIRT_CUDA')
```

Useful extras on the class: `crop(new_y, new_x, anchor='center'|'bottom')`,
`reset_workingProjections()`, `jitter()` / `add_noise()` for simulating misalignment,
`shift_envelope` (how many edge pixels the accumulated shifts have exposed),
`simulateProjections()` (ASTRA GPU forward projection with a tomopy fallback), and the
`makeNotebookProjMovie()` / `makeNotebookReconMovie()` / `displayReconOrthogonalSlices()` viewers.

---

### [gpu.py](gpu.py) and [helperFunctions.py](helperFunctions.py) — support modules

`gpu.py` probes for a working GPU once at import time and exports the backends everything else
uses: `xp` (CuPy or NumPy), `cp`, `torch`, `svmbir`, GPU-aware drop-ins for
`scipy.ndimage.shift` / `gaussian_filter` / `fourier_shift`, and `to_numpy()`. Other modules import
from here rather than running their own try/except ladders, so the code runs unchanged on a laptop
without a GPU.

`helperFunctions.py` holds the shared utilities: `subpixel_shift` (Fourier-domain subpixel shift),
`convert_to_numpy` / `convert_to_tiff` / `convert_to_2Dtiff` (TIFF I/O carrying scale metadata),
`DualLogger` (tees stdout to a log file), `MoviePlotter` / `runwidget` (interactive viewers), and
`degree_to_positiveRadians`.

---

### [align.py](align.py) — staged alignment pipeline

The production alignment script for real experimental data. It runs a **three-stage coarse-to-fine pipeline**:

1. **Stage 1 — 4× downsampled**: Full XCA (multiple passes with decreasing downsampling) + PMA to get a rough alignment quickly.
2. **Stage 2 — 2× downsampled**: Seeds from the scaled-up 4× shifts, then refines with PMA.
3. **Stage 3 — Full resolution**: Seeds from scaled-up 2× shifts, then one final PMA pass.

At each stage, aligned projections, a mid-stack sinogram, and the cumulative shifts are saved under `alignedProjections/`, and a reconstruction is optionally saved to `reconstructions/`. Quality metrics (RCS and FSC resolution) are printed for each stage and summarized at the end so the stages can be compared.

Run it directly or via the SLURM cluster:

```bash
python align.py                 # interactive
sbatch runGPUAlign.sh           # cluster (6 h, 1 GPU, ~400 GB RAM)
```

`align.py` takes no command-line arguments — the input HDF5 path, dropped angle indices, output directories, and per-stage alignment parameters are edited inside the script.

---

### [main.py](main.py) — standalone reconstruction

Takes an already-aligned projection TIFF (output of `align.py`) and runs a single reconstruction. Edit the configuration block at the top:

```bash
python main.py
```

Configuration variables in the file:
- `TIFF_FILE` — path to aligned projections
- `OUTPUT_DIR` — where to save the result
- `ALGORITHM` — e.g. `'SIRT_CUDA'`, `'gridrec'`, `'svmbir'`
- `NUM_ITER` — iteration count (relevant for iterative algorithms)
- `RAW_HDF5` — raw file the acquisition angles are read from
- `DROP_ANGLES` — list of projection indices to exclude (bad angles)

---

### [recon_param_search.py](recon_param_search.py) — algorithm and hyperparameter comparison

Reconstructs the same pre-aligned projections many times over and scores each result, so the best settings for a dataset can be picked by number rather than by eye. Two sweeps run back to back:

1. **Algorithm comparison** — `SIRT_CUDA`, `ART_CUDA`, `FBP_CUDA`, `gridrec`, `tv`.
2. **SIRT_CUDA hyperparameters** — 100 / 200 / 400 / 600 iterations, plus a positivity-constrained run (`MinConstraint=0`).

Every configuration is scored with RCS and FSC and saved as a TIFF plus an orthogonal-slice PNG. The projections can be cropped in y and to a centered width so a sweep runs on a manageable subvolume.

```bash
python recon_param_search.py --tiff-file alignedProjections/.../yourfile.tif \
    --y-start 40 --y-end 440 --width 1200
sbatch runTomopyParamSearch.sh  # cluster (6 h, 1 GPU, ~500 GB RAM)
```

---

### Importing alignment, metrics, and filters

Alignment algorithms, quality metrics, and post-reconstruction filters live in
the `alignment/`, `metrics/`, and `filters/` subpackages. Import them directly
from the subpackage that owns them:

```python
from alignment import cross_correlate_align, projection_matching_alignment
from metrics import fourier_shell_correlation, reprojection_consistency_score
from filters import kovacik_filter
```

Each of these is a free function whose first argument is a `tomoData` object, and each is also
auto-attached as a method on the class. So these two calls are equivalent:

```python
cross_correlate_align(tomo, max_iterations=10)   # free function
tomo.cross_correlate_align(max_iterations=10)    # attached method
```

To add a new alignment method, metric, or filter: write it as a free function taking `tomo` first,
export it from the subpackage's `__init__.py`, and add it to the delegate list at the bottom of
[tomoDataClass.py](tomoDataClass.py).

---

### Archived scripts

[Archive/](Archive/) holds code that is no longer part of the main pipeline but is kept for
reference: `Archive/svmbir/` (SVMBIR reconstruction and parameter-search scripts),
`Archive/ganrec/` (GANrec scripts, walkthrough notebooks, and a vendored `ganrec_pkg`),
`Archive/hyperparameter_search.py` (XCA/PMA alignment config search), and older aligned/recon
TIFFs. Paths and imports in these scripts predate the current package layout and need checking
before reuse.

---

## Alignment Algorithms

All methods are accessible as methods on the `tomoData` object. The recommended approach is to use **cross-correlation first** (fast, global), then **PMA** (slow, accurate) as a final refinement.

### Cross-Correlation Alignment (`cross_correlate_align`)

Aligns projections sequentially by maximizing the phase cross-correlation between adjacent projections. Key options:

| Parameter | Effect |
|---|---|
| `downsample` | Run at reduced resolution for speed (e.g. `4` = 4× smaller) |
| `use_grad` | Operate on gradient images — makes features sharper and improves XC peak quality |
| `yROI_Range`, `xROI_Range` | Restrict the correlation to a region of interest to avoid edge artifacts |
| `max_iterations` | How many passes to run |
| `stepRatio` | Fraction of computed shift to apply per step (< 1.0 dampens oscillation) |

**When to use**: Always run this first. It is fast and handles large global offsets well. Run multiple passes at decreasing `downsample` values for best results.

```python
tomo.cross_correlate_align(
    max_iterations=10, downsample=4, use_grad=True
)
tomo.cross_correlate_align(
    max_iterations=10, downsample=1, use_grad=True,
    yROI_Range=[0, tomo.workingProjections.shape[1] - 50]
)
```

---

### Projection Matching Alignment (`projection_matching_alignment` / `PMA`)

The most accurate alignment method. Each iteration:

1. Reconstructs the 3D volume from current projections
2. Forward-projects the reconstruction at each angle to produce simulated projections
3. Measures the shift between each real projection and its simulated counterpart
4. Applies the correction

Key options:

| Parameter | Effect |
|---|---|
| `iterations_per_level` | List of iteration counts per multi-scale level (e.g. `[10, 5]`) |
| `algorithm` | Algorithm used for the internal reconstruction step (e.g. `'SIRT_CUDA'`) |
| `shift_method` | `'cross_correlation'` or `'optical_flow'` — see note below |
| `levels` | Number of multi-scale levels |
| `scale` | Downsampling factor between levels — `scale=2` with `levels=2` runs the coarse level at 2× reduced resolution, capturing large shifts cheaply before refining at full resolution |
| `xROI_Range`, `yROI_Range` | Limit the region used for shift measurement |
| `stepRatio` | Damping factor per step |

> **`shift_method='optical_flow'` inside PMA** uses a Lucas-Kanade formulation to solve for a single global translation (dy, dx) per projection — the image is still shifted rigidly, exactly like `'cross_correlation'`. It is *not* the same as standalone `optical_flow_align` (see below), which computes a per-pixel displacement field and warps/deforms the image non-rigidly.

**When to use**: After cross-correlation alignment converges. PMA is expensive (one full reconstruction per iteration) but achieves sub-pixel accuracy. Always call `make_updates_shift()` after PMA before reconstructing.

```python
tomo.projection_matching_alignment(
    levels=2, scale=2, iterations_per_level=[10, 5],
    algorithm='SIRT_CUDA',
    shift_method='cross_correlation'
)
tomo.make_updates_shift()
```

---

### Vertical Mass Fluctuation Alignment (`vertical_mass_fluctuation_align`)

Aligns projection pairs at opposite angles (e.g. 0° and 180°) by matching their vertical center-of-mass profiles. Useful for correcting vertical drift that cross-correlation misses.

```python
tomo.vertical_mass_fluctuation_align()
```


---

### Optical Flow Alignment (`optical_flow_align`)

Uses scikit-image's dense TV-L1 optical flow to compute a per-pixel displacement field between adjacent projections, then applies `warp()` to deform each image non-rigidly. This modifies the actual pixel layout of the projections — it is not a simple translation.

> **Do not confuse with `shift_method='optical_flow'` in PMA.** That option uses a Lucas-Kanade formulation to estimate a single global (dy, dx) translation per projection and applies it as a rigid subpixel shift. The word "optical flow" in PMA refers to the *measurement technique*, not a non-rigid warp. This standalone `optical_flow_align` is a structurally different operation: it deforms the image itself.

This function is in `legacy.py` and is **not recommended for routine use** — it does not update `tracked_shifts`, so the deformation cannot be composed with other alignment steps. Useful only for exploratory or diagnostic purposes.

---

## Reconstruction Algorithms

`tomo.reconstruct(algorithm='...')` dispatches on a string.

### SIRT_CUDA ⭐ recommended for GPU systems

**Simultaneous Iterative Reconstruction Technique** run on the GPU via the ASTRA toolbox. Iteratively updates the volume to minimize the L2 difference between measured and re-projected sinograms.

- 400 iterations by default
- Excellent balance of speed and quality
- Handles noisy data and slight misalignment well
- **Requires a CUDA-capable GPU**

```python
tomo.reconstruct(algorithm='SIRT_CUDA', num_iter=400)
```

Other ASTRA GPU variants: `'ART_CUDA'` (algebraic, faster convergence on sparse data) and `'FBP_CUDA'` (filtered back-projection, fastest, least artifact-suppression).

---

### SVMBIR ⭐ recommended for highest quality

**Model-Based Iterative Reconstruction** using a qGGMRF prior. Produces the sharpest, lowest-noise reconstructions, especially for incomplete or noisy data. CPU-only but highly parallelized.

- Much slower than SIRT_CUDA (hours vs. minutes for full-resolution data)
- Best for final high-quality reconstructions after alignment is done
- Production run scripts live in [Archive/svmbir/](Archive/svmbir/) (`runSVMBIRrec.py`, `runSVMBIRparamSearch.py`)

```python
tomo.reconstruct(algorithm='svmbir')
```

---

### gridrec — fast CPU reconstruction (no GPU required)

Fourier-based filtered back-projection. Runs in seconds on CPU. Good for quick sanity checks or when no GPU is available.

```python
tomo.reconstruct(algorithm='gridrec')
```

---

### Other CPU algorithms

| Algorithm | Description |
|---|---|
| `'sirt'` | CPU SIRT — same math as SIRT_CUDA but slower |
| `'art'` | Algebraic Reconstruction Technique — good for sparse angular sampling |
| `'tv'` | Total Variation minimization — strong noise suppression, can over-smooth |
| `'fbp'` | Filtered Back-Projection — fastest CPU option, equivalent to gridrec |

---

### Algorithm Selection Guide

| Situation | Recommendation |
|---|---|
| GPU available, want speed + quality | `SIRT_CUDA` |
| Need the absolute best reconstruction | `svmbir` |
| No GPU, quick check | `gridrec` or `fbp` |
| Sparse angles or high noise | `svmbir` or `tv` |
| Inside PMA alignment loop | `SIRT_CUDA` (fast iterations matter) |

Note that `GRIDREC_CUDA` and `TV_CUDA` do not exist — ASTRA and TomoPy provide gridrec and TV on
the CPU only.

---

## Post-Reconstruction Filtering

### Kovacik filter (`kovacik_filter`)

A soft Fourier angular filter (Kovacik et al. 2014, *J. Struct. Biol.* **186**, 141-152) that
suppresses missing-wedge ray artifacts. It smooths the sharp Fourier-space transition between the
sampled region and the missing wedge — Butterworth ramps adjacent to the highest-tilt projection
boundaries, combined with a central stripe that protects low spatial frequencies from attenuation.

It edits `tomo.recon` in place, caching the unfiltered volume as `tomo._recon_pre_kovacik` so it
can be re-run with different parameters without reconstructing again.

```python
tomo.reconstruct(algorithm='SIRT_CUDA')
tomo.kovacik_filter(plot=True)     # tilt_max defaults to the largest angle in tomo.ang
```

---

## Quality Metrics

The `metrics/` package provides ways to quantify how good an alignment or a
reconstruction is. They are imported from `metrics` and most are also attached
as methods on the `tomoData` object.

```python
from metrics import (
    fourier_shell_correlation,
    reprojection_consistency_score,
    sinogram_consistency_score,
)
```

| Metric | Measures | Reliability |
|---|---|---|
| `fourier_shell_correlation` (FSC) | **Reconstruction quality / resolution** | ⭐ Best reconstruction-quality metric |
| `reprojection_consistency_score` (RCS) | **Alignment quality** | ⭐ Best alignment-quality metric |
| `sinogram_consistency_score` | Alignment (CoM consistency) | Rough gauge — use as a sanity check |

### Fourier Shell Correlation — best for reconstruction quality

FSC splits the tilt series into two independent half-sets, reconstructs each,
and compares the two volumes shell-by-shell in 3D Fourier space. The frequency
at which they stop agreeing is a genuine, noise-independent resolution estimate
(reported in pixels, or nm if a pixel size is given). **This is the most
trustworthy way to judge how good a reconstruction is.**

```python
tomo.fourier_shell_correlation(algorithm='SIRT_CUDA', pixel_size_nm=10.0)
```

### Reprojection Consistency Score — best for alignment quality

RCS reprojects the reconstructed volume at every angle and measures the
per-angle NRMSE against the measured projections. Well-aligned projections are
mutually consistent, so the reconstruction reprojects back to match them closely
and the score drops. **This is the most reliable metric for testing how good an
alignment is**, and the per-angle bar chart pinpoints which projections are
still misaligned (outliers).

```python
tomo.reprojection_consistency_score(plot=True)   # lower is better; < 0.10 is excellent
```

### Sinogram Consistency Score — rough gauge / sanity check

This checks the Helgason-Ludwig center-of-mass consistency conditions on the
sinogram. It is **not a super-reliable score** — the conditions are only an
approximate proxy and are sensitive to background, asymmetric objects, and the
missing wedge. Use it to get a coarse sense of how close the alignment is, to
find outlier projections, and to visualize the central-slice sinogram.

```python
tomo.sinogram_consistency_score(plot=True)
```

---

## Notebooks

Three notebooks at the root are the active workflows; everything in [notebooks/](notebooks/) is
reference or experimental material kept for future use.

| Notebook | Purpose |
|---|---|
| [tomoMono_demo.ipynb](tomoMono_demo.ipynb) | End-to-end pipeline on a simulated phantom — **start here** |
| [weddingCake_roughDraft.ipynb](weddingCake_roughDraft.ipynb) | The same sequence run on real measured phase projections (TPP foam "wedding cake" sample) |
| [densityConversion.ipynb](densityConversion.ipynb) | Scale a reconstruction and convert voxel intensity to physical mass density |

Notable notebooks in [notebooks/](notebooks/): `debug_FSC_resolution.ipynb` (FSC resolution
analysis), `lookAtRecons.ipynb` (browse and compare reconstructions), `featureSizeFFT.ipynb`
(feature size from the 3D radial power spectrum), `projectXRFonVolume.ipynb` (project XRF maps
onto aligned projections), `recon_algorithm_comparison.ipynb`, and `test_notebook_phanton.ipynb`
(alignment method comparison on a phantom). They import the root package via a `sys.path` preamble,
so they run from inside the `notebooks/` directory.

### [densityConversion.ipynb](densityConversion.ipynb) — mass density analysis

Takes a reconstructed volume (TIFF) and converts the voxel intensity values to physical mass density:

1. Loads a reconstruction TIFF and computes its scale information
2. Plots intensity histograms to identify material phases
3. Segments the volume by region (whole volume, thresholded sample, interior prism)
4. Converts intensities to mass density using the phase-shift calibration
5. Saves mass density maps to `massDensity/` and histogram figures to `figures/`

### [tomoMono_demo.ipynb](tomoMono_demo.ipynb) — the walkthrough

It walks through the entire pipeline on a simulated Shepp-Logan phantom so you can see every step without needing experimental data:

1. **Generate a phantom** — uses TomoPy to create a 3D phantom and simulate projections at many angles
2. **Add jitter and noise** — simulates realistic projection misalignment and detector noise
3. **Cross-correlation alignment** — aligns the jittered projections and shows the convergence
4. **Projection Matching Alignment** — refines alignment using the reconstructed volume
5. **`make_updates_shift()`** — explains the two-buffer pattern and why it matters
6. **Reconstruct** — runs SIRT_CUDA (or falls back to CPU gridrec) and displays the result
7. **Kovacik filter** — applies the post-reconstruction angular filter and shows the improvement
8. **Visualization** — interactive slice viewers and projection movies

Each cell prints metrics (shift magnitudes, reprojection consistency score) so you can see quantitatively how much each step improves alignment quality.

---

## Running on the HPC Cluster

This project runs on the BYU HPC cluster (SLURM). The `tomoMono` conda environment is pre-installed at `/home/ljh79/.conda/envs/tomoMono/`.

| Task | Command | Resources requested |
|---|---|---|
| Staged alignment (GPU) | `sbatch runGPUAlign.sh` | 6 h, 1 GPU, 4 CPUs × 100 GB |
| Algorithm / hyperparameter search | `sbatch runTomopyParamSearch.sh` | 6 h, 1 GPU, 4 CPUs × 125 GB |

Full-resolution Oct 2025 data needs roughly 160 GB in memory, which is why the requests are so
large; a 4× downsampled run fits in about 12 GB per CPU.

Logs go to `logs/`, SLURM stdout/stderr goes to `sbatch_output/<jobname>/` — create that
subdirectory before submitting if it does not exist. Compute directories on the cluster are not
backed up; only home directories are.

---

## Contact

Questions or feedback? Contact Levi Hancock (levijh1@gmail.com) or open an issue on GitHub.
