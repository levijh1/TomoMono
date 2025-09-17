# TomoMono

**TomoMono** is a Python-based toolkit for tomographic data alignment, reconstruction, and analysis. It leverages the TomoPy and SVMBIR libraries to provide a flexible, scriptable workflow for both simulated and experimental tomography data, with a focus on materials science and imaging research.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Conda Environment Setup](#conda-environment-setup)
  - [Installing Dependencies](#installing-dependencies)
- [Repository Structure](#repository-structure)
- [Basic Usage](#basic-usage)
- [Alignment Strategies](#alignment-strategies)
- [Tomographic Reconstruction Algorithms](#tomographic-reconstruction-algorithms)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Flexible Alignment**: Multiple alignment strategies for correcting projection misalignments, including cross-correlation, projection matching, optical flow, and more.
- **3D Reconstruction**: Supports a variety of reconstruction algorithms, including GPU-accelerated and model-based iterative methods.
- **Simulated & Real Data**: Easily switch between simulated phantoms and experimental datasets.
- **Visualization**: Tools for visualizing projections and reconstructions interactively in Jupyter notebooks or scripts.
- **Batch Processing**: Scripts for automated alignment and reconstruction pipelines.

---

### Installing Dependencies

```bash
# Clone the repository
git clone https://github.com/levijh1/TomoMono.git
cd TomoMono

# Install dependencies (This will take a while)
# Will create an envrionment called tomoMono
conda env create -f environment.yml
```

---

## Repository Structure

- **main.py**: Main script for running 3D reconstructions on aligned data.
- **align.py**: Script for aligning projection data using various strategies.
- **tomoDataClass.py**: Defines the `tomoData` class, which manages data, alignment, and reconstruction.
- **alignment_methods.py**: Contains all alignment algorithms (see below).
- **helperFunctions.py**: Utility functions for plotting, shifting, and file I/O.
- **tomoMono_demo.ipynb**: Demo walking through the basics of tomoMono
- **notebooks/**: Example workflows and analysis (e.g., `tomoMono_demo.ipynb`, `densityConversion.ipynb`).
- **data/**: Directory for raw or simulated projection data.
- **alignedProjections/**: Stores aligned projection TIFF files.
- **logs/**: Output logs from script runs.
- **reconstructions/**: Output 3D reconstructions.

---

## Basic Usage

1. **Simulate or Load Data**  
   Use TomoPy to generate a phantom or load your own projection data.

2. **Alignment**  
   Use `align.py` or the `tomoData` class to align projections. Choose an alignment strategy (see below).

3. **Reconstruction**  
   Use `main.py` or the `tomoData.reconstruct()` method to reconstruct the 3D volume using your preferred algorithm.

4. **Visualization**  
   Use the provided plotting functions or Jupyter notebooks to visualize projections and reconstructions.

**Example (Jupyter Notebook):**
```python
from tomoDataClass import tomoData
import tomopy

# Simulate data
obj = tomopy.shepp3d(size=256)
angles = tomopy.angles(nang=200, ang1=0, ang2=360)
projections = tomopy.project(obj, angles, pad=False)

# Initialize tomoData object
tomo = tomoData(projections)

# Add jitter and noise
tomo.jitter(maxShift=7)
tomo.add_noise()

# Align and reconstruct
tomo.cross_correlate_align(tolerance=0.1, max_iterations=15)
tomo.PMA(max_iterations=5, tolerance=0.05, algorithm='art')
tomo.make_updates_shift()
tomo.reconstruct(algorithm='art')

# Visualize
tomo.makeNotebookReconMovie()
```

---

## Alignment Strategies

All alignment methods are implemented in `alignment_methods.py` and accessible via the `tomoData` class. The main strategies are:

- **cross_correlate_align**  
  Aligns projections by maximizing cross-correlation between consecutive images. Fast and robust for most datasets.

- **rotate_correlate_align**  
  Corrects rotational misalignments by maximizing cross-correlation after rotating projections.

- **PMA (Projection Matching Alignment)**  
  Iteratively aligns projections by comparing them to simulated projections from the current 3D reconstruction. Highly accurate, recommended as a final alignment step.

- **vertical_mass_fluctuation_align**  
  Aligns pairs of projections at opposite angles by minimizing vertical center-of-mass differences.

- **tomopy_align**  
  Uses TomoPy's joint reprojection algorithm for global alignment.

- **optical_flow_align**  
  Uses dense optical flow (TV-L1) to align projections. Useful for complex, non-rigid misalignments.

Each method can be called as a method of the `tomoData` object, e.g.:
```python
tomo.cross_correlate_align(tolerance=0.1, max_iterations=10)
tomo.PMA(max_iterations=5, tolerance=0.05, algorithm='art')
```

---

## Tomographic Reconstruction Algorithms

The following algorithms are available for 3D reconstruction (see `tomoData.reconstruct()`):

- **sirt**  
  Simultaneous Iterative Reconstruction Technique. CPU-based, robust, and widely used.

- **art**  
  Algebraic Reconstruction Technique. CPU-based, iterative, and good for sparse data.

- **tv**  
  Total Variation minimization. Useful for denoising and edge preservation.

- **gridrec**  
  Fast Fourier-based reconstruction. Very fast, but less robust to noise and misalignment.

- **SIRT_CUDA**  
  GPU-accelerated SIRT using the ASTRA toolbox. **Recommended for best speed and quality if you have a CUDA-capable GPU.**  
  *Note: Will only work if a compatible GPU is available.*

- **svmbir**  
  Model-Based Iterative Reconstruction (MBIR) via SVMBIR. Produces high-quality results, especially for noisy or incomplete data, but is slower than SIRT_CUDA.

**Algorithm Selection Tips:**
- Use **SIRT_CUDA** if you have a GPU—it's the fastest and often produces the best results.
- Use **svmbir** for the highest quality, especially with challenging data, but expect longer runtimes.
- Use **sirt**, **art**, or **gridrec** for quick CPU-based reconstructions or for testing.

**Example:**
```python
tomo.reconstruct(algorithm='SIRT_CUDA')  # Fast, high-quality (GPU required)
tomo.reconstruct(algorithm='svmbir')     # High-quality, slower (CPU or GPU)
tomo.reconstruct(algorithm='art')        # CPU-based, iterative
```

---

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request. For major changes, open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact the repository owner, Levi Hancock.

---
