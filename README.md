# TomoMono

TomoMono is a Python-based tool designed for reconstructing and analyzing tomographic data. It leverages the capabilities of the TomoPy and SVMBIR libraries to perform advanced image processing and 3D reconstruction, specifically targeting applications in materials science and imaging studies.

## Features

- **2D and 3D Reconstruction**: Implements various algorithms like Filtered Back Projection (FBP), Algebraic Reconstruction Technique (ART), and Model-Based Iterative Reconstruction (MBIR) for accurate 3D reconstruction.
- **Data Alignment**: Provides tools for correcting misalignments in projection data using cross-correlation and other techniques.
- **Visualization**: Includes visualization utilities using Dragonfly scientific image processing software for analyzing reconstructed volumes.
- **Mass Density Calculation**: Computes electron and mass density based on phase shift measurements from X-ray ptychography data.

## Installation

To get started with TomoMono, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/levijh1/TomoMono.git
cd TomoMono
pip install -r requirements.txt
