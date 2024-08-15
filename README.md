# TomoMono

TomoMono is a Python-based tool designed for reconstructing and analyzing tomographic data. It leverages the capabilities of the TomoPy and SVMBIR libraries to perform advanced image processing and 3D reconstruction, specifically targeting applications in materials science and imaging studies.

## Features

- **2D and 3D Reconstruction**: Implements various algorithms like Filtered Back Projection (FBP), Algebraic Reconstruction Technique (ART), and Model-Based Iterative Reconstruction (MBIR) for accurate 3D reconstruction.
- **Data Alignment**: Provides tools for correcting misalignments in projection data using cross-correlation and other techniques.
- **Mass Density Calculation**: Computes electron and mass density based on phase shift measurements from X-ray ptychography data.

## Installation

To get started with TomoMono, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/levijh1/TomoMono.git
cd TomoMono
pip install -r requirements.txt
```

## Repository Structure
Python Scripts
- **main.py**: This is the primary script that orchestrates the entire tomographic reconstruction pipeline. It includes the loading of data, reconstruction using different algorithms, and saving of the results for further analysis. It implements various 3D reconstruction algorithms, including Filtered Back Projection (FBP), Algebraic Reconstruction Technique (ART), Simultaneous Iterative Reconstruction Technique (SIRT), and Model-Based Iterative Reconstruction (MBIR). The MBIR algorithm is noted to produce the best visual results and is preferred for final reconstructions.

- **align.py**: This script focuses on aligning the 2D ptychographic reconstructions using methods such as cross-correlation, joint projection matching, and optical flow alignment. Proper alignment is crucial for accurate 3D reconstruction.

Support python scripts
- **tomoDataClass.py**: Defines the tomoData class and contains all implementation of alignment and reconstruction

- **helperFunctions.py**: Contains several functions referenced within the code

- **pltwidget.py**: Simple script that contains the definition of a function to output a widget with a slider to look at a 3D array of projections

- **tiffConverter.py**: Contains functions for converting numpy arrays into tif files and tif files into numpy arrays (preserving the scale information in the original tif files)

Jupyter Notebooks
- **densityConversion.ipynb**: This notebook contains the workflow for calculating the mass density of the foam based on the reconstructed 3D volumes. It includes the conversion of phase shift values to electron density and the subsequent calculation of mass density.

- **Drop worst contrast images.ipynb**: Not a very important notebook. It was an experiment I tried to generate a dataset where the 'worst' 10 percent of projections based on contrast and feature sharpness were dropped from the dataset.

- **SeedingPtychographyRecons.ipynb**: Not a very important notebook. It was another experiment I tried to try creating a dataset that could seed the 2D reconstruction ptychography code written by Kevin Mertes (LANL), KMPty, to potentially create better reconstructions


Directories
- **data**: Must contain a tif file containing a 3D array of all of your data. The shape of the array should be like this [numAngles, imageHeight, imageWdith]
- **alignedProjections**: A tif array of all of the projections after the alignment process will be saved here with a timestamp in the fileName for everytime you run align.py with the 'save' boolean variable set to True.
- **logs**: If variable 'log' is set to True in either align.py or main.py, log files of the code output will be saved here
- **reconstructions**: All 3D reconstructions from main.py will be saved here if 'saveToFile' variable is set to True

## Dependencies
Ensure that the following Python libraries are installed:

numpy
scipy
tomopy
ASTRA
skimage
matplotlib
dragonfly
These dependencies can be installed using pip or conda.


##Usage 
The main functionality is demonstrated in the 'align.py' and 'main.py' python files. These files guide you through loading data, performing reconstructions, and visualizing results

##Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

##License
This project is licensed under the MIT License - see the LICENSE file for details.

##Contact
For any questions or inquiries, please contact the repository owner, Levi Hancock.
