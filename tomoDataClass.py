import tomopy
import numpy as np
import random
import scipy as sp
from helperFunctions import MoviePlotter, subpixel_shift, runwidget
# import torch
from tqdm import tqdm
from scipy.ndimage import rotate
# import svmbir
from alignment_methods import *

class tomoData:
    """
    Class for handling tomographic data, including preprocessing, alignment, and reconstruction.
    """

    def __init__(self, data):
        """
        Initializes the TomoData object with the provided dataset.
        Make sure that you check that the ang variable has the same range and number of angles as the collected data.
        This assumes that all angles are evenly spaced between 0 and 360 degrees.
        Parameters:
        - data (np.array): The tomographic data as a 3D numpy array. The first dimension is the projection number.
        """
        self.num_angles = data.shape[0]
        self.image_size = data.shape[1:]
        self.data = data
        self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=360)
        self.workingProjections = np.copy(data)
        self.rotation_center = 0
        self.center_offset = 0
        self.finalProjections = np.copy(data)
        self.tracked_shifts = np.zeros((self.num_angles, 2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def reset_workingProjections(self, x_size=900, y_size=650):
        """
        Resets working and final projections to the original data and crops to the specified center size.
        This is useful for starting fresh with the original projections after modifications back to back.

        Parameters:
        - x_size (int): Width of the crop.
        - y_size (int): Height of the crop.
        """
        self.workingProjections = np.copy(self.data)
        self.finalProjections = np.copy(self.data)
        self.crop_center(x_size, y_size)

    def get_recon(self):
        """
        Returns the reconstructed 3D model.

        Returns:
        - np.array: The reconstructed volume.
        """
        return self.recon

    def get_workingprojections(self):
        """
        Returns the current state of working projections (The projections that are being modified).

        Returns:
        - np.array: The working projections.
        """
        return self.workingProjections

    def get_finalProjections(self):
        """
        Returns the final projections (the projections that are modified last and saved for reconstruction).

        Returns:
        - np.array: The final projections.
        """
        return self.finalProjections

    def jitter(self, maxShift=5):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        Jitter ranges from -maxShift to +maxShift pixels in both x and y directions. Affects data variable as well.

        Parameters:
        - maxShift (float): Maximum shift in pixels for both x and y directions.
        """
        multiplier = maxShift * 2
        for i in range(1, self.num_angles-1):
            x_shift = multiplier * (random.random() - 0.5)
            y_shift = multiplier * (random.random() - 0.5)
            # self.data[i] = sp.ndimage.shift(self.data[i], (x_shift, y_shift), mode="constant")
            self.data[i] = subpixel_shift(self.data[i], y_shift, x_shift)
            self.data[i] = self.data[i] * (self.data[i] > 0)

        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()

    def add_noise(self):
        self.data = tomopy.prep.alignment.add_noise(self.data)
        self.workingProjections = self.data.copy()
        self.finalProjections = self.data.copy()   

    def crop_center(self, new_x, new_y):
        """
        Crops each 2D array in the 3D array to a specified size centered in the middle.

        Parameters:
        - new_x (int): Target width of the crop.
        - new_y (int): Target height of the crop.
        """
        y, x = self.workingProjections[0].shape
        startx = x // 2 - new_x // 2
        endx = startx + new_x
        starty = y // 2 - new_y // 2
        endy = starty + new_y
        startx, endx = max(0, startx), min(x, endx)
        starty, endy = max(0, starty), min(y, endy)
        self.workingProjections = self.workingProjections[:,starty:endy, startx:endx]
        self.finalProjections = self.finalProjections[:,starty:endy, startx:endx]
        self.image_size = self.workingProjections.shape[1:]

    # def crop_bottom_center(self, new_y, new_x):
    #     """
    #     Crops the center portion of the image without cutting off anything from the bottom.

    #     Parameters:
    #     - new_y (int): Target height of the crop.
    #     - new_x (int): Target width of the crop.
    #     """
    #     cropped_array = np.zeros((self.workingProjections.shape[0], new_y, new_x), dtype=self.workingProjections.dtype)
    #     print(f"Cropping projections to size: {new_x}x{new_y}")
    #     for i, array in enumerate(self.workingProjections):
    #         y, x = array.shape
    #         startx = x // 2 - new_x // 2
    #         endx = startx + new_x
    #         starty = y - new_y
    #         endy = y
    #         startx, endx = max(0, startx), min(x, endx)
    #         starty, endy = max(0, starty), min(y, endy)
    #         cropped_array[i] = array[starty:endy, startx:endx]
    #     self.workingProjections = cropped_array
    #     self.finalProjections = cropped_array
    #     self.image_size = self.workingProjections.shape[1:]

    def crop_bottom_center(self, new_y, new_x):
        """
        Crops each 2D array in the 3D array to a specified size, aligned to the bottom and centered horizontally.

        Parameters:
        - new_y (int): Target height of the crop.
        - new_x (int): Target width of the crop.
        """
        y, x = self.workingProjections[0].shape
        startx = x // 2 - new_x // 2
        endx = startx + new_x
        starty = y - new_y
        endy = y
        startx, endx = max(0, startx), min(x, endx)
        starty, endy = max(0, starty), min(y, endy)
        self.workingProjections = self.workingProjections[:, starty:endy, startx:endx]
        self.finalProjections = self.finalProjections[:, starty:endy, startx:endx]
        self.image_size = self.workingProjections.shape[1:]


    def track_shifts(self):
        """
        Resets tracked shifts and rotations, and sets final projections to the current working projections.
        Allows you to track all changes to workingProjections so that you can apply the final changes later to finalProjections.
        """
        self.finalProjections = self.workingProjections.copy()
        self.tracked_shifts = np.zeros((self.num_angles, 2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def make_updates_shift(self):
        """
        Applies tracked subpixel shifts from workingProjections to the finalProjections and resets tracked shifts.
        This is done so that the finalProjections can be updated just once and not lose any information.
        """
        for m in tqdm(range(self.num_angles), desc='Apply shifts to final projections'):
            self.finalProjections[m] = subpixel_shift(self.finalProjections[m], self.tracked_shifts[m, 0], self.tracked_shifts[m, 1])
        self.tracked_shifts = np.zeros((self.num_angles, 2))

    def make_updates_rotate(self):
        """
        Applies tracked rotations to the final projections and resets tracked shifts.
        """
        for m in tqdm(range(self.num_angles), desc='Apply rotations to final projections'):
            self.finalProjections[m] = rotate(self.finalProjections[m], self.tracked_rotations[m], reshape=False, mode='constant')
        self.tracked_shifts = np.zeros((self.num_angles, 2))

    def normalize(self, isPhaseData):
        """
        Normalizes all projections to be positive values between 0 and 1.
        """
        print("\n")
        print("Normalizing projections")
        if isPhaseData:
            self.workingProjections = -self.workingProjections
        self.workingProjections = (self.workingProjections - np.min(self.workingProjections)) / (np.max(self.workingProjections) - np.min(self.workingProjections))
        self.finalProjections = np.copy(self.workingProjections)

    def standardize(self, isPhaseData):
        """
        Standardizes projections to have zero mean and unit variance, then inverts the sign.

        Parameters:
        - isPhaseData (bool): If True, inverts the sign of the projections, which is often necessary for phase data since it can be negative.
        """
        self.workingProjections = (self.workingProjections - np.mean(self.workingProjections)) / np.std(self.workingProjections)
        if isPhaseData:
            #This is done to invert the sign of the projections, which is necessary for phase data since it is often negative.
            self.workingProjections *= -1

    def threshold(self, threshold=-0.1):
        """
        Thresholds the projections, setting values below the threshold to 1 and others to 0.

        Parameters:
        - threshold (float): Threshold value.
        """
        self.workingProjections = (self.workingProjections <= threshold).astype(float)

    def makeNotebookProjMovie(self):
        """
        Displays a movie of the final projections in a Jupyter notebook.
        """
        MoviePlotter(self.finalProjections)

    def makeScriptProjMovie(self):
        """
        Displays a movie of the final projections in a script environment.
        """
        runwidget(self.finalProjections)

    def makeNotebookReconMovie(self):
        """
        Displays a movie of the reconstructed volume in a Jupyter notebook.
        """
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        """
        Displays a movie of the reconstructed volume in a script environment.
        """
        runwidget(self.recon)

    def bilateralFilter(self, *args, **kwargs):
        print("\n")
        return bilateralFilter(self, *args, **kwargs)

    def cross_correlate_align(self, *args, **kwargs):
        print("\n")
        return cross_correlate_align(self, *args, **kwargs)

    def rotate_correlate_align(self, *args, **kwargs):
        print("\n")
        return rotate_correlate_align(self, *args, **kwargs)

    def PMA(self, *args, **kwargs):
        print("\n")
        return PMA(self, *args, **kwargs)

    def vertical_mass_fluctuation_align(tomo, *args, **kwargs):
        print("\n")
        return vertical_mass_fluctuation_align(tomo, *args, **kwargs)

    def tomopy_align(self, *args, **kwargs):
        print("\n")
        return tomopy_align(self, *args, **kwargs)

    def optical_flow_align(self, *args, **kwargs):
        print("\n")
        return optical_flow_align(self, *args, **kwargs)

    def center_projections(self):
        """
        Determines and adjusts the center of rotation for 2D projection images by finding the initial center,
        shifting the projections to center them, and calculating any remaining offset (to check if it needs to be done again).
        """
        print("Centering Projections")
        self.center_offset = 10
        while self.center_offset > 1:
            self.rotation_center = tomopy.find_center_vo(self.workingProjections)
            print("Original center: {}".format(self.rotation_center))
            print("Center of frame: {}".format(self.image_size[1] // 2))
            x_shift = (self.image_size[1] / 2 - (self.rotation_center))
            y_shift = 0
            x_shift_check = 3
            if abs(x_shift_check) > 2:
                for m in range(self.num_angles):
                    self.workingProjections[m] = subpixel_shift(self.workingProjections[m], y_shift, x_shift)
                self.rotation_center = tomopy.find_center_vo(self.workingProjections)
                print("Aligned projections shifted by {} pixels".format(x_shift))
                x_shift_check = (self.image_size[1] // 2 - (self.rotation_center))
            self.center_offset = abs(x_shift_check)
            print(f"Projections are currently centered at pixel {self.rotation_center}. Residual offset: {self.center_offset}")
            self.tracked_shifts[:, 1] += x_shift

    def reconstruct(self, algorithm, snr_db=None):
        """
        Reconstructs the 3D volume from projections using the specified algorithm.

        Parameters:
        - algorithm (str): The reconstruction algorithm to use.
        - snr_db (float or None): Signal-to-noise ratio for SVMBIR, if applicable.
        """
        #Center projections before reconstruction. So reconstruction knows where center is.
        self.rotation_center = tomopy.find_center_vo(self.finalProjections)

        print("\n")
        if algorithm.endswith("CUDA"):
            if torch.cuda.is_available():
                print("Using GPU-accelerated reconstruction, Algorithm: ", algorithm)
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': 400,
                    'extra_options': {}
                }
                self.recon = tomopy.recon(
                    self.finalProjections,
                    self.ang,
                    center=self.rotation_center,
                    algorithm=tomopy.astra,
                    options=options,
                    ncore=1
                )
            else:
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            print("Using SVMBIR-based reconstruction.")
            print("center_offset assumed to be : {}".format(self.center_offset))
            if snr_db is None:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset=self.center_offset, verbose=1)
            else:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset=self.center_offset, snr_db=snr_db, verbose=1)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(
                self.finalProjections,
                self.ang,
                center=self.rotation_center,
                algorithm=algorithm,
                sinogram_order=False
            )
        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.98)
        print("Reconstruction completed.")