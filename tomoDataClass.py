from helperFunctions import FFT2, IFFT2
import tomopy
import numpy as np
import random
import scipy as sp
import skimage.transform as sk
from helperFunctions import show, MoviePlotter
from pltwidget import runwidget
from skimage.transform import warp
import torch
from skimage.registration import optical_flow_tvl1
from tqdm import tqdm
from scipy.signal import correlate
from scipy.ndimage import shift
import cv2 as cv
import svmbir

class tomoData:

    def __init__(self, data, total_angles=400):
        """
        Initializes the TomoData object with the provided dataset.
        
        Parameters:
        - data (np.array): The tomographic data as a 3D numpy array.
        - total_angles (int): Total number of angles where measurements were taken
        """
        self.num_angles = data.shape[0]
        self.image_size = data.shape[1:]
        self.original = data
        self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=(360 / total_angles) * self.num_angles)
        self.projections = np.copy(data)
    
    def get_recon(self):
        """Returns the reconstructed 3D Model."""
        return self.recon
    
    def get_projections(self):
        """Returns the current state of projections."""
        return self.projections

    def jitter(self):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        """
        self.prj_jitter = self.projections.copy()
        for i in range(self.num_angles):  # Now includes the first image as well
            x_shift = 8 * random.random() - 4
            y_shift = 8 * random.random() - 4
            self.prj_jitter[i] = sp.ndimage.shift(self.prj_jitter[i], (x_shift, y_shift), mode="wrap")
        self.projections = self.prj_jitter

    def crop(self, new_x, new_y):
        """
        Crops each 2D numpy array in a 3D array to a specified size centered in the middle of the array.
        
        Parameters:
        - new_x (int): The target width of the crop.
        - new_y (int): The target height of the crop.
        
        Returns:
        - np.array: Cropped 3D numpy array.
        """
        cropped_array = np.zeros((self.projections.shape[0], new_y, new_x), dtype=self.projections.dtype)
        
        for i, array in enumerate(tqdm(self.projections, desc=f"Cropping projections to size: {new_x}x{new_y}")):
            y, x = array.shape  # Get the current dimensions of the array
            
            # Calculate the starting and ending indices for the crop
            startx = x // 2 - new_x // 2
            endx = startx + new_x
            starty = y // 2 - new_y // 2
            endy = starty + new_y
            
            # Ensure indices are within bounds
            startx, endx = max(0, startx), min(x, endx)
            starty, endy = max(0, starty), min(y, endy)
            
            # Crop the array and assign to the corresponding position in the output array
            cropped_array[i] = array[starty:endy, startx:endx]
        
        self.projections = cropped_array

    def normalize(self):
        """Normalize all projections to be positive values between 1 and 0"""
        self.projections = -self.projections
        self.projections = (self.projections - np.min(self.projections)) / (np.max(self.projections) - np.min(self.projections))

    def makeNotebookProjMovie(self):
        MoviePlotter(self.projections)

    def makeScriptProjMovie(self):
        runwidget(self.projections)

    def makeNotebookReconMovie(self):
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        runwidget(self.recon)

    def cross_correlate_align(self):
        """
        Aligns projections using cross-correlation to find the shift between consecutive images.
        """
        num_rows, num_cols = self.projections[0].shape
        for m in tqdm(range(1, self.num_angles + 1), desc='Cross-Correlation Alignment of Projections'):
            # Handle circular indexing for the last projection
            img1 = self.projections[m - 1]
            img2 = self.projections[m % self.num_angles]

            # Compute the cross-correlation between two consecutive images
            correlation = correlate(img1, img2, mode='same')
            
            # Find the index of the maximum correlation value
            y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate the shifts, considering the center of the images as the origin
            y_shift -= num_rows // 2
            x_shift -= num_cols // 2

            # Apply the calculated shift to align the images
            self.projections[m % self.num_angles] = shift(img2, shift=[y_shift, x_shift], mode='nearest')

    def tomopy_align(self, iterations = 10):
        """
        Aligns projections using tomopy's join repojection Alignment algorithm
        """
        print("Tomopy Joint Reprojection Alignment of Projections (" + str(iterations) + " iterations)")
        align_info = tomopy.prep.alignment.align_joint(self.projections, self.ang, algorithm='sirt', iters=iterations, debug=True)
        self.projections = tomopy.shift_images(self.projections, align_info[1], align_info[2])

    def optical_flow_align(self):
        """
        Aligns projections using optical flow to estimate the motion between consecutive images.
        """
        num_rows, num_cols = self.projections[0].shape
        row_coords, col_coords = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        for m in tqdm(range(1, self.num_angles + 1), desc='Optical Flow Alignment of Projections'):
            # Handle circular indexing for the last projection
            prev_img = self.projections[m - 1]
            current_img = self.projections[m % self.num_angles]

            # Compute optical flow between two consecutive images
            v, u = optical_flow_tvl1(prev_img, current_img)

            # Apply the flow vectors to align the current image
            aligned_img = warp(current_img, np.array([row_coords + v, col_coords + u]), mode='edge')
            self.projections[m % self.num_angles] = aligned_img

    def reconstruct(self, algorithm):
        """
        Performs reconstruction of the projections, utilizing GPU acceleration if available.
        """
        print("Finding center of rotation")
        print("Middle of image: ", self.image_size[1] // 2)
        rotation_center = tomopy.find_center_vo(self.projections)
        print("Estimated center of rotation: ", rotation_center)

        if algorithm == 'gpu':
            if torch.cuda.is_available():
                print("Using GPU-accelerated reconstruction.")
                options = {
                    'proj_type': 'cuda',
                    'method': 'SIRT_CUDA',
                    'num_iter': 200,
                    'extra_options': {}
                }
                self.recon = tomopy.recon(self.projections,
                                        self.ang,
                                        center=rotation_center,
                                        algorithm=tomopy.astra,
                                        options=options,
                                        ncore=1)
            else: 
                raise ValueError("GPU is available, but the selected algorithm is not GPU-accelerated.")
        elif algorithm == 'svmbir':
            print("Using SVMBIR-based reconstruction.")
            self.recon = svmbir.recon(self.projections, self.ang, verbose=2)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(self.projections,
                                      self.ang,
                                      center=rotation_center,
                                      algorithm=algorithm,
                                      sinogram_order=False)

        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.90)
        print("Reconstruction completed.")
