import tomopy
import numpy as np
import random
import scipy as sp
from helperFunctions import MoviePlotter, subpixel_shift, runwidget, convert_to_numpy
# from skimage.transform import warp
import torch
# from skimage.registration import optical_flow_tvl1
from tqdm import tqdm
# from scipy.signal import correlate
from scipy.ndimage import shift, center_of_mass, rotate
import svmbir
import cv2
import matplotlib.pyplot as plt
from alignment_methods import *


class tomoData:

    def __init__(self, data):
        """
        Initializes the TomoData object with the provided dataset.
        
        Parameters:
        - data (np.array): The tomographic data as a 3D numpy array. The first dimension being the projection number.
        """
        self.num_angles = data.shape[0]
        self.image_size = data.shape[1:]
        self.data = data
        self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=360)
        self.workingProjections = np.copy(data)
        self.rotation_center = 0
        self.center_offset = 0
        self.finalProjections = np.copy(data)
        self.tracked_shifts = np.zeros((self.num_angles,2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def reset_workingProjections(self, x_size = 900, y_size = 650):
        self.workingProjections = np.copy(self.data)
        self.finalProjections = np.copy(self.data)
        self.crop_center(x_size, y_size)

    
    def get_recon(self):
        """Returns the reconstructed 3D Model."""
        return self.recon
    
    def get_workingprojections(self):
        """Returns the current state of projections."""
        return self.workingProjections
    
    def get_finalProjections(self):
        return self.finalProjections

    def jitter(self, multiplier = 10):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        """
        self.prj_jitter = self.workingProjections.copy()
        for i in range(self.num_angles):  # Now includes the first image as well
            x_shift = multiplier * (random.random() - 0.5)
            y_shift = multiplier * (random.random() - 0.5)
            self.prj_jitter[i] = sp.ndimage.shift(self.prj_jitter[i], (x_shift, y_shift), mode="constant")
        self.workingProjections = self.prj_jitter
        self.finalProjections = self.prj_jitter
        self.data = self.prj_jitter

    # def add_noise(self):
    #     noisyData = tomopy.sim.project.add_poisson(self.workingProjections)
    #     self.workingProjections = noisyData
    #     self.data = noisyData

    def crop_center(self, new_x, new_y):
        """
        Crops each 2D numpy array in a 3D array to a specified size centered in the middle of the array.
        
        Parameters:
        - new_x (int): The target width of the crop.
        - new_y (int): The target height of the crop.
        
        Returns:
        - np.array: Cropped 3D numpy array.
        """
        print("\n")
        cropped_array = np.zeros((self.workingProjections.shape[0], new_y, new_x), dtype=self.workingProjections.dtype)
        
        for i, array in enumerate(tqdm(self.workingProjections, desc=f"Cropping projections to size: {new_x}x{new_y}")):
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
        
        self.workingProjections = cropped_array
        self.finalProjections = cropped_array

        self.image_size = self.workingProjections.shape[1:]

    def crop_bottom_center(self, new_y, new_x):
        print("\n")
        cropped_array = np.zeros((self.workingProjections.shape[0], new_y, new_x), dtype=self.workingProjections.dtype)

        print(f"Cropping projections to size: {new_x}x{new_y}")
        for i, array in enumerate(self.workingProjections):
            y, x = array.shape  # Get the current dimensions of the array
            
            # Calculate the starting and ending indices for the crop
            startx = x // 2 - new_x // 2
            endx = startx + new_x
            starty = y - new_y
            endy = y
            
            # Ensure indices are within bounds
            startx, endx = max(0, startx), min(x, endx)
            starty, endy = max(0, starty), min(y, endy)
            
            # Crop the array and assign to the corresponding position in the output array
            cropped_array[i] = array[starty:endy, startx:endx]
        
        self.workingProjections = cropped_array
        self.finalProjections = cropped_array

        self.image_size = self.workingProjections.shape[1:]

    def track_shifts(self):
        self.finalProjections = self.workingProjections.copy()
        self.tracked_shifts = np.zeros((self.num_angles,2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def make_updates_shift(self):
        for m in tqdm(range(self.num_angles), desc='Apply shifts to final projections'):
            self.finalProjections[m] = subpixel_shift(self.finalProjections[m], self.tracked_shifts[m,0], self.tracked_shifts[m,1])
            # self.finalProjections[m] = shift(self.finalProjections[m], shift=[round(self.tracked_shifts[m,0]), round(self.tracked_shifts[m,1])], mode='nearest')
        self.tracked_shifts = np.zeros((self.num_angles,2))

    def make_updates_rotate(self):
        for m in tqdm(range(self.num_angles), desc='Apply rotations to final projections'):
            self.finalProjections[m] = rotate(self.finalProjections[m], self.tracked_rotations[m], reshape=False, mode='constant')

        self.tracked_shifts = np.zeros((self.num_angles,2))
        

    def normalize(self):
        print("\n")
        print("Normalizing projections")
        """Normalize all projections to be positive values between 1 and 0"""
        self.workingProjections = -self.workingProjections
        self.workingProjections = (self.workingProjections - np.min(self.workingProjections)) / (np.max(self.workingProjections) - np.min(self.workingProjections))

        self.finalProjections = np.copy(self.workingProjections)

    def standardize(self):
        """Values in terms of number of standard deviations from the mean"""
        self.workingProjections = (self.workingProjections-np.mean(self.workingProjections))/np.std(self.workingProjections)
        self.workingProjections *= -1 ##REMOVE THIS LINE FOR PHANTOM DATA

    def threshold(self, threshold=-0.1):
        self.workingProjections = (self.workingProjections<=threshold).astype(float)

    def makeNotebookProjMovie(self):
        MoviePlotter(self.finalProjections)
        # MoviePlotter(self.workingProjections)

    def makeScriptProjMovie(self):
        # toShow = self.finalProjections[:,150:-100,150:-150]
        # runwidget(toShow)
        runwidget(self.finalProjections)

    def makeNotebookReconMovie(self):
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        runwidget(self.recon)
    
    def bilateralFilter(self, *args, **kwargs):
        print("\n")
        return bilateralFilter(self, *args, **kwargs)

    def cross_correlate_align(self, *args, **kwargs):
        print("\n")
        return cross_correlate_align(self, *args, **kwargs)

    def cross_correlate_tip(self, *args, **kwargs):
        print("\n")
        return cross_correlate_tip(self, *args, **kwargs)

    def rotate_correlate_align(self, *args, **kwargs):
        print("\n")
        return rotate_correlate_align(self, *args, **kwargs)

    def bilateralFilter(self, *args, **kwargs):
        print("\n")
        return bilateralFilter(self, *args, **kwargs)

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

    def optical_flow_align_chill(self, *args, **kwargs):
        print("\n")
        return optical_flow_align(self, *args, **kwargs)

    
    def center_projections(self):
        """"
        This function determines and adjusts the center of rotation for 2D projection images by finding the initial center, shifting the projections to center them, and calculating any remaining offset.
        """
        print("\n")
        print("Centering Projections")
        self.center_offset = 10
        while self.center_offset > 1:
            self.rotation_center = tomopy.find_center_vo(self.workingProjections)
            print("Original center: {}".format(self.rotation_center))
            print("Center of frame: {}".format(self.image_size[1]//2))
            x_shift = (self.image_size[1]/2 - (self.rotation_center))
            y_shift = 0
            x_shift_check = 3
            if abs(x_shift_check) > 2:
                for m in range(self.num_angles):
                    self.workingProjections[m] = subpixel_shift(self.workingProjections[m], y_shift, x_shift)
                
                self.rotation_center = tomopy.find_center_vo(self.workingProjections)
                print("Aligned projections shifted by {} pixels".format(x_shift))
                x_shift_check = (self.image_size[1]//2 - (self.rotation_center))
   
            #Check how well center_projections actually performed
            self.center_offset = abs(x_shift_check)
            print("Projections are currently centered at pixel {}".format(self.rotation_center))
            print("But it is still offset by {} pixels".format(self.center_offset))

            #Track whatever shift we added
            self.tracked_shifts[:,1] += x_shift
        

    def reconstruct(self, algorithm, snr_db):
        ##Optional thing you can do if you want to initialize the reconstruction with a guess prior to beginning to reconstruct
        # recon_location = "reconstructions/foamRecon_Normalized_20240801-132117_svmbir.tif"
        # tomo, tomo_scale_info = convert_to_numpy(recon_location)

        #Check if data has been centered yet
        self.center_projections()

        #Check which algorithm is being used
        print("\n")
        if algorithm.endswith("CUDA"):
            if torch.cuda.is_available():
                print("Using GPU-accelerated reconstruction.")
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': 400,
                    'extra_options': {}
                }
                self.recon = tomopy.recon(self.finalProjections,
                                        self.ang,
                                        center=self.rotation_center,
                                        algorithm=tomopy.astra,
                                        options=options,
                                        # init_recon=tomo,
                                        ncore=1)
            else: 
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            print("Using SVMBIR-based reconstruction.")
            print("center_offset assumed to be : {}".format(self.center_offset))
            if snr_db == None:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset = self.center_offset, verbose=1)
            else:
                self.recon = svmbir.recon(self.finalProjections, self.ang, center_offset = self.center_offset, snr_db=snr_db, verbose=1)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(self.finalProjections,
                                      self.ang,
                                      center=self.rotation_center,
                                      algorithm=algorithm,
                                    #   init_recon=tomo,
                                      sinogram_order=False)

        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.98)
        print("Reconstruction completed.")
