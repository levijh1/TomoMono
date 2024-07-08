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
import cv2




class tomoData:
    #Dimensions
    numAngles = 180

    def __init__(self, data, totalNumAngles = 800):
        self.numAngles = data.shape[0]
        self.imageSize = data.shape[1:]
        self.original = data
        self.ang = tomopy.angles(nang=self.numAngles, ang1=0, ang2=(360/totalNumAngles)*self.numAngles)
        self.projections = np.copy(data)

    def get_prj(self):
        return self.prj
    
    def get_recon(self):
        return self.recon
    
    def get_projections(self):
        return self.projections

    def jitter(self):
        self.prj_jitter = self.projections.copy()
        m = self.prj_jitter
        for i in range(0, self.numAngles):  ###As of now the first image stays consistent)
            xshift = 4 * random.random() - 2
            yshift = 4 * random.random() - 2
            m[i] = sp.ndimage.shift(m[i], (xshift, yshift), mode="wrap")
        self.projections = self.prj_jitter.copy()

    def crop(self, new_x, new_y):
        """
        Crop each 2D numpy array in a 3D array to a specified size (new_x, new_y) centered in the middle of the array.

        Parameters:
        - new_x: The target width of the crop.
        - new_y: The target height of the crop.

        Returns:
        - Cropped 3D numpy array.
        """
        cropped_array = np.zeros((self.projections.shape[0], new_y, new_x), dtype=self.projections.dtype)
        
        for i, array in enumerate(self.projections):
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

    def makeNotebookProjMovie(self):
        MoviePlotter(self.projections)

    def makeScriptProjMovie(self):
        runwidget(self.projections)

    def makeNotebookReconMovie(self):
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        runwidget(self.recon)

    def tomopyAlign(self, iterations = 10):
        align_info = tomopy.prep.alignment.align_joint(self.projections, self.ang, algorithm='sirt', iters=iterations, debug=True)
        self.projections = tomopy.shift_images(self.projections, align_info[1], align_info[2])

    def opticalFlowAlign(self):
        nr, nc = self.projections[0].shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        if torch.cuda.is_available():
            ...
        else:
            for m in tqdm(range(1, self.numAngles+1), desc='Optical Flow Alignment of Projections'):
                if m < self.numAngles:
                    v, u = optical_flow_tvl1(self.projections[m-1], self.projections[m])
                    self.projections[m] = warp(self.projections[m], np.array([row_coords + v, col_coords + u]), mode='edge')
                else:
                    v, u = optical_flow_tvl1(self.projections[m-1], self.projections[0])
                    self.projections[0] = warp(self.projections[0], np.array([row_coords + v, col_coords + u]), mode='edge')

    def recon(self):
        #print("Normalizing projections")
        #self.projections = tomopy.prep.normalize.normalize_bg(self.projections, air=10)

        print("Finding center of rotation")
        rot_center = tomopy.find_center(self.projections, self.ang)



        # Check if ASTRA is available and if a GPU device is present
        if torch.cuda.is_available():
            # Use an ASTRA-supported GPU algorithm, e.g., 'SIRT_CUDA'

            # extra_options = {'MinConstraint': 0}
            extra_options = {}
            options = {
                'proj_type': 'cuda',
                'method': 'SIRT_CUDA',
                'num_iter': 200,
                'extra_options': extra_options
            }
            print("Using GPU-accelerated reconstruction.")
            recon = tomopy.recon(self.projections,
                     self.ang,
                     center=rot_center,
                     algorithm=tomopy.astra,
                     options=options,
                     ncore=1)
        else:
            # Fallback to a CPU-based algorithm if no GPU is available
            print("Using CPU-based reconstruction.")
            recon = tomopy.recon(self.projections, self.ang, center=rot_center, algorithm='sirt', sinogram_order=False)

        # print("Applying circular mask")
        # self.recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
