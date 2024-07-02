from helperFunctions import FFT2, IFFT2
import tomopy
import numpy as np
import random
import scipy as sp
import skimage.transform as sk
from helperFunctions import show, MoviePlotter
from pltwidget import runwidget



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
        return self.recon_normal
    
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

    def makeNotebookMovie(self):
        MoviePlotter(self.projections)

    def makeScriptMovie(self):
        runwidget(self.projections)

    def tomopyAlign(self, iterations = 10):
        align_info = tomopy.prep.alignment.align_joint(self.projections, self.ang, algorithm='sirt', iters=iterations)
        self.projections = tomopy.shift_images(self.projections, align_info[1], align_info[2])

    def recon(self):
        self.recon = tomopy.recon(tomo=self.projections, theta=self.ang, algorithm='sirt')