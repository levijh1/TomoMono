import tomopy
import numpy as np
import random
import scipy as sp
from helperFunctions import MoviePlotter, subpixel_shift
from pltwidget import runwidget
from skimage.transform import warp
import torch
from skimage.registration import optical_flow_tvl1
from tqdm import tqdm
from scipy.signal import correlate
from scipy.ndimage import shift, center_of_mass, rotate
import svmbir
from tiffConverter import convert_to_numpy

class tomoData:

    def __init__(self, data, total_angles=400):
        """
        Initializes the TomoData object with the provided dataset.
        
        Parameters:
        - data (np.array): The tomographic data as a 3D numpy array. The first dimension being the projection number.
        - total_angles (int): Total number of angles where measurements were taken
        """
        self.num_angles = data.shape[0]
        self.image_size = data.shape[1:]
        self.data = data
        self.ang = tomopy.angles(nang=self.num_angles, ang1=0, ang2=(360 / total_angles) * self.num_angles)
        # self.ang = np.load('data/angles_90p.npy')
        self.projections = np.copy(data)
        self.rotation_center = 0
        self.center_offset = 0
        self.originalProjections = self.projections.copy()
        self.tracked_shifts = np.zeros((self.num_angles,2))
    
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

    def crop_center(self, new_x, new_y):
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

        self.image_size = self.projections.shape[1:]

    def crop_bottom_center(self, new_y, new_x):
        cropped_array = np.zeros((self.projections.shape[0], new_y, new_x), dtype=self.projections.dtype)
        
        for i, array in enumerate(tqdm(self.projections, desc=f"Cropping projections to size: {new_x}x{new_y}")):
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
        
        self.projections = cropped_array

        self.image_size = self.projections.shape[1:]

    def track_shifts(self):
        self.originalProjections = self.projections.copy()
        self.tracked_shifts = np.zeros((self.num_angles,2))

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

            self.tracked_shifts[m % self.num_angles][0] = y_shift
            self.tracked_shifts[m % self.num_angles][1] = x_shift


    def find_optimal_rotation(self, img1, img2, angle_range=[-10, 10], angle_step=1):
        """
        Calculates the rotation angle between two projections that maximizes their similarity.
        
        Parameters:
        - img1: The first projection image.
        - img2: The second projection image, which will be rotated to find the optimal alignment.
        - angle_range: A tuple (min_angle, max_angle) defining the range of angles to test.
        - angle_step: The granularity of angles to test within the range.
        
        Returns:
        - optimal_angle: The angle that maximizes the similarity between the two projections.
        - max_similarity: The maximum similarity score achieved.
        """
        # print("Finding optimal rotation")
        max_similarity = -100000
        optimal_angle = 0

        for angle in np.arange(angle_range[0],angle_range[1] + angle_step, angle_step):
            # Rotate img2 by the current angle
            rotated_img2 = rotate(img2, angle, reshape=False, mode='nearest')
            
            # Compute the similarity (cross-correlation) between img1 and the rotated img2
            similarity = np.max(correlate(img1, rotated_img2, mode='same'))
            
            # Update the optimal angle if the current similarity is the highest found so far
            if similarity > max_similarity:
                max_similarity = similarity
                optimal_angle = angle

        # print(f"Optimal rotation angle: {optimal_angle} degrees, Maximum similarity: {max_similarity}")
        return optimal_angle, max_similarity

    def rotate_correlate_align(self):
        """
        WARNING: Does not have ability to be tracked my track_shifts
        """
        for i in tqdm(range(self.num_angles//2), desc='rotation alignment'):
            angle, maxSim = self.find_optimal_rotation(self.projections[i], self.projections[(i+400)%800])
            self.projections[i] = rotate(self.projections[i], -angle/2, reshape=False, mode='nearest')
            self.projections[(i+400)%800] = rotate(self.projections[(i+400)%800], angle/2, reshape=False, mode='nearest')

    def vertical_mass_fluctuation_align(self):
        """
        This function aligns 2D projection images vertically based on their center-of-mass fluctuations by cross-correlating each projection with the reference projection and shifting accordingly.
        """
        print("Vertical Mass Fluctuation Alignment")
        sums = []
        for k in range(self.num_angles):
            sums.append(np.sum(self.projections[k], axis=1).tolist())
            if k > 0:
                CC = sp.signal.correlate(sums[0], sums[k], mode='same', method='fft')
                maxpoint = np.where(CC == CC.max())
                yshift = int(self.image_size[0] / 2 - maxpoint[0])
                self.projections[k] = subpixel_shift(self.projections[k], -yshift, 0)
                self.tracked_shifts[k,0] += yshift

    def tomopy_align(self, iterations = 10):
        """
        Aligns projections using tomopy's join repojection Alignment algorithm
        """
        print("Tomopy Joint Reprojection Alignment of Projections (" + str(iterations) + " iterations)")
        scale = max(abs(self.projections.max()), abs(self.projections.min()))
        align_info = tomopy.prep.alignment.align_joint(self.projections, self.ang, algorithm='sirt', iters=iterations, debug=True)[1:3]
        align_info = np.array(align_info)
        self.tracked_shifts[:,0] += align_info[0] * scale
        self.tracked_shifts[:,1] += align_info[1] * scale


        self.projections = tomopy.shift_images(self.projections, align_info[0], align_info[1])

    def optical_flow_align(self):
        """
        Aligns projections using optical flow to estimate the motion between consecutive images.

        WARNING: Does not have ability to be tracked my track_shifts
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
    
    def center_projections(self):
            """"
            This function determines and adjusts the center of rotation for 2D projection images by finding the initial center, shifting the projections to center them, and calculating any remaining offset.
            """
            print("Finding center of rotation for projections")
            self.rotation_center = tomopy.find_center_vo(self.projections)
            print("Original center: {}".format(self.rotation_center))
            print("Center of frame: {}".format(self.image_size[1]//2))
            x_shift = (self.image_size[1]/2 - self.rotation_center)
            y_shift = 0
            if abs(x_shift) > 1:
                for m in tqdm(range(self.num_angles), desc='Center projections'):
                    self.projections[m] = subpixel_shift(self.projections[m], y_shift, x_shift)
                
                self.rotation_center = tomopy.find_center_vo(self.projections)
                print("Aligned projections shifted by {} pixels".format(x_shift))
                x_shift = (self.image_size[1]//2 - self.rotation_center)
   
            #Check how well center_projections actually performed
            self.center_offset = abs(x_shift)
            print("Projections are currently centered at pixel {}".format(self.rotation_center))
            print("But it is still offset by {} pixels".format(self.center_offset))

            #Track whatever shift we added
            self.tracked_shifts[:,1] += x_shift
            

    def reconstruct(self, algorithm, snr_db):
        
        recon_location = "reconstructions/foamRecon_Normalized_20240801-132117_svmbir.tif"
        tomo, tomo_scale_info = convert_to_numpy(recon_location)

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
                self.recon = tomopy.recon(self.projections,
                                        self.ang,
                                        center=self.rotation_center,
                                        algorithm=tomopy.astra,
                                        options=options,
                                        init_recon=tomo,
                                        ncore=1)
            else: 
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            print("Using SVMBIR-based reconstruction.")
            print("center_offset assumed to be : {}".format(self.center_offset))
            if snr_db == None:
                self.recon = svmbir.recon(self.projections, self.ang, center_offset = self.center_offset, init_image = tomo, verbose=1)
            else:
                self.recon = svmbir.recon(self.projections, self.ang, center_offset = self.center_offset, init_image = tomo, snr_db=snr_db, verbose=1)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(self.projections,
                                      self.ang,
                                      center=self.rotation_center,
                                      algorithm=algorithm,
                                    #   init_recon=tomo,
                                      sinogram_order=False)

        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.98)
        print("Reconstruction completed.")
