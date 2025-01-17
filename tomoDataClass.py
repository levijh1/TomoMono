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
import cv2
import matplotlib.pyplot as plt


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
        self.finalProjections = self.workingProjections.copy()
        self.tracked_shifts = np.zeros((self.num_angles,2))
        self.tracked_rotations = np.zeros(self.num_angles)

    def reset_workingProjections(self):
        self.workingProjections = np.copy(self.data)
        self.crop_center(900,650)

    
    def get_recon(self):
        """Returns the reconstructed 3D Model."""
        return self.recon
    
    def get_workingprojections(self):
        """Returns the current state of projections."""
        return self.workingProjections
    
    def get_finalProjections(self):
        return self.finalProjections

    def jitter(self):
        """
        Applies random jitter to the projections to simulate real-world misalignments.
        """
        self.prj_jitter = self.workingProjections.copy()
        for i in range(self.num_angles):  # Now includes the first image as well
            x_shift = 4 * random.random() - 2
            y_shift = 4 * random.random() - 2
            self.prj_jitter[i] = sp.ndimage.shift(self.prj_jitter[i], (x_shift, y_shift), mode="constant")
        self.workingProjections = self.prj_jitter

    def crop_center(self, new_x, new_y):
        """
        Crops each 2D numpy array in a 3D array to a specified size centered in the middle of the array.
        
        Parameters:
        - new_x (int): The target width of the crop.
        - new_y (int): The target height of the crop.
        
        Returns:
        - np.array: Cropped 3D numpy array.
        """
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

        self.image_size = self.workingProjections.shape[1:]

    def crop_bottom_center(self, new_y, new_x):
        cropped_array = np.zeros((self.workingProjections.shape[0], new_y, new_x), dtype=self.workingProjections.dtype)
        
        for i, array in enumerate(tqdm(self.workingProjections, desc=f"Cropping projections to size: {new_x}x{new_y}")):
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
        """Normalize all projections to be positive values between 1 and 0"""
        self.workingProjections = -self.workingProjections
        self.workingProjections = (self.workingProjections - np.min(self.workingProjections)) / (np.max(self.workingProjections) - np.min(self.workingProjections))

    def makeNotebookProjMovie(self):
        MoviePlotter(self.finalProjections)

    def makeScriptProjMovie(self):
        toShow = self.finalProjections[:,150:-100,150:-150]
        runwidget(toShow)

    def makeNotebookReconMovie(self):
        MoviePlotter(self.recon)

    def makeScriptReconMovie(self):
        runwidget(self.recon)
        
    def bilateralFilter(self, d = 15, sigmaColor = 0.3, sigmaSpace = 100):
        print("Bilateral filter being applied")
        for i in tqdm(range(self.workingProjections.shape[0]), desc="Applying bilateral filter to projections"):
            self.workingProjections[i] = cv2.bilateralFilter(self.workingProjections[i], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    def cross_correlate_align(self, tolerance=1, max_iterations=15):
        """
        Aligns projections using cross-correlation to find the shift between consecutive images.
        Iterates until the average shift in pixels is less than the specified tolerance.
        """
        for iteration in tqdm(range(max_iterations), desc='Cross-Correlation Alignment Iterations'):
            total_shift = 0
            for m in tqdm(range(1, self.num_angles + 1), desc=f'Iteration {iteration + 1}'):
                # Handle circular indexing for the last projection
                img1 = self.workingProjections[m - 1][150:-50,120:-120]
                img2 = self.workingProjections[m % self.num_angles][150:-50,120:-120]
                num_rows, num_cols = img1.shape

                # Compute the cross-correlation between two consecutive images
                correlation = correlate(img1, img2, mode='same')
                
                # Find the index of the maximum correlation value
                y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                # Calculate the shifts, considering the center of the images as the origin
                y_shift -= num_rows // 2
                x_shift -= num_cols // 2

                # Apply the calculated shift to align the images
                self.workingProjections[m % self.num_angles] = subpixel_shift(self.workingProjections[m % self.num_angles], y_shift, x_shift)

                # Store the shifts
                self.tracked_shifts[m % self.num_angles][0] += y_shift
                self.tracked_shifts[m % self.num_angles][1] += x_shift

                # Accumulate the total shift magnitude for this iteration
                total_shift += np.sqrt(y_shift**2 + x_shift**2)
            
            # Calculate the average shift for this iteration
            average_shift = total_shift / self.num_angles
            print(f"\nAverage pixel shift of iteration {iteration}: {average_shift}")

            # Check if the average shift is below the tolerance
            if average_shift < tolerance:
                print(f'Convergence reached after {iteration + 1} iterations.')
                break
        print(f'Maximum iterations reached without convergence.')


    def cross_correlate_tip(self, tolerance=0.5, max_iterations=15):
        """
        Aligns projections using cross-correlation to find the shift between consecutive images.
        Iterates until the average shift in pixels is less than the specified tolerance.
        """
        for iteration in tqdm(range(max_iterations), desc='Cross-Correlation of the pillar tip iterations'):
            total_shift = 0
            for m in tqdm(range(1, self.num_angles + 1), desc=f'Iteration {iteration + 1}'):
                # Handle circular indexing for the last projection
                img1 = self.workingProjections[m - 1][150:230, 350:-350]
                img2 = self.workingProjections[m % self.num_angles][150:230, 350:-350]
                num_rows, num_cols = img1.shape

                # plt.imshow(self.workingProjections[m-1])
                # plt.show()
                # plt.imshow(img1)
                # plt.show()

                # Compute the cross-correlation between two consecutive images
                correlation = correlate(img1, img2, mode='same')
                
                # Find the index of the maximum correlation value
                y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                # Calculate the shifts, considering the center of the images as the origin
                y_shift -= num_rows // 2
                x_shift -= num_cols // 2

                # Apply the calculated shift to align the images
                self.workingProjections[m % self.num_angles] = subpixel_shift(self.workingProjections[m % self.num_angles], y_shift, x_shift)

                # Store the shifts
                self.tracked_shifts[m % self.num_angles][0] += y_shift
                self.tracked_shifts[m % self.num_angles][1] += x_shift

                # Accumulate the total shift magnitude for this iteration
                total_shift += np.sqrt(y_shift**2 + x_shift**2)
            
            # Calculate the average shift for this iteration
            average_shift = total_shift / self.num_angles
            print(f"\nAverage pixel shift of iteration {iteration}: {average_shift}")

            # Check if the average shift is below the tolerance
            if average_shift < tolerance:
                print(f'Convergence reached after {iteration + 1} iterations.')
                break
        print(f'Maximum iterations reached without convergence.')



    def find_optimal_rotation(self, img1, img2, angle_range=[-5, 5], angle_step=0.25):
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
            rotated_img2 = rotate(np.copy(img2), angle, reshape=False, mode='constant')
            
            # Compute the similarity (cross-correlation) between img1 and the rotated img2
            similarity = np.max(correlate(img1[200:-50,120:-120], rotated_img2[200:-50,120:-120], mode='same'))
            
            # Update the optimal angle if the current similarity is the highest found so far
            if similarity > max_similarity:
                max_similarity = similarity
                optimal_angle = angle

        # print(f"Optimal rotation angle: {optimal_angle} degrees, Maximum similarity: {max_similarity}")
        return optimal_angle, max_similarity

    def rotate_correlate_align(self, max_iterations = 10, tolerance = 0.5):
        for iteration in tqdm(range(max_iterations), desc='Rotation Correlation Alignment Iterations'):
            total_angle_rotation = 0
            for i in tqdm(range(self.num_angles//2), desc=f'Iteration {iteration + 1}'):
                angle, maxSim = self.find_optimal_rotation(self.workingProjections[i], self.workingProjections[(i+self.num_angles//2)%self.num_angles])
                self.workingProjections[i] = rotate(self.workingProjections[i], -angle/2, reshape=False, mode='constant')
                self.workingProjections[(i+self.num_angles//2)%self.num_angles] = rotate(self.workingProjections[(i+self.num_angles//2)%self.num_angles], angle/2, reshape=False, mode='constant')
                
                self.tracked_rotations[i] += -angle/2
                self.tracked_rotations[(i+self.num_angles//2)%self.num_angles] += angle/2

                total_angle_rotation += abs(angle/2)
            
            # Calculate the average shift for this iteration
            average_angle_rotation = total_angle_rotation / (self.num_angles//2)
            print(f"Average degree rotation of iteration {iteration}: {average_angle_rotation}")

            # Check if the average shift is below the tolerance
            if average_angle_rotation < tolerance:
                print(f'Convergence reached after {iteration + 1} iterations.')
                break
        print(f'Maximum iterations reached without convergence.')

    def unrotate(self):
        for i in tqdm(range(self.num_angles//2), desc=f'Un-rotate image'):
            self.finalProjections[i] = rotate(self.finalProjections[i], -self.tracked_rotations[i], reshape=False, mode='constant')
            self.finalProjections[(i+self.num_angles//2)%self.num_angles] = rotate(self.finalProjections[(i+self.num_angles//2)%self.num_angles], -self.tracked_rotations[(i+self.num_angles//2)%self.num_angles], reshape=False, mode='constant')
            

    def PMA(self, iterations = 10):
        m = self.workingProjections
        scale = m.max()
        m = m / scale
        for k in tqdm(range(iterations), desc = 'PMA Algorithm iterations'):
            m = self.workingProjections
            recon_iterated = tomopy.recon(tomo=m, theta=self.ang, algorithm='sirt')
            iterated = tomopy.project(recon_iterated, self.ang, pad=False)
            iterated = iterated / iterated.max()
            for i in range(self.num_angles):

                img1 = iterated[i][200:-50,120:-120]
                img2 = m[i][200:-50,120:-120]
                # img1 = iterated[i]
                # img2 = m[i]
                num_rows, num_cols = img1.shape

                # Compute the cross-correlation between two consecutive images
                correlation = correlate(img1, img2, mode='same')
                
                # Find the index of the maximum correlation value
                y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                # Calculate the shifts, considering the center of the images as the origin
                y_shift -= num_rows // 2
                x_shift -= num_cols // 2

                print(x_shift, y_shift)

                # Apply the calculated shift to align the images
                self.workingProjections[i % self.num_angles] = subpixel_shift(self.workingProjections[i % self.num_angles], y_shift, x_shift)

                # Store the shifts
                self.tracked_shifts[i % self.num_angles][0] += y_shift
                self.tracked_shifts[i % self.num_angles][1] += x_shift
        m = m * scale
        self.workingProjections = m

    def vertical_mass_fluctuation_align(self, numIterations=5):
        """
        Aligns 2D projection images vertically based on their center-of-mass fluctuations by cross-correlating each projection with the reference projection and shifting accordingly.
        The iteration stops when the average shift in pixels is less than 0.5.
        """
        print("Vertical Mass Fluctuation Alignment")
        
        # for i in range(numIterations):
        #     sums = []
        #     shifts = []
            
        #     for k in range(self.num_angles):
        #         sums.append(np.sum(self.workingProjections[k][200:-50,120:-120], axis=1).tolist())
        #         if k > 0:
        #             CC = sp.signal.correlate(sums[0], sums[k], mode='same', method='fft')
        #             maxpoint = np.where(CC == CC.max())
        #             yshift = int(self.image_size[0] / 2 - maxpoint[0])
        #             self.workingProjections[k] = subpixel_shift(self.workingProjections[k], -yshift, 0)
        #             self.tracked_shifts[k, 0] += yshift
        #             shifts.append(abs(yshift))
            
        #     # Check for convergence
        #     if len(shifts) > 0:
        #         average_shift = np.mean(shifts)
        #         print(f"Iteration {i+1}: Average shift = {average_shift:.2f} pixels")
        #         if average_shift < 0.5:
        #             print("Convergence reached: Average shift is less than 0.5 pixels.")
        #             break
        #     else:
        #         print("No shifts calculated, possibly on the first iteration.")
        

        #Equalize with opposite angle
        print("Matching height of opposite angles")
        sums = []
        shifts = []
        
        for k in range(self.num_angles):
            sums.append(np.sum(self.workingProjections[k], axis=1).tolist())
        for i in range(self.num_angles//2):
            CC = sp.signal.correlate(sums[i], sums[(i+self.num_angles//2)%self.num_angles], mode='same', method='fft')
            maxpoint = np.where(CC == CC.max())
            yshift = int(self.image_size[0] / 2 - maxpoint[0])
            self.workingProjections[i] = subpixel_shift(self.workingProjections[i], -yshift/2, 0)
            self.workingProjections[(i+self.num_angles//2)%self.num_angles] = subpixel_shift(self.workingProjections[(i+self.num_angles//2)%self.num_angles], yshift/2, 0)
            self.tracked_shifts[k, 0] += yshift/2
            self.tracked_shifts[(i+self.num_angles//2)%self.num_angles,0] -= yshift/2
            shifts.append(abs(yshift))


    def tomopy_align(self, iterations = 15, alg = "sirt"):
        """
        Aligns projections using tomopy's join repojection Alignment algorithm
        """
        print("Tomopy Joint Reprojection Alignment of Projections (" + str(iterations) + " iterations)")

        center = tomopy.find_center_vo(self.workingProjections)
        # align_info = tomopy.prep.alignment.align_joint(self.workingProjections[:,200:-50,120:-120], self.ang, algorithm='sirt', iters=iterations, debug=True)[1:3]
        proj, sy, sx, _ = tomopy.prep.alignment.align_joint(self.workingProjections, self.ang, algorithm='sirt', iters=iterations, center = center, debug=True)

        print(sy)
        print(sx)

        self.tracked_shifts[:,0] += sy
        self.tracked_shifts[:,1] += sx


        self.workingProjections = tomopy.shift_images(proj, sy, sx)


    def optical_flow_align(self):
        """
        Aligns projections using optical flow to estimate the motion between consecutive images.

        WARNING: Does not have ability to be tracked my track_shifts
        """
        num_rows, num_cols = self.finalProjections[0].shape
        row_coords, col_coords = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        for m in tqdm(range(1, self.num_angles + 1), desc='Optical Flow Alignment of Projections'):
            # Handle circular indexing for the last projection
            prev_img = self.finalProjections[m - 1]
            current_img = self.finalProjections[m % self.num_angles]

            # Compute optical flow between two consecutive images
            v, u = optical_flow_tvl1(prev_img, current_img)

            # Apply the flow vectors to align the current image
            aligned_img = warp(current_img, np.array([row_coords + v, col_coords + u]), mode='constant')
            # aligned_img[60:,250:-250] = current_img[60:,250:-250]
            self.finalProjections[m % self.num_angles] = aligned_img
    
    def center_projections(self):
            """"
            This function determines and adjusts the center of rotation for 2D projection images by finding the initial center, shifting the projections to center them, and calculating any remaining offset.
            """
            print("Finding center of rotation for projections")
            self.rotation_center = tomopy.find_center_vo(self.workingProjections)
            print("Original center: {}".format(self.rotation_center))
            print("Center of frame: {}".format(self.image_size[1]//2))
            x_shift = (self.image_size[1]/2 - (self.rotation_center))
            y_shift = 0
            x_shift_check = 3
            if abs(x_shift_check) > 2:
                for m in tqdm(range(self.num_angles), desc='Center projections'):
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
                self.recon = tomopy.recon(self.workingProjections,
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
                self.recon = svmbir.recon(self.workingProjections, self.ang, center_offset = self.center_offset, verbose=1)
            else:
                self.recon = svmbir.recon(self.workingProjections, self.ang, center_offset = self.center_offset, snr_db=snr_db, verbose=1)
        else:
            print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            self.recon = tomopy.recon(self.workingProjections,
                                      self.ang,
                                      center=self.rotation_center,
                                      algorithm=algorithm,
                                    #   init_recon=tomo,
                                      sinogram_order=False)

        self.recon = tomopy.circ_mask(self.recon, axis=0, ratio=0.98)
        print("Reconstruction completed.")
