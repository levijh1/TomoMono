import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from skimage.transform import rotate
from helperFunctions import subpixel_shift
import tomopy
from skimage.restoration import denoise_bilateral
from skimage.registration import optical_flow_tvl1
from scipy.ndimage import map_coordinates
import torch
import matplotlib.pyplot as plt
from helperFunctions import MoviePlotter
from skimage.transform import warp



        
def bilateralFilter(tomo, d = 15, sigmaColor = 0.3, sigmaSpace = 100):
    # print("Bilateral filter being applied")
    # plt.imshow(tomo.workingProjections[0])
    # plt.colorbar()
    # plt.show()
    for i in tqdm(range(tomo.workingProjections.shape[0]), desc="Applying bilateral filter to projections"):
        tomo.workingProjections[i] = cv2.bilateralFilter(tomo.workingProjections[i], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    # plt.imshow(tomo.workingProjections[0])
    # plt.colorbar()
    # plt.show()


def cross_correlate_align(tomo, tolerance=1, max_iterations=15, stepRatio=1):
    """
    Aligns projections using cross-correlation to find the shift between consecutive images.
    Iterates until the average shift in pixels is less than the specified tolerance.
    """
    for iteration in tqdm(range(max_iterations), desc='Cross-Correlation Alignment Iterations'):
        total_shift = 0
        for m in tqdm(range(1, tomo.num_angles + 1), desc=f'Iteration {iteration + 1}'):
            # Handle circular indexing for the last projection
            img1 = tomo.workingProjections[m - 1][200:-100, 170:-170]
            img2 = tomo.workingProjections[m % tomo.num_angles][200:-100, 170:-170]
            num_rows, num_cols = img1.shape
    
            # Compute the cross-correlation between two consecutive images
            correlation = correlate(img1, img2, mode='same')
    
            # Find the index of the maximum correlation value
            y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    
            # Calculate the shifts, considering the center of the images as the origin
            y_shift -= num_rows // 2
            x_shift -= num_cols // 2
    
            y_shift *= stepRatio
            x_shift *= stepRatio
    
            # Apply the calculated shift to align the images
            tomo.workingProjections[m % tomo.num_angles] = subpixel_shift(tomo.workingProjections[m % tomo.num_angles], y_shift, x_shift)
    
            # Store the shifts
            tomo.tracked_shifts[m % tomo.num_angles][0] += y_shift
            tomo.tracked_shifts[m % tomo.num_angles][1] += x_shift
    
            # Accumulate the total shift magnitude for this iteration
            total_shift += np.sqrt(y_shift**2 + x_shift**2)
    
        # Calculate the average shift for this iteration
        average_shift = total_shift / tomo.num_angles
        print(f"Average pixel shift of iteration {iteration+1}: {average_shift}")
    
        # Check if the average shift is below the tolerance
        if average_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break
    print(f'Maximum iterations reached without convergence.')


def cross_correlate_tip(tomo, tolerance=0.5, max_iterations=15, stepRatio = 1):
    """
    Aligns projections using cross-correlation to find the shift between consecutive images.
    Iterates until the average shift in pixels is less than the specified tolerance.
    """
    for iteration in tqdm(range(max_iterations), desc='Cross-Correlation of pillar tip Iterations'):
        total_shift = 0
        for m in tqdm(range(1, tomo.num_angles + 1), desc=f'Iteration {iteration + 1}'):
            # Handle circular indexing for the last projection
            img1 = tomo.workingProjections[m - 1][200:300, 350:-350]
            img2 = tomo.workingProjections[m % tomo.num_angles][200:300, 350:-350]
            num_rows, num_cols = img1.shape

            # Compute the cross-correlation between two consecutive images
            correlation = correlate(img1, img2, mode='same')
            
            # Find the index of the maximum correlation value
            y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate the shifts, considering the center of the images as the origin
            y_shift -= num_rows // 2
            x_shift -= num_cols // 2

            y_shift *= stepRatio
            x_shift *= stepRatio

            # Apply the calculated shift to align the images
            tomo.workingProjections[m % tomo.num_angles] = subpixel_shift(tomo.workingProjections[m % tomo.num_angles], y_shift, x_shift)

            # Store the shifts
            tomo.tracked_shifts[m % tomo.num_angles][0] += y_shift
            tomo.tracked_shifts[m % tomo.num_angles][1] += x_shift

            # Accumulate the total shift magnitude for this iteration
            total_shift += np.sqrt(y_shift**2 + x_shift**2)
        
        # Calculate the average shift for this iteration
        average_shift = total_shift / tomo.num_angles
        print(f"Average pixel shift of iteration {iteration+1}: {average_shift}")

        # Check if the average shift is below the tolerance
        if average_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break
    print(f'Maximum iterations reached without convergence.')

def find_optimal_rotation(tomo, img1, img2, angle_range=[-5, 5], angle_step=0.25):
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
        rotated_img2 = rotate(np.copy(img2), angle, reshape=False, mode='wrap')
        
        # Compute the similarity (cross-correlation) between img1 and the rotated img2
        similarity = np.max(correlate(img1[200:-100,170:-170], rotated_img2[200:-100,170:-170], mode='same'))
        
        # Update the optimal angle if the current similarity is the highest found so far
        if similarity > max_similarity:
            max_similarity = similarity
            optimal_angle = angle

    # print(f"Optimal rotation angle: {optimal_angle} degrees, Maximum similarity: {max_similarity}")
    return optimal_angle, max_similarity
        
def rotate_correlate_align(tomo, max_iterations = 10, tolerance = 0.5):
    for iteration in tqdm(range(max_iterations), desc='Rotation Correlation Alignment Iterations'):
        total_angle_rotation = 0
        for i in tqdm(range(tomo.num_angles//2), desc=f'Iteration {iteration + 1}'):
            angle, maxSim = tomo.find_optimal_rotation(tomo.workingProjections[i], tomo.workingProjections[(i+tomo.num_angles//2)%tomo.num_angles])
            tomo.workingProjections[i] = rotate(tomo.workingProjections[i], -angle/2, reshape=False, mode='wrap')
            tomo.workingProjections[(i+tomo.num_angles//2)%tomo.num_angles] = rotate(tomo.workingProjections[(i+tomo.num_angles//2)%tomo.num_angles], angle/2, reshape=False, mode='wrap')
            
            tomo.tracked_rotations[i] += -angle/2
            tomo.tracked_rotations[(i+tomo.num_angles//2)%tomo.num_angles] += angle/2
    
            total_angle_rotation += abs(angle/2)
        
        # Calculate the average shift for this iteration
        average_angle_rotation = total_angle_rotation / (tomo.num_angles//2)
        print(f"Average degree rotation of iteration {iteration}: {average_angle_rotation}")
    
        # Check if the average shift is below the tolerance
        if average_angle_rotation < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break
    print(f'Maximum iterations reached without convergence.')

def unrotate(tomo):
    for i in tqdm(range(tomo.num_angles//2), desc=f'Un-rotate image'):
        tomo.finalProjections[i] = rotate(tomo.finalProjections[i], -tomo.tracked_rotations[i], reshape=False, mode='wrap')
        tomo.finalProjections[(i+tomo.num_angles//2)%tomo.num_angles] = rotate(tomo.finalProjections[(i+tomo.num_angles//2)%tomo.num_angles], -tomo.tracked_rotations[(i+tomo.num_angles//2)%tomo.num_angles], reshape=False, mode='wrap')



def PMA(tomo, max_iterations = 5, tolerance = 0.1, algorithm = 'art', crop_bottom_center_y = 500, crop_bottom_center_x = 750):
    """WARNING: Projections must be normalized for this method to work"""
    tomo.crop_bottom_center(crop_bottom_center_y, crop_bottom_center_x)
    tomo.standardize()
    tomo.center_projections() #This needs to happen so that the algorithm knows where the center is
    for k in tqdm(range(max_iterations), desc = 'PMA Algorithm iterations'):
        #Check which algorithm is being used
        if algorithm.endswith("CUDA"):
            if torch.cuda.is_available():
                # print("Using GPU-accelerated reconstruction.")
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': 400,
                    'extra_options': {}
                }
                recon_iterated = tomopy.recon(tomo.workingProjections,
                                        tomo.ang,
                                        center=tomo.rotation_center,
                                        algorithm=tomopy.astra,
                                        options=options,
                                        # init_recon=tomo,
                                        ncore=1)
            else: 
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            # print("Using SVMBIR-based reconstruction.")
            # print("center_offset assumed to be : {}".format(tomo.center_offset))
            recon_iterated = svmbir.recon(tomo.workingProjections, tomo.ang, center_offset = tomo.center_offset, verbose=1)
        else:
            # print("Using CPU-based reconstruction. Algorithm: ", algorithm)
            recon_iterated = tomopy.recon(tomo.workingProjections,
                                      tomo.ang,
                                      center=tomo.rotation_center,
                                      algorithm=algorithm,
                                    #   init_recon=tomo,
                                      sinogram_order=False
                                         )

        recon_iterated = tomopy.circ_mask(recon_iterated, axis=0, ratio=0.95)

        # MoviePlotter(recon_iterated)
        
        iterated = tomopy.project(recon_iterated, tomo.ang, pad=False)
        iterated_preScaling = iterated.copy()
        iterated = (iterated-np.mean(iterated)) / np.std(iterated)
        total_shift = 0
        total_x_shift = 0
        total_y_shift = 0
        for i in range(tomo.num_angles):
            imgTest = iterated_preScaling[i]
            img1 = iterated[i]
            img2 = tomo.workingProjections[i]

            # if i==0:
            #     print(img1.shape)
            #     # plt.imshow(imgTest)
            #     # plt.colorbar()
            #     # plt.show()
                
            #     plt.imshow(img1)
            #     plt.colorbar()
            #     plt.show()
                
            #     plt.imshow(img2)
            #     plt.colorbar()
            #     plt.show()
            
            num_rows, num_cols = img1.shape

            # Compute the cross-correlation between two consecutive images
            correlation = correlate(img1, img2, mode='same')
            
            # Find the index of the maximum correlation value
            y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate the shifts, considering the center of the images as the origin
            y_shift -= num_rows // 2
            x_shift -= num_cols // 2

            # Apply the calculated shift to align the images
            tomo.workingProjections[i % tomo.num_angles] = subpixel_shift(tomo.workingProjections[i % tomo.num_angles], y_shift, x_shift)

            # Store the shifts
            tomo.tracked_shifts[i % tomo.num_angles][0] += y_shift
            tomo.tracked_shifts[i % tomo.num_angles][1] += x_shift
            
            # Accumulate the total shift magnitude for this iteration
            total_shift += np.sqrt(y_shift**2 + x_shift**2)
            total_x_shift += np.abs(x_shift)
            total_y_shift += np.abs(y_shift)
        
        # Calculate the average shift for this iteration
        average_shift = total_shift / tomo.num_angles
        print(f"Average pixel shift of iteration {k}: {average_shift}")

        average_x_shift = total_x_shift / tomo.num_angles
        print(f"Average x shift of iteration {k}: {average_x_shift}")

        average_y_shift = total_y_shift / tomo.num_angles
        print(f"Average y shift of iteration {k}: {average_y_shift}")

        if average_shift < tolerance:
            print(f'Convergence reached after {k + 1} iterations.')
            break
    tomo.center_projections()


def vertical_mass_fluctuation_align(tomo, tolerance = 0.1, max_iterations=15):
    """
    Aligns 2D projection images vertically based on their center-of-mass fluctuations by cross-correlating each projection with the reference projection and shifting accordingly.
    The iteration stops when the average shift in pixels is less than 0.5.
    """
    print("Vertical Mass Fluctuation Alignment")



    for iteration in tqdm(range(max_iterations), desc="VMF Alignment Iterations"):
        #Equalize with opposite angle
        # print("Matching height of opposite angles")
        sums = []
        shifts = []

        total_shift = 0
    
        for k in range(tomo.num_angles):
            sums.append(np.sum(tomo.workingProjections[k], axis=1).tolist())
        for i in range(tomo.num_angles//2):
            CC = sp.signal.correlate(sums[i], sums[(i+tomo.num_angles//2)%tomo.num_angles], mode='same', method='fft')
            maxpoint = np.where(CC == CC.max())
            yshift = int(tomo.image_size[0] / 2 - maxpoint[0])
            tomo.workingProjections[i] = subpixel_shift(tomo.workingProjections[i], -yshift/2, 0)
            tomo.workingProjections[(i+tomo.num_angles//2)%tomo.num_angles] = subpixel_shift(tomo.workingProjections[(i+tomo.num_angles//2)%tomo.num_angles], yshift/2, 0)
            tomo.tracked_shifts[i, 0] += yshift/2
            tomo.tracked_shifts[(i+tomo.num_angles//2)%tomo.num_angles,0] -= yshift/2
            # shifts.append(abs(yshift))

            # Accumulate the total shift magnitude for this iteration
            total_shift += abs(yshift)
        
        # Calculate the average shift for this iteration
        average_shift = total_shift / tomo.num_angles
        print(f"Average pixel shift of iteration {iteration}: {average_shift}")

        # Check if the average shift is below the tolerance
        if average_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break


def tomopy_align(tomo, tolerance=0.1, max_iterations = 15, alg = "sirt"):
    """
    Aligns projections using tomopy's join repojection Alignment algorithm
    """
    print("Tomopy Joint Reprojection Alignment of Projections (" + str(max_iterations) + " iterations)")

    for iteration in tqdm(range(max_iterations), desc='Tomopy Joint Reprojeciton Align Iterations'):
        center = tomopy.find_center_vo(tomo.workingProjections)
        # align_info = tomopy.prep.alignment.align_joint(tomo.workingProjections[:,200:-50,120:-120], tomo.ang, algorithm='sirt', iters=iterations, debug=True)[1:3]
        proj, sy, sx, _ = tomopy.prep.alignment.align_joint(tomo.workingProjections, tomo.ang, algorithm='sirt', iters=1, center = center, debug=True)

        print("Shifts in y from tomopy_align")
        print(sy)
        print("Shifts in x from tomopy_align")
        print(sx)

        tomo.tracked_shifts[:,0] += sy
        tomo.tracked_shifts[:,1] += sx


        tomo.workingProjections = tomopy.shift_images(proj, sy, sx)

        # Calculate and return the overall average shift distance across all images
        avg_shifts = np.sqrt(sy**2 + sx**2)
        overall_avg_shift = np.mean(avg_shifts)
        print(f"Average pixel shift of tomopy_align for iteration {iteration}: {overall_avg_shift}")

        # Check if the average shift is below the tolerance
        if overall_avg_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break




def optical_flow_align(tomo):
    """
    Aligns projections using optical flow to estimate the motion between consecutive images.

    WARNING: Does not have ability to be tracked my track_shifts
    """
    print("Executing optical flow alignment")
    num_rows, num_cols = tomo.finalProjections[0].shape
    row_coords, col_coords = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    # for m in tqdm(range(1, tomo.num_angles + 1), desc='Optical Flow Alignment of Projections'):
    for m in tqdm(range(0,tomo.num_angles), desc='Optical Flow Alignment of Projections'):
        # Handle circular indexing for the last projection
        # prev_img = tomo.finalProjections[m-1]
        prev_img = tomo.finalProjections[(m + 1) % tomo.num_angles]
        # current_img = tomo.finalProjections[m % tomo.num_angles]
        current_img = tomo.finalProjections[m]

        # Compute optical flow between two consecutive images
        v, u = optical_flow_tvl1(prev_img, current_img)

        # Apply the flow vectors to align the current image
        aligned_img = warp(current_img, np.array([row_coords + v, col_coords + u]), mode='constant')
        # aligned_img[60:,250:-250] = current_img[60:,250:-250]
        tomo.finalProjections[m % tomo.num_angles] = aligned_img

def optical_flow_align_chill(tomo):
    """
    Aligns projections using optical flow to estimate the motion between consecutive images.

    WARNING: Does not have ability to be tracked my track_shifts
    """
    print("Executing optical flow alignment")
    num_rows, num_cols = tomo.finalProjections[0].shape
    row_coords, col_coords = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    for m in tqdm(range(1, tomo.num_angles + 1), desc='Chill Optical Flow Alignment of Projections'):
        # Handle circular indexing for the last projection
        # prev_img = tomo.finalProjections[m-1]
        prev_img = tomo.finalProjections[(m + 1) % tomo.num_angles]
        # current_img = tomo.finalProjections[m % tomo.num_angles]
        current_img = tomo.finalProjections[m]

        # Compute optical flow between two consecutive images
        v, u = optical_flow_tvl1(prev_img, current_img, 
                                   attachment=50,       # More smoothing
                                   tightness=0.1,       # Smaller shifts
                                   num_warp=3,          # Fewer opportunities for big shifts
                                   prefilter=True       # Suppress outliers
                                  )

        # Apply the flow vectors to align the current image
        aligned_img = warp(current_img, np.array([row_coords + v, col_coords + u]), mode='constant')
        # aligned_img[60:,250:-250] = current_img[60:,250:-250]
        tomo.finalProjections[m % tomo.num_angles] = aligned_img