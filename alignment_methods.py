import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from skimage.transform import rotate
from helperFunctions import subpixel_shift
import tomopy
from skimage.registration import optical_flow_tvl1
# import torch
from skimage.transform import warp
import cv2
import scipy as sp
import matplotlib.pyplot as plt

def bilateralFilter(tomo, d=15, sigmaColor=0.3, sigmaSpace=100):
    """
    Applies a bilateral filter to each projection image to reduce noise while preserving edges.

    Parameters:
    - tomo: Tomography object containing the projections.
    - d (int): Diameter of the pixel neighborhood used during filtering.
    - sigmaColor (float): Filter sigma in the color space (intensity differences).
    - sigmaSpace (float): Filter sigma in the coordinate space (spatial distance).
    """
    for i in tqdm(range(tomo.workingProjections.shape[0]), desc="Applying bilateral filter to projections"):
        tomo.workingProjections[i] = cv2.bilateralFilter(
            tomo.workingProjections[i], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
        )

def cross_correlate_align(tomo, tolerance=1, max_iterations=15, stepRatio=1, yROI_Range=[200, -100], xROI_Range=[170, -170]):
    """
    Aligns projection images by maximizing cross-correlation between consecutive slices.
    Iterates until the average shift per iteration is below the specified tolerance.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object with .workingProjections and .tracked_shifts.
    - tolerance (float): Convergence threshold for average pixel shift.
    - max_iterations (int): Maximum number of alignment iterations.
    - stepRatio (float): Scaling factor for the computed shifts.
    - yROI_Range (list): Range of y-coordinates to consider for correlation [start, end].
    - xROI_Range (list): Range of x-coordinates to consider for correlation [start, end].
    """
    print("Cross-Correlation Alignment")



    for iteration in tqdm(range(max_iterations), desc='Cross-Correlation Alignment Iterations'):
        total_shift = 0

        for m in tqdm(range(1, tomo.num_angles + 1), desc=f'Iteration {iteration + 1}'):
            # Handle circular indexing for the last projection
            if xROI_Range == None and yROI_Range == None:
                img1 = tomo.workingProjections[m - 1]
                img2 = tomo.workingProjections[m % tomo.num_angles]
            else:
                img1 = tomo.workingProjections[m - 1][yROI_Range[0]:yROI_Range[1], xROI_Range[0]:xROI_Range[1]]
                img2 = tomo.workingProjections[m % tomo.num_angles][yROI_Range[0]:yROI_Range[1], xROI_Range[0]:xROI_Range[1]]
            
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

def find_optimal_rotation(img1, img2, angle_range=[-5, 5], angle_step=0.25):
    """
    Finds the rotation angle between two projections that maximizes their similarity (cross-correlation).

    Parameters:
    - img1 (np.array): Reference image.
    - img2 (np.array): Image to be rotated.
    - angle_range (list): Range of angles to search over [min, max].
    - angle_step (float): Increment between angle steps.

    Returns:
    - optimal_angle (float): Rotation angle that gives the maximum similarity.
    - max_similarity (float): Maximum cross-correlation score found.
    """
    max_similarity = -100000
    optimal_angle = 0

    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        rotated_img2 = rotate(np.copy(img2), angle, reshape=False, mode='wrap')
        similarity = np.max(correlate(img1[200:-100, 170:-170], rotated_img2[200:-100, 170:-170], mode='same'))
        if similarity > max_similarity:
            max_similarity = similarity
            optimal_angle = angle

    return optimal_angle, max_similarity

def rotate_correlate_align(tomo, max_iterations=10, tolerance=0.5):
    """
    Aligns projections by correcting rotational misalignment using pairwise image rotation and cross-correlation.

    Parameters:
    - tomo: Tomography object with .workingProjections and .tracked_rotations.
    - max_iterations (int): Maximum number of alignment iterations.
    - tolerance (float): Average rotation threshold to consider convergence.
    """
    for iteration in tqdm(range(max_iterations), desc='Rotation Correlation Alignment Iterations'):
        total_angle_rotation = 0
        for i in tqdm(range(tomo.num_angles // 2), desc=f'Iteration {iteration + 1}'):
            angle, _ = tomo.find_optimal_rotation(
                tomo.workingProjections[i],
                tomo.workingProjections[(i + tomo.num_angles // 2) % tomo.num_angles]
            )
            tomo.workingProjections[i] = rotate(
                tomo.workingProjections[i], -angle / 2, reshape=False, mode='wrap'
            )
            tomo.workingProjections[(i + tomo.num_angles // 2) % tomo.num_angles] = rotate(
                tomo.workingProjections[(i + tomo.num_angles // 2) % tomo.num_angles], angle / 2, reshape=False, mode='wrap'
            )
            tomo.tracked_rotations[i] += -angle / 2
            tomo.tracked_rotations[(i + tomo.num_angles // 2) % tomo.num_angles] += angle / 2
            total_angle_rotation += abs(angle / 2)

        average_angle_rotation = total_angle_rotation / (tomo.num_angles // 2)
        print(f"Average degree rotation of iteration {iteration}: {average_angle_rotation}")

        if average_angle_rotation < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break
    print(f'Maximum iterations reached without convergence.')

def unrotate(tomo):
    """
    Reverses the rotational shifts stored in tracked_rotations and applies them to finalProjections.

    Parameters:
    - tomo: Tomography object with .finalProjections and .tracked_rotations.
    """
    for i in tqdm(range(tomo.num_angles // 2), desc='Un-rotate image'):
        tomo.finalProjections[i] = rotate(
            tomo.finalProjections[i], -tomo.tracked_rotations[i], reshape=False, mode='wrap'
        )
        tomo.finalProjections[(i + tomo.num_angles // 2) % tomo.num_angles] = rotate(
            tomo.finalProjections[(i + tomo.num_angles // 2) % tomo.num_angles],
            -tomo.tracked_rotations[(i + tomo.num_angles // 2) % tomo.num_angles],
            reshape=False, mode='wrap'
        )

def PMA(tomo, max_iterations=5, tolerance=0.1, algorithm='art', crop_bottom_center_y=500, crop_bottom_center_x=750, isPhaseData=False):
    """
    Performs Projection Matching Alignment (PMA) by comparing 2D projections to simulated projections of the current 3D reconstruction and minimizing differences.
    Projections must be normalized for this method to work.
    Automatically centers projections before and after running the algorithm.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object containing projections, angles, and reconstruction settings.
    - max_iterations (int): Number of PMA refinement iterations to perform.
    - tolerance (float): Average pixel shift to consider convergence.
    - algorithm (str): Reconstruction algorithm to use.
    - crop_bottom_center_y (int): Height for cropping bottom center.
    - crop_bottom_center_x (int): Width for cropping bottom center.
    - isPhaseData (bool): Whether the data is phase data, which may require sign inversion.
    """
    print("Projection Matching Alignment (PMA)")
    tomo.crop_bottom_center(crop_bottom_center_y, crop_bottom_center_x)
    tomo.standardize(isPhaseData=isPhaseData)
    tomo.center_projections()
    for k in tqdm(range(max_iterations), desc='PMA Algorithm iterations'):
        if algorithm.endswith("CUDA"):
            if torch.cuda.is_available():
                options = {
                    'proj_type': 'cuda',
                    'method': algorithm,
                    'num_iter': 400,
                    'extra_options': {}
                }
                recon_iterated = tomopy.recon(
                    tomo.workingProjections,
                    tomo.ang,
                    center=tomo.rotation_center,
                    algorithm=tomopy.astra,
                    options=options,
                    ncore=1
                )
            else:
                raise ValueError("GPU is not available, but the selected algorithm was 'gpu'.")
        elif algorithm == 'svmbir':
            recon_iterated = svmbir.recon(
                tomo.workingProjections, tomo.ang, center_offset=tomo.center_offset, verbose=1
            )
        else:
            recon_iterated = tomopy.recon(
                tomo.workingProjections,
                tomo.ang,
                center=tomo.rotation_center,
                algorithm=algorithm,
                sinogram_order=False
            )

        recon_iterated = tomopy.circ_mask(recon_iterated, axis=0, ratio=0.95)
        iterated = tomopy.project(recon_iterated, tomo.ang, pad=False)
        iterated_preScaling = iterated.copy()
        iterated = (iterated - np.mean(iterated)) / np.std(iterated)
        total_shift = 0
        total_x_shift = 0
        total_y_shift = 0
        for i in range(tomo.num_angles):
            imgTest = iterated_preScaling[i]
            img1 = iterated[i]
            img2 = tomo.workingProjections[i]
            num_rows, num_cols = img1.shape

            correlation = correlate(img1, img2, mode='same')
            y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
            y_shift -= num_rows // 2
            x_shift -= num_cols // 2

            tomo.workingProjections[i % tomo.num_angles] = subpixel_shift(
                tomo.workingProjections[i % tomo.num_angles], y_shift, x_shift
            )
            tomo.tracked_shifts[i % tomo.num_angles][0] += y_shift
            tomo.tracked_shifts[i % tomo.num_angles][1] += x_shift
            total_shift += np.sqrt(y_shift**2 + x_shift**2)
            total_x_shift += np.abs(x_shift)
            total_y_shift += np.abs(y_shift)

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

def vertical_mass_fluctuation_align(tomo, tolerance=0.1, max_iterations=15):
    """
    Aligns projection pairs at opposite angles by minimizing vertical center-of-mass differences.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object with projection data and shift tracking.
    - tolerance (float): Convergence threshold for average vertical shift.
    - max_iterations (int): Maximum number of iterations.
    """
    print("Vertical Mass Fluctuation Alignment")
    for iteration in tqdm(range(max_iterations), desc="VMF Alignment Iterations"):
        sums = []
        total_shift = 0
        for k in range(tomo.num_angles):
            sums.append(np.sum(tomo.workingProjections[k], axis=1).tolist())
        for i in range(tomo.num_angles // 2):
            CC = sp.signal.correlate(
                sums[i], sums[(i + tomo.num_angles // 2) % tomo.num_angles], mode='same', method='fft'
            )
            maxpoint = np.where(CC == CC.max())
            yshift = int(tomo.image_size[0] / 2 - maxpoint[0])
            tomo.workingProjections[i] = subpixel_shift(tomo.workingProjections[i], -yshift / 2, 0)
            tomo.workingProjections[(i + tomo.num_angles // 2) % tomo.num_angles] = subpixel_shift(
                tomo.workingProjections[(i + tomo.num_angles // 2) % tomo.num_angles], yshift / 2, 0
            )
            tomo.tracked_shifts[i, 0] += yshift / 2
            tomo.tracked_shifts[(i + tomo.num_angles // 2) % tomo.num_angles, 0] -= yshift / 2
            total_shift += abs(yshift)

        average_shift = total_shift / tomo.num_angles
        print(f"Average pixel shift of iteration {iteration}: {average_shift}")

        if average_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break

def tomopy_align(tomo, tolerance=0.1, max_iterations=15, alg="sirt"):
    """
    Uses TomoPy's joint reprojection algorithm to iteratively align all projections.

    Parameters:
    - tomo: Tomography object with projections, angles, and shift tracking.
    - tolerance (float): Convergence threshold for average shift.
    - max_iterations (int): Number of alignment iterations.
    - alg (str): TomoPy reconstruction algorithm to use (e.g., 'sirt').
    """
    print(f"Tomopy Joint Reprojection Alignment of Projections ({max_iterations} iterations)")
    for iteration in tqdm(range(max_iterations), desc='Tomopy Joint Reprojeciton Align Iterations'):
        center = tomopy.find_center_vo(tomo.workingProjections)
        proj, sy, sx, _ = tomopy.prep.alignment.align_joint(
            tomo.workingProjections, tomo.ang, algorithm='sirt', iters=1, center=center, debug=True
        )

        print("Shifts in y from tomopy_align")
        print(sy)
        print("Shifts in x from tomopy_align")
        print(sx)

        tomo.tracked_shifts[:, 0] += sy
        tomo.tracked_shifts[:, 1] += sx
        tomo.workingProjections = tomopy.shift_images(proj, sy, sx)

        avg_shifts = np.sqrt(sy**2 + sx**2)
        overall_avg_shift = np.mean(avg_shifts)
        print(f"Average pixel shift of tomopy_align for iteration {iteration}: {overall_avg_shift}")

        if overall_avg_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            break

def optical_flow_align(tomo):
    """
    Aligns each projection to the next using dense optical flow (TV-L1).
    WARNING: Does not update tracked_shifts.

    Parameters:
    - tomo: Tomography object containing finalProjections to align.

    Notes:
    - Updates tomo.finalProjections in place. Does not update tracked_shifts.
    """
    print("Executing optical flow alignment")
    num_rows, num_cols = tomo.finalProjections[0].shape
    row_coords, col_coords = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    for m in tqdm(range(0, tomo.num_angles), desc='Optical Flow Alignment of Projections'):
        prev_img = tomo.finalProjections[(m + 1) % tomo.num_angles]
        current_img = tomo.finalProjections[m]
        v, u = optical_flow_tvl1(prev_img, current_img)
        aligned_img = warp(current_img, np.array([row_coords + v, col_coords + u]), mode='constant')
        tomo.finalProjections[m % tomo.num_angles] = aligned_img