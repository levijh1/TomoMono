"""
Legacy / less-successful alignment methods.

These functions are kept for reference and reproducibility, but are not
recommended for routine use. They have not been updated to current standards
(some carry pre-existing bugs) — modify only with care.

Contained methods:
    tomopy_align            — TomoPy joint reprojection
    optical_flow_align      — Dense TV-L1 optical flow (does not update tracked_shifts)
    shift_min_to_middle     — Coarse centering by argmin
    bilateralFilter         — OpenCV bilateral filter (denoising, not alignment)
    find_optimal_rotation   — Brute-force angle scan
    rotate_correlate_align  — Iterative rotation alignment
    unrotate                — Reverse tracked rotations onto finalProjections
"""

import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from skimage.transform import rotate, warp
from skimage.registration import optical_flow_tvl1
import tomopy
import cv2

from helperFunctions import subpixel_shift


def tomopy_align(tomo, tolerance=0.1, max_iterations=15, alg="sirt", crop_bottom_center_y=500, crop_bottom_center_x=750, isPhaseData = False):
    """
    Uses TomoPy's joint reprojection algorithm to iteratively align all projections.

    Parameters:
    - tomo: Tomography object with projections, angles, and shift tracking.
    - tolerance (float): Convergence threshold for average shift.
    - max_iterations (int): Number of alignment iterations.
    - alg (str): TomoPy reconstruction algorithm to use (e.g., 'sirt').
    """
    print(f"Tomopy Joint Reprojection Alignment of Projections ({max_iterations} iterations)")
    tomo.crop_bottom_center(crop_bottom_center_y, crop_bottom_center_x)
    if standardize:
        tomo.standardize(isPhaseData=isPhaseData)
    tomo.center_projections()
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

def shift_min_to_middle(tomo):
    print("Shifting min values to middle")
    n_images, height, width = tomo.workingProjections.shape
    center_x = width // 2  # middle of array

    for m in tqdm(range(n_images), desc="Shifting min to middle"):
        img = tomo.workingProjections[m]

        # Find index of minimum value
        min_idx = np.unravel_index(np.argmin(img), img.shape)
        min_y, min_x = min_idx

        # Compute shift (positive means left/up)
        x_shift = (center_x - min_x)

        tomo.tracked_shifts[m % tomo.num_angles][1] += x_shift

        tomo.workingProjections[m % tomo.num_angles] = subpixel_shift(tomo.workingProjections[m % tomo.num_angles], 0, x_shift)


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
