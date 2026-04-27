import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from skimage.transform import rotate
from helperFunctions import subpixel_shift
import tomopy
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from skimage.transform import warp, pyramid_gaussian
import cv2
import scipy as sp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from math import ceil
try:
    import torch
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        torch = None
except ImportError:
    torch = None

def compute_grad_image(image):
    gy, gx = np.gradient(image)
    return np.sqrt(gx**2 + gy**2)

def cross_correlate_align(
        tomo,
        tolerance=1,
        max_iterations=15,
        stepRatio=1,
        yROI_Range=[200, -100],
        xROI_Range=[170, -170],
        maxShiftTolerance=1,
        isFull360=False,
        num_images_for_median=None,
        upsample_factor=20,
        downsample=1,
        use_grad=False,
        ):
    """
    Aligns projection images by maximizing cross-correlation between consecutive slices.
    Iterates until the average shift per iteration is below the specified tolerance.

    Uses shift accumulation: each iteration computes all relative shifts from a snapshot
    of the current images, then applies cumulative absolute shifts from that snapshot.
    This avoids compounding interpolation errors from repeated in-place shifting.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object with .workingProjections and .tracked_shifts.
    - tolerance (float): Convergence threshold for average pixel shift per iteration.
    - max_iterations (int): Maximum number of alignment iterations.
    - stepRatio (float): Scaling factor for the computed shifts.
    - yROI_Range (list): [start, end] y-crop for correlation region.
    - xROI_Range (list): [start, end] x-crop for correlation region.
    - isFull360 (bool): If True, also correlates last image against first to close the loop.
    - num_images_for_median (int): If set, use a rolling median of the last K images as
      the reference instead of just the previous image. More robust to noisy projections.
    - upsample_factor (int): Passed to phase_cross_correlation for sub-pixel accuracy.
      upsample_factor=1 gives pixel-accurate shifts; higher values (e.g. 10, 100) give
      finer sub-pixel precision at increased compute cost.
    - downsample (int): Downscale factor applied before shift detection via pyramid_gaussian.
      Speeds up correlation on large images; detected shifts are rescaled back automatically.
      Use 1 (default) to skip downsampling.
    - use_grad (bool): If True, replace each reference image with its gradient magnitude before
      computing the cross-correlation. Gradient images are edge-enhanced, which can give
      more accurate shifts when features have strong edges but low overall contrast.
    """
    print("Cross-Correlation Alignment")

    n = tomo.num_angles
    K = num_images_for_median if (num_images_for_median is not None and num_images_for_median > 1) else None

    def _crop(img):
        if yROI_Range is not None and xROI_Range is not None:
            return img[yROI_Range[0]:yROI_Range[1], xROI_Range[0]:xROI_Range[1]]
        return img

    def _maybe_downsample(img):
        if downsample == 1:
            return img
        return tuple(pyramid_gaussian(img, max_layer=1, downscale=downsample, sigma=None, preserve_range=False))[1]

    def _compute_shift(ref, mov):
        ref_c = _crop(ref)
        mov_c = _crop(mov)
        if use_grad:
            ref_c = compute_grad_image(ref_c)
            mov_c = compute_grad_image(mov_c)
        if downsample != 1:
            ref_c = _maybe_downsample(ref_c)
            mov_c = _maybe_downsample(mov_c)
        shift_rc, _, _ = phase_cross_correlation(ref_c, mov_c, upsample_factor=upsample_factor)
        return shift_rc[0] * downsample * stepRatio, shift_rc[1] * downsample * stepRatio
    
    for iteration in range(max_iterations):
        snapshot = tomo.workingProjections.copy()
        rel_shifts = np.zeros((n, 2), dtype=np.float64)

        for i in tqdm(range(1, n), desc=f'Iteration {iteration + 1}/{max_iterations}'):
            if K is None:
                ref = snapshot[i - 1]
            else:
                ref = np.median(snapshot[max(0, i - K):i], axis=0)
            y_shift, x_shift = _compute_shift(ref, snapshot[i])
            rel_shifts[i] = [y_shift, x_shift]

        # Absolute shifts: image[i] needs the cumulative sum of all relative shifts up to i.
        # rel_shifts[0] stays zero (image[0] is the anchor), so cumsum gives:
        #   abs_shifts[i] = rel_shifts[1] + ... + rel_shifts[i]
        abs_shifts = np.cumsum(rel_shifts, axis=0)

        if isFull360:
            # Correlate last image against first to close the loop; shift image[0] to close drift.
            y_shift, x_shift = _compute_shift(snapshot[n - 1], snapshot[0])
            abs_shifts[0] = [y_shift, x_shift]

        # Apply cumulative shifts from the snapshot — one interpolation per image, no compounding.
        for i in range(1, n):
            tomo.workingProjections[i] = subpixel_shift(snapshot[i], abs_shifts[i, 0], abs_shifts[i, 1])
        if isFull360:
            tomo.workingProjections[0] = subpixel_shift(snapshot[0], abs_shifts[0, 0], abs_shifts[0, 1])

        tomo.tracked_shifts += abs_shifts

        shift_magnitudes = np.linalg.norm(rel_shifts[1:], axis=1)
        average_shift = shift_magnitudes.mean()
        max_shift = shift_magnitudes.max()
        print(f"Iteration {iteration + 1}: avg shift = {average_shift:.4f} px, max shift = {max_shift:.4f} px")

        if average_shift <= tolerance and max_shift <= maxShiftTolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            return

    print('Maximum iterations reached without convergence.')

def PMA(
        tomo,
        max_iterations=5,
        tolerance=0.1,
        algorithm='art',
        crop_bottom_center_y=500,
        crop_bottom_center_x=750,
        isPhaseData=False,
        standardize=True,
        levels=1,
        scale=2,
        iterations_per_level=None,
        upsample_factor=20,
        ):
    """
    Performs Projection Matching Alignment (PMA) using a multi-resolution pyramid.

    Runs alignment at progressively finer resolutions (coarse to fine). At each level,
    shifts are computed on downsampled images and upscaled back to full-resolution units.
    All levels accumulate into a single set of cumulative shifts applied from the original
    projections, avoiding chained interpolation errors across levels.

    Step-size regularization damps shifts by 0.2x when the max shift drops below 0.05 px,
    preventing overshoot near convergence.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object containing projections, angles, and reconstruction settings.
    - max_iterations (int): Default number of iterations per pyramid level.
    - tolerance (float): Average pixel shift threshold for early stopping within a level.
    - algorithm (str): Reconstruction algorithm ('art', 'gridrec', 'sirt', or CUDA variant).
    - crop_bottom_center_y/x (int): Spatial crop applied before alignment begins.
    - isPhaseData (bool): Whether data is phase contrast (affects standardization sign).
    - standardize (bool): Whether to standardize projections and reprojections before comparing.
    - levels (int): Number of pyramid levels. 1 = full resolution only (original behaviour).
      level 0 = full res, level 1 = scale^1 downsampled, etc. Runs coarse → fine.
    - scale (int): Downscale factor between pyramid levels (e.g. 2 → 2x, 4x, 8x, ...).
    - iterations_per_level (list): Per-level iteration counts ordered coarse → fine.
      Length must equal `levels`. If None, uses `max_iterations` at every level.
      Example: levels=3, iterations_per_level=[3, 3, 5] runs 3 iters at 4x, 3 at 2x, 5 at 1x.
    """
    print("Projection Matching Alignment (PMA)")
    ratio = 0.95

    tomo.crop_bottom_center(crop_bottom_center_y, crop_bottom_center_x)
    if standardize:
        tomo.standardize(isPhaseData=isPhaseData)
    tomo.center_projections()

    if iterations_per_level is None:
        iters_per_level = [max_iterations] * levels
    else:
        iters_per_level = list(iterations_per_level)
        assert len(iters_per_level) == levels, "iterations_per_level must have one entry per level"

    def _reconstruct(projs, center):
        if algorithm.endswith("CUDA"):
            if torch is None:
                raise ValueError("GPU requested but torch is unavailable.")
            options = {'proj_type': 'cuda', 'method': algorithm, 'num_iter': 400, 'extra_options': {}}
            recon = tomopy.recon(projs, tomo.ang, center=center, algorithm=tomopy.astra, options=options, ncore=1)
        elif algorithm == 'svmbir':
            recon = svmbir.recon(projs, tomo.ang, center_offset=tomo.center_offset, verbose=1)
        else:
            recon = tomopy.recon(projs, tomo.ang, center=center, algorithm=algorithm, sinogram_order=False)
        return tomopy.circ_mask(recon, axis=0, ratio=ratio)

    def _compute_shift(ref, mov, cropping):
        ref_c = ref[:, cropping:-cropping] if cropping > 0 else ref
        mov_c = mov[:, cropping:-cropping] if cropping > 0 else mov
        shift_rc, _, _ = phase_cross_correlation(ref_c, mov_c, upsample_factor=upsample_factor)
        return shift_rc[0], shift_rc[1]

    # Snapshot taken after preprocessing; all pyramid levels shift from this base.
    original = tomo.workingProjections.copy()
    pma_shifts = np.zeros((tomo.num_angles, 2), dtype=np.float64)

    for level_idx, level in enumerate(reversed(range(levels))):
        downsample_factor = scale ** level
        n_iters = iters_per_level[level_idx]
        print(f"\n--- PMA Level {level} ({downsample_factor}x downsampled, {n_iters} iterations) ---")

        # Apply current cumulative PMA shifts from original — no chained interpolation.
        current_projs = np.stack([
            subpixel_shift(original[i], pma_shifts[i, 0], pma_shifts[i, 1])
            for i in range(tomo.num_angles)
        ])

        # Spatially downsample for this pyramid level.
        if level > 0:
            scaled_projs = list(pyramid_gaussian(
                current_projs, downscale=scale, max_layer=level, channel_axis=0
            ))[level].astype(np.float32)
        else:
            scaled_projs = current_projs.astype(np.float32)

        scaled_center = tomo.rotation_center / downsample_factor
        ny_scaled = scaled_projs.shape[1]
        cropping = ceil((1 - ratio) * ny_scaled / 2)

        level_shifts = np.zeros((tomo.num_angles, 2), dtype=np.float64)

        for k in tqdm(range(n_iters), desc=f'PMA Level {level} iterations'):
            recon = _reconstruct(scaled_projs, scaled_center)
            reproj = tomopy.project(recon, tomo.ang, pad=False)
            if standardize:
                reproj = (reproj - np.mean(reproj)) / np.std(reproj)

            dy = np.zeros(tomo.num_angles)
            dx = np.zeros(tomo.num_angles)
            for i in range(tomo.num_angles):
                dy[i], dx[i] = _compute_shift(reproj[i], scaled_projs[i], cropping)

            # Step-size regularization: damp when near convergence to avoid overshoot.
            dxmax = np.max(np.abs(dx))
            dymax = np.max(np.abs(dy))
            alpha = 0.2 if max(dxmax, dymax) < 0.05 else 1.0
            dy *= alpha
            dx *= alpha
            if alpha < 1.0:
                print(f"  Regularized alpha={alpha} (max: dx={dxmax:.3f}, dy={dymax:.3f})")

            for i in range(tomo.num_angles):
                scaled_projs[i] = subpixel_shift(scaled_projs[i], dy[i], dx[i])
            level_shifts[:, 0] += dy
            level_shifts[:, 1] += dx

            avg_shift = np.mean(np.sqrt((dy* downsample_factor)**2 + (dx*downsample_factor)**2))
            print(f"  Iter {k+1}: avg={avg_shift:.4f} px (x:{np.mean(np.abs(dx*downsample_factor)):.4f}, y:{np.mean(np.abs(dy*downsample_factor)):.4f})")

            if avg_shift*downsample_factor < tolerance:
                print(f"  Convergence at level {level} after {k+1} iterations.")
                break

        # Upscale shifts to full-resolution pixel units before accumulating.
        pma_shifts += level_shifts * downsample_factor

    # Apply all accumulated PMA shifts in one pass from the original (no compounding).
    for i in range(tomo.num_angles):
        tomo.workingProjections[i] = subpixel_shift(original[i], pma_shifts[i, 0], pma_shifts[i, 1])
    tomo.tracked_shifts += pma_shifts
    print("\nPMA complete.")

def vertical_mass_fluctuation_align(
        tomo,
        tolerance=0.1,
        max_iterations=15,
        y_range=None,
        sigma=2.0,
        upsample_factor=100,
        smooth_sigma=None,
        ):
    """
    Aligns projections vertically by correlating high-pass-filtered column-sum profiles.

    Source: Odstrčil. Alignment methods for nanotomography with deep subpixel accuracy.
    https://doi.org/10.1364/oe.27.036637

    Parameters:
    - tomo: Tomography object with projection data and shift tracking.
    - tolerance (float): Convergence threshold for average vertical shift per iteration.
    - max_iterations (int): Maximum number of iterations.
    - y_range (list): Optional [start, end] row crop applied before summing columns.
    - sigma (float): Gaussian sigma for high-pass filtering each mass profile
      (profile = raw_sum - gaussian_filter(raw_sum, sigma)).
    - upsample_factor (int): Sub-pixel precision for phase_cross_correlation (e.g. 100 = 0.01 px).
    - smooth_sigma (float): If set, smooths detected shifts across angles with gaussian_filter1d
      before applying them, suppressing noisy frame-to-frame outliers.
    """
    print("Vertical Mass Fluctuation Alignment")

    n = tomo.num_angles

    for iteration in tqdm(range(max_iterations), desc="VMF Alignment Iterations"):
        snapshot = tomo.workingProjections.copy()

        # Build high-pass-filtered column-sum profiles for each projection.
        # Small sigma removes only pixel noise; the broad mass peak is preserved for correlation.
        profiles_list = []
        for k in range(n):
            img = snapshot[k] if y_range is None else snapshot[k][y_range[0]:y_range[1]]
            m = np.sum(img, axis=1).astype(np.float64)
            profiles_list.append(m - gaussian_filter(m, sigma=sigma))
        profiles = np.array(profiles_list)

        ref = profiles.mean(axis=0)

        # Detect sub-pixel vertical shifts via phase cross-correlation
        shifts_y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d_y, _, _ = phase_cross_correlation(ref, profiles[i], upsample_factor=upsample_factor)
            shifts_y[i] = d_y[0]

        # Optionally smooth shifts across angles to suppress outliers
        if smooth_sigma is not None and smooth_sigma > 0:
            shifts_y = gaussian_filter1d(shifts_y, sigma=smooth_sigma)

        # Apply all shifts from the snapshot in one pass (no compounding interpolation errors)
        for i in range(n):
            tomo.workingProjections[i] = subpixel_shift(snapshot[i], shifts_y[i], 0)
        tomo.tracked_shifts[:, 0] += shifts_y

        average_shift = np.mean(np.abs(shifts_y))
        print(f"Iteration {iteration + 1}: avg shift = {average_shift:.4f} px")

        if average_shift < tolerance:
            print(f'Convergence reached after {iteration + 1} iterations.')
            return

    print('Maximum iterations reached without convergence.')

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


def sinogram_consistency_score(tomo, plot=True, bg_percentile=None):
    """
    Quantifies alignment quality using the Helgason-Ludwig consistency conditions:

      x_cm(θ) = A·cos(θ) + B·sin(θ) + C   (horizontal CM follows a sinusoid)
      y_cm(θ) = constant                    (vertical CM is invariant to rotation)

    x_cm deviation captures horizontal misalignment (what XCA corrects).
    y_cm deviation captures vertical drift (what VMF corrects).
    Both are combined into a single overall RMSE.

    Parameters:
    - tomo: Tomography object with .workingProjections and .ang (angles in radians).
    - plot (bool): If True, plots both CM trajectories with fits and residuals.
    - bg_percentile (float or None): If set (e.g. 10), subtracts the Nth percentile of
      each projection as a background estimate before computing the CM. Improves accuracy
      when bright background pulls the CM away from the sample signal.

    Returns:
    - combined_rmse (float): RMS of x and y RMSE combined — the single overall score.
    - x_rmse (float): RMSE of x_cm vs best-fit sinusoid (horizontal alignment, pixels).
    - y_rmse (float): RMSE of y_cm vs its mean (vertical alignment, pixels).
    - x_cm, y_cm (ndarray): Measured CM trajectories, shape (n,).
    """
    n = tomo.num_angles
    angles = tomo.ang.ravel()
    ny, nx = tomo.workingProjections.shape[1], tomo.workingProjections.shape[2]
    x_coords = np.arange(nx, dtype=np.float64)
    y_coords = np.arange(ny, dtype=np.float64)

    x_cm = np.zeros(n)
    y_cm = np.zeros(n)

    for i in range(n):
        img = tomo.workingProjections[i].astype(np.float64)

        # Background subtraction: remove Nth percentile intensity before weighting.
        # Prevents bright background from dominating the center-of-mass estimate.
        if bg_percentile is not None:
            bg = np.percentile(img, bg_percentile)
            img = np.clip(img - bg, 0, None)

        col_sums = img.sum(axis=0)  # (nx,) — mass at each column position
        row_sums = img.sum(axis=1)  # (ny,) — mass at each row position
        total = col_sums.sum()

        if total > 0:
            x_cm[i] = (x_coords * col_sums).sum() / total
            y_cm[i] = (y_coords * row_sums).sum() / total
        else:
            x_cm[i] = nx / 2.0
            y_cm[i] = ny / 2.0

    # x_cm consistency: fit sinusoid A·cos(θ) + B·sin(θ) + C
    design = np.column_stack([np.cos(angles), np.sin(angles), np.ones(n)])
    coeffs_x, _, _, _ = np.linalg.lstsq(design, x_cm, rcond=None)
    x_fit = design @ coeffs_x
    x_residuals = x_cm - x_fit
    x_rmse = np.sqrt(np.mean(x_residuals ** 2))
    ss_tot_x = np.sum((x_cm - x_cm.mean()) ** 2)
    r2_x = 1.0 - np.sum(x_residuals ** 2) / ss_tot_x if ss_tot_x > 0 else 1.0

    # y_cm consistency: should be constant across angles (invariant to rotation)
    y_fit = np.full(n, y_cm.mean())
    y_residuals = y_cm - y_fit
    y_rmse = np.sqrt(np.mean(y_residuals ** 2))
    ss_tot_y = np.sum((y_cm - y_cm.mean()) ** 2)
    r2_y = 1.0 - np.sum(y_residuals ** 2) / ss_tot_y if ss_tot_y > 0 else 1.0

    # Combined score: quadrature sum of both directions
    combined_rmse = np.sqrt((x_rmse ** 2 + y_rmse ** 2) / 2)

    print(f"Sinogram consistency:")
    print(f"  x_cm (horizontal) — RMSE: {x_rmse:.4f} px  |  R²: {r2_x:.6f}")
    print(f"  y_cm (vertical)   — RMSE: {y_rmse:.4f} px  |  R²: {r2_y:.6f}")
    print(f"  Combined RMSE:       {combined_rmse:.4f} px")

    if plot:
        angles_deg = np.rad2deg(angles)
        order = np.argsort(angles_deg)
        ad = angles_deg[order]

        fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
        fig.suptitle(f'Sinogram Consistency  —  Combined RMSE={combined_rmse:.4f} px', fontsize=12)

        axes[0, 0].plot(ad, x_cm[order], '.', markersize=3, label='x_cm')
        axes[0, 0].plot(ad, x_fit[order], '-', linewidth=1.5, label='Sinusoid fit')
        axes[0, 0].set_ylabel('x center of mass (px)')
        axes[0, 0].set_title(f'Horizontal  RMSE={x_rmse:.4f} px  R²={r2_x:.4f}')
        axes[0, 0].legend(markerscale=3)

        axes[1, 0].plot(ad, x_residuals[order], '.', markersize=3, color='tomato')
        axes[1, 0].axhline(0, color='k', linewidth=0.8, linestyle='--')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Residual (px)')

        axes[0, 1].plot(ad, y_cm[order], '.', markersize=3, color='steelblue', label='y_cm')
        axes[0, 1].axhline(y_cm.mean(), color='orange', linewidth=1.5, label='Mean (expected)')
        axes[0, 1].set_ylabel('y center of mass (px)')
        axes[0, 1].set_title(f'Vertical  RMSE={y_rmse:.4f} px  R²={r2_y:.4f}')
        axes[0, 1].legend(markerscale=3)

        axes[1, 1].plot(ad, y_residuals[order], '.', markersize=3, color='tomato')
        axes[1, 1].axhline(0, color='k', linewidth=0.8, linestyle='--')
        axes[1, 1].set_xlabel('Angle (degrees)')
        axes[1, 1].set_ylabel('Residual (px)')

        plt.tight_layout()
        plt.show()

    return combined_rmse, x_rmse, y_rmse, x_cm, y_cm














    