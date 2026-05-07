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
from matplotlib import gridspec

try:
    import torch
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        torch = None
except ImportError:
    torch = None

try:
    import cupy as cp
    cp.array([1])  # real allocation — raises if GPU is unavailable or busy
    from cupyx.scipy.ndimage import gaussian_filter as _gaussian_filter
    xp = cp
except Exception:
    cp = None
    _gaussian_filter = gaussian_filter
    xp = np

def compute_grad_image(image):
    arr = xp.asarray(image)
    gy, gx = xp.gradient(arr)
    result = xp.hypot(gx, gy)  # equivalent to sqrt(gx²+gy²) but avoids allocating gx², gy², and their sum as separate temporaries
    return result.get() if xp is not np else result

def cross_correlate_align(
        tomo,
        tolerance=1,
        max_iterations=15,
        stepRatio=1,
        yROI_Range=None,
        xROI_Range=None,
        maxShiftTolerance=1,
        isFull360=False,
        num_images_for_median=None,
        upsample_factor=20,
        downsample=1,
        use_grad=False,
        plot=False,
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
    roi_str = f"ROI y={yROI_Range} x={xROI_Range}" if (yROI_Range is not None and xROI_Range is not None) else "full frame"
    ds_str = f"{downsample}x downsample" if downsample > 1 else "full resolution"
    grad_str = "gradient mode" if use_grad else "intensity mode"
    print(f"Cross-Correlation Alignment  [{ds_str} | {roi_str} | {grad_str}]")

    n = tomo.num_angles
    K = num_images_for_median if (num_images_for_median is not None and num_images_for_median > 1) else None

    def _crop(img):
        if yROI_Range is not None and xROI_Range is not None:
            return img[yROI_Range[0]:yROI_Range[1], xROI_Range[0]:xROI_Range[1]]
        return img

    def _downsample(img):
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
            ref_c = _downsample(ref_c)
            mov_c = _downsample(mov_c)
        shift_rc, _, _ = phase_cross_correlation(ref_c, mov_c, upsample_factor=upsample_factor)
        return shift_rc[0] * downsample * stepRatio, shift_rc[1] * downsample * stepRatio
    
    for iteration in range(max_iterations):
        snapshot = tomo.workingProjections.copy()
        rel_shifts = np.zeros((n, 2), dtype=np.float64)
        _plot_data = None

        for i in tqdm(range(1, n), desc=f'Iteration {iteration + 1}/{max_iterations}'):
            if K is None:
                ref = snapshot[i - 1]
            else:
                ref = np.median(snapshot[max(0, i - K):i], axis=0)
            y_shift, x_shift = _compute_shift(ref, snapshot[i])
            rel_shifts[i] = [y_shift, x_shift]
            if plot and i == n//2:
                _plot_data = (snapshot[n//2].copy(), ref.copy(), y_shift, x_shift)

        if plot and _plot_data is not None and iteration == 0:
            mov_img, ref_img, y_sh, x_sh = _plot_data
            print(f"  Projection {n//2} shift: y={y_sh:.4f} px, x={x_sh:.4f} px")
            vmin = min(ref_img.min(), mov_img.min())
            vmax = max(ref_img.max(), mov_img.max())
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Cross-Correlation Alignment — Iteration {iteration + 1}  |  Proj {n//2} shift: y={y_sh:.4f}, x={x_sh:.4f} px")
            im0 = axes[0].imshow(mov_img, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Projection {n//2} (being aligned)")
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(ref_img, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Projection {n//2} (reference)")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            if xROI_Range is not None and yROI_Range is not None:
                for ax in axes:
                    ax.add_patch(plt.Rectangle(
                        (xROI_Range[0], yROI_Range[0]),
                        xROI_Range[1] - xROI_Range[0],
                        yROI_Range[1] - yROI_Range[0],
                        linewidth=2, edgecolor='red', facecolor='none'
                    ))
            plt.tight_layout()
            plt.show()

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






def _highpass(img, sigma):
    arr = xp.asarray(img, dtype=xp.float64)
    result = arr - _gaussian_filter(arr, sigma)
    return result.get() if xp is not np else result

def _fourier_gradients(img):
    ny, nx = img.shape
    arr = xp.asarray(img)
    F  = xp.fft.fft2(arr)
    ux = xp.fft.fftfreq(nx).reshape(1, -1)
    uy = xp.fft.fftfreq(ny).reshape(-1, 1)
    dy = xp.real(xp.fft.ifft2(2j * np.pi * uy * F))
    dx = xp.real(xp.fft.ifft2(2j * np.pi * ux * F))
    if xp is not np:
        return dy.get(), dx.get()
    return dy, dx

# def _optical_flow_shift(proj, reproj_img, sigma):
#     """Estimate (dy, dx) via Lucas-Kanade optical flow on high-pass filtered images.

#     Minimizes ||r - dy·∂p̂/∂y - dx·∂p̂/∂x||² where r = proj - reproj_img.
#     The closed-form solution is  d = Σ(hp(∂p̂/∂d) · hp(r)) / Σ(hp(∂p̂/∂d)²),
#     so the denominator is gradient energy of the reprojection, not residual energy.
#     """
#     r_hp             = _highpass(proj - reproj_img, sigma)
#     dy_grad, dx_grad = _fourier_gradients(reproj_img)
#     hp_dy            = _highpass(dy_grad, sigma)
#     hp_dx            = _highpass(dx_grad, sigma)
#     dy = np.sum(hp_dy * r_hp) / (np.sum(hp_dy ** 2) + 1e-8)
#     dx = np.sum(hp_dx * r_hp) / (np.sum(hp_dx ** 2) + 1e-8)
#     return float(dy), float(dx)


def _pma_reconstruct(projs, ang, center, algorithm, ratio=0.99):
    if algorithm.endswith("CUDA"):
        if torch is None:
            raise ValueError("GPU requested but torch is unavailable.")
        options = {'proj_type': 'cuda', 'method': algorithm, 'num_iter': 400, 'extra_options': {}}
        recon = tomopy.recon(projs, ang, center=center, algorithm=tomopy.astra, options=options, ncore=1)
    else:
        recon = tomopy.recon(projs, ang, center=center, algorithm=algorithm, sinogram_order=False)
    return tomopy.circ_mask(recon, axis=0, ratio=ratio)

def _cross_correlation_shift(ref, mov, upsample_factor):
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample_factor)
    return float(shift[0]), float(shift[1])

# --- NEW: Matching preprocess (easy to disable/remove) ---
def _preprocess_for_matching(img, sigma=2):
    img = img.astype(np.float64)
    img = img - gaussian_filter(img, sigma)  # high-pass
    img /= (np.std(img) + 1e-8)
    return img


def _optical_flow_shift(proj, reproj_img, sigma):
    """Improved Lucas-Kanade optical flow with full 2x2 system."""
    r_hp = _highpass(proj - reproj_img, sigma)

    dy_grad, dx_grad = _fourier_gradients(reproj_img)
    hp_dy = _highpass(dy_grad, sigma)
    hp_dx = _highpass(dx_grad, sigma)

    # --- NEW: full 2x2 system ---
    A11 = np.sum(hp_dy * hp_dy)
    A22 = np.sum(hp_dx * hp_dx)
    A12 = np.sum(hp_dy * hp_dx)

    b1 = np.sum(hp_dy * r_hp)
    b2 = np.sum(hp_dx * r_hp)

    det = A11 * A22 - A12 * A12 + 1e-8

    dy = ( A22 * b1 - A12 * b2) / det
    dx = (-A12 * b1 + A11 * b2) / det

    return float(dy), float(dx)


def PMA(
        tomo,
        max_iterations=5,
        tolerance=0.1,
        algorithm='art',
        xROI_Range=None,
        yROI_Range=None,
        isPhaseData=False,
        standardize=False,
        levels=1,
        scale=2,
        iterations_per_level=None,
        upsample_factor=20,
        shift_method='cross_correlation',
        of_sigma=3.0,
        smooth_sigma=None,
        plot=False,

        use_matching_preprocess=True,
        matching_sigma=2,
        use_grad=False,

        max_step=0.5,
        stepRatio=1,
        ):
    """
    PMA with ROI-based alignment (non-destructive cropping).
    """
    recon_crop_ratio = 0.99
    grad_str = " | gradient mode" if use_grad else ""
    preprocess_str = " | matching_preprocess" if use_matching_preprocess else ""
    print(f"Projection Matching Alignment (PMA) [{shift_method}{grad_str}{preprocess_str}]")

    if standardize:
        tomo.standardize(isPhaseData=isPhaseData)
    tomo.center_projections()

    iters_per_level = ([max_iterations] * levels if iterations_per_level is None
                       else list(iterations_per_level))
    assert len(iters_per_level) == levels

    original = tomo.workingProjections.copy()
    pma_shifts = np.zeros((tomo.num_angles, 2), dtype=np.float64)

    for level_idx, level in enumerate(reversed(range(levels))):
        downsample_factor = scale ** level
        n_iters = iters_per_level[level_idx]
        print(f"\n--- PMA Level {level} ({downsample_factor}x downsampled, {n_iters} iterations) ---")

        current_projs = np.stack([
            subpixel_shift(original[i], pma_shifts[i, 0], pma_shifts[i, 1])
            for i in range(tomo.num_angles)
        ], dtype=np.float32)

        if level > 0:
            *_, scaled_projs = pyramid_gaussian(
                current_projs, downscale=scale, max_layer=level, channel_axis=0
            )
            scaled_projs = scaled_projs.astype(np.float32)
        else:
            scaled_projs = current_projs

        scaled_center = tomo.rotation_center / downsample_factor
        level_shifts = np.zeros((tomo.num_angles, 2), dtype=np.float64)
        level_snapshot = scaled_projs.copy()

        # ROI bounds check
        if xROI_Range is not None and yROI_Range is not None:
            H, W = scaled_projs.shape[1:]
            print(f"Using ROI: x={xROI_Range}, y={yROI_Range} (downsampled by {downsample_factor}x)")
            print(f"H and W values are {H} and {W}")
            print(f"Downsample ROI bounds are x={xROI_Range[0]//downsample_factor} to {xROI_Range[1]//downsample_factor}, y={yROI_Range[0]//downsample_factor} to {yROI_Range[1]//downsample_factor}")
            assert 0 <= xROI_Range[0]//downsample_factor < xROI_Range[1]//downsample_factor <= W
            assert 0 <= yROI_Range[0]//downsample_factor < yROI_Range[1]//downsample_factor <= H

        for k in tqdm(range(n_iters), desc=f'PMA Level {level} iterations'):
            recon  = _pma_reconstruct(scaled_projs, tomo.ang, scaled_center, algorithm, recon_crop_ratio)
            reproj = tomo.simulateProjections(recon=recon, pad=False, center=None)

            if standardize:
                reproj = (reproj - np.mean(reproj)) / np.std(reproj)

            dy = np.zeros(tomo.num_angles)
            dx = np.zeros(tomo.num_angles)

            plot_idx = tomo.num_angles // 2
            plot_ref = plot_mov = None

            for i in range(tomo.num_angles):

                ref = reproj[i]
                mov = scaled_projs[i]

                # ROI extraction
                if xROI_Range is not None and yROI_Range is not None:
                    y0 = int(yROI_Range[0]) // downsample_factor
                    y1 = int(yROI_Range[1]) // downsample_factor
                    x0 = int(xROI_Range[0]) // downsample_factor
                    x1 = int(xROI_Range[1]) // downsample_factor
                    ref_roi = ref[y0:y1, x0:x1]
                    mov_roi = mov[y0:y1, x0:x1]
                else:
                    ref_roi = ref
                    mov_roi = mov

                # Preprocessing
                if use_matching_preprocess:
                    ref_roi = _preprocess_for_matching(ref_roi, matching_sigma)
                    mov_roi = _preprocess_for_matching(mov_roi, matching_sigma)

                if use_grad:
                    ref_roi = compute_grad_image(ref_roi)
                    mov_roi = compute_grad_image(mov_roi)

                if plot and k == 0 and i == plot_idx:
                    plot_ref, plot_mov = ref_roi, mov_roi

                # Shift estimation
                if shift_method == 'optical_flow':
                    dy[i], dx[i] = _optical_flow_shift(mov_roi, ref_roi, of_sigma)
                else:
                    dy[i], dx[i] = _cross_correlation_shift(ref_roi, mov_roi, upsample_factor)

            # Plot matching inputs
            if plot and k == 0 and plot_ref is not None:
                vmin = min(plot_mov.min(), plot_ref.min())
                vmax = max(plot_mov.max(), plot_ref.max())

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f"PMA Level {level} — Matching Inputs (Projection {plot_idx})")

                im0 = axes[0].imshow(plot_mov, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
                axes[0].set_title("mov (ROI)")
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                im1 = axes[1].imshow(plot_ref, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
                axes[1].set_title("ref (ROI)")
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.show()

            # Center shifts
            dy -= np.mean(dy)
            dx -= np.mean(dx)

            if smooth_sigma:
                dy = gaussian_filter1d(dy, sigma=smooth_sigma)
                dx = gaussian_filter1d(dx, sigma=smooth_sigma)

            # Clamp
            dy = np.clip(dy, -max_step, max_step)
            dx = np.clip(dx, -max_step, max_step)

            dy *= stepRatio
            dx *= stepRatio

            level_shifts[:, 0] += dy
            level_shifts[:, 1] += dx

            # Apply shifts
            for i in range(tomo.num_angles):
                scaled_projs[i] = subpixel_shift(
                    level_snapshot[i],
                    level_shifts[i, 0],
                    level_shifts[i, 1]
                )

            # --- Updated reporting ---
            shift_magnitudes = np.sqrt(dy**2 + dx**2)
            avg_shift = np.mean(shift_magnitudes)
            max_shift = np.max(shift_magnitudes)

            clamp_flag = " [CLAMPED]" if np.isclose(max_shift, max_step) else ""

            print(f"Iteration {k+1}: avg shift = {downsample_factor * avg_shift:.4f} px, "
                  f"max shift = {downsample_factor * max_shift:.4f} px{clamp_flag}")

            if downsample_factor * avg_shift < tolerance:
                print(f"  Convergence at level {level} after {k+1} iterations.")
                break

        pma_shifts += level_shifts * downsample_factor

    # Apply final shifts
    for i in range(tomo.num_angles):
        tomo.workingProjections[i] = subpixel_shift(
            original[i],
            pma_shifts[i, 0],
            pma_shifts[i, 1]
        )

    tomo.tracked_shifts += pma_shifts
    print("\nPMA complete.")

def vertical_mass_fluctuation_align(
    tomo,
    tolerance=0.0,
    max_iterations=10,
    y_range=None,
    upsample_factor=50,
    window='hanning',     # 'hanning', 'soft_roi', or None — suppresses cut-off boundary artifacts
    roi_sigma=0.3,        # Gaussian half-width as fraction of frame height (only for 'soft_roi')
    use_gradient=True,   # Differentiate mass profile — sensitive to internal features, ignores bulk cut-off
    plot=False,           # Plot window profile, final overall profile, and second projection profile
    stepRatio=1.0,        # Fraction of computed shift to apply each iteration (damping)

):
    print(f"VMF Alignment (upsample={upsample_factor}, window={window}, gradient={use_gradient}, stepRatio={stepRatio})")
    n = tomo.num_angles

    for iteration in range(max_iterations):
        # We work from the same 'snapshot' to avoid multiple interpolation blurs
        snapshot = tomo.workingProjections.copy()

        profiles = []
        win_to_plot = None
        for k in range(n):
            img = snapshot[k] if y_range is None else snapshot[k][y_range[0]:y_range[1]]

            # 1. Generate Vertical Profile
            m = np.sum(img, axis=1).astype(np.float64)

            # 2. (Optional) Gradient-based profile — the derivative highlights internal
            #    density transitions while making the cut-off boundary a single spike
            #    that the window (step 3) can then suppress.
            if use_gradient:
                m = np.gradient(m)

            # 3. Intensity Normalization (Crucial for VMF)
            # This makes the sum independent of beam intensity fluctuations.
            # Use mean-abs for gradient profiles since they can be signed.
            if use_gradient:
                m /= (np.mean(np.abs(m)) + 1e-9)
            else:
                m /= (np.mean(m) + 1e-9)

            # 4. Apply vertical window to taper boundary artifacts to zero.
            #    'hanning'  — cosine taper, both edges fade to 0; best general choice.
            #    'soft_roi' — Gaussian centred in frame; useful when top is in-frame
            #                 but bottom is cut off (upweights centre over edges).

            if window == 'hanning':
                win = np.hanning(len(m))
                m = m * win
            elif window == 'soft_roi':
                y = np.arange(len(m))
                center = (len(m) - 1) / 2.0
                win = np.exp(-0.5 * ((y - center) / (roi_sigma * len(m))) ** 2)
                m = m * win
            else:
                win = np.ones(len(m))

            if plot and k == 0:
                win_to_plot = win.copy()

            profiles.append(m)

        profiles = np.array(profiles)

        # Robust Reference: The average fluctuation across all angles
        ref = np.mean(profiles, axis=0)

        shifts_y = np.zeros(n)
        for i in range(n):
            shift, error, diffphase = phase_cross_correlation(
                ref[:, np.newaxis],
                profiles[i][:, np.newaxis],
                upsample_factor=upsample_factor
            )
            shifts_y[i] = shift[0]

        # Subtract the mean shift to keep the volume centered in the FOV
        shifts_y -= np.mean(shifts_y)

        shifts_y *= stepRatio

        # Apply shifts
        for i in range(n):
            tomo.workingProjections[i] = subpixel_shift(snapshot[i], shifts_y[i], 0)
            tomo.tracked_shifts[i, 0] += shifts_y[i]

        avg_delta = np.mean(np.abs(shifts_y))
        print(f"  Iteration {iteration + 1}: Mean Correction = {avg_delta:.4f} px")

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"VMF Alignment — Iteration {iteration + 1}")

            # Window profile
            axes[0].plot(win_to_plot)
            axes[0].set_title(f"Window Profile ({window if window else 'none'})")
            axes[0].set_xlabel("Pixel (vertical)")
            axes[0].set_ylabel("Weight")

            # Overall reference profile (mean across all projections)
            axes[1].plot(ref)
            axes[1].set_title("Overall Reference Profile\n(mean fluctuation across all angles)")
            axes[1].set_xlabel("Pixel (vertical)")
            axes[1].set_ylabel("Fluctuation")

            # Second projection's processed profile (index 1, or 0 if only one angle)
            second_idx = min(1, n - 1)
            axes[2].plot(profiles[second_idx])
            axes[2].set_title(f"Processed Profile — Projection {second_idx}")
            axes[2].set_xlabel("Pixel (vertical)")
            axes[2].set_ylabel("Fluctuation")

            plt.tight_layout()
            plt.show()

        if avg_delta < tolerance:
            print(f"  Converged.")
            break

















###Less successful algorithms below
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












###Scoring functions for alignment quality assessment

def sinogram_consistency_score(tomo, plot=True, bg_percentile=None):
    """
    Quantifies alignment quality using the Helgason-Ludwig consistency conditions.
    Modified to handle datasets containing negative values.
    Also displays a central-slice sinogram for visual alignment assessment.
    """
    n = tomo.num_angles
    angles = tomo.ang.ravel()
    data_to_measure = tomo.workingProjections
    
    ny, nx = data_to_measure.shape[1], data_to_measure.shape[2]
    x_coords = np.arange(nx, dtype=np.float64)
    y_coords = np.arange(ny, dtype=np.float64)

    x_cm = np.zeros(n)
    y_cm = np.zeros(n)

    for i in range(n):
        img = data_to_measure[i].astype(np.float64)

        # Shift to positive
        img = img - np.min(img)

        if bg_percentile is not None:
            bg = np.percentile(img, bg_percentile)
            img = np.clip(img - bg, 0, None)

        col_sums = img.sum(axis=0)
        row_sums = img.sum(axis=1)
        total = col_sums.sum()

        if total > 1e-9:
            x_cm[i] = (x_coords * col_sums).sum() / total
            y_cm[i] = (y_coords * row_sums).sum() / total
        else:
            x_cm[i] = nx / 2.0
            y_cm[i] = ny / 2.0

    # Fit sinusoid to x_cm
    design = np.column_stack([np.cos(angles), np.sin(angles), np.ones(n)])
    coeffs_x, _, _, _ = np.linalg.lstsq(design, x_cm, rcond=None)
    x_fit = design @ coeffs_x
    x_residuals = x_cm - x_fit
    x_rmse = np.sqrt(np.mean(x_residuals ** 2))
    
    ss_tot_x = np.sum((x_cm - x_cm.mean()) ** 2)
    r2_x = 1.0 - np.sum(x_residuals ** 2) / ss_tot_x if ss_tot_x > 0 else 0.0

    # y_cm should be constant
    y_fit = np.full(n, y_cm.mean())
    y_residuals = y_cm - y_fit
    y_rmse = np.sqrt(np.mean(y_residuals ** 2))
    
    ss_tot_y = np.sum((y_cm - y_cm.mean()) ** 2)
    r2_y = 1.0 - np.sum(y_residuals ** 2) / ss_tot_y if ss_tot_y > 0 else 0.0

    combined_rmse = np.sqrt((x_rmse ** 2 + y_rmse ** 2) / 2)

    print(f"Sinogram consistency:")
    print(f"  x_cm (horizontal) — RMSE: {x_rmse:.4f} px  |  R²: {r2_x:.6f}")
    print(f"  y_cm (vertical)   — RMSE: {y_rmse:.4f} px  |  R²: {r2_y:.6f}")
    print(f"  Combined RMSE:       {combined_rmse:.4f} px")

    # Compute or reuse reprojections for the sinogram comparison panel
    reprojections = None
    if hasattr(tomo, 'finalReprojections') and tomo.finalReprojections is not None:
        reprojections = tomo.finalReprojections
    elif hasattr(tomo, 'recon') and tomo.recon is not None:
        print("Computing reprojections from reconstruction for sinogram comparison...")
        recon_masked = tomopy.circ_mask(tomo.recon.copy(), axis=0, ratio=0.99)
        reprojections = tomo.simulateProjections(recon=recon_masked, emission=True, pad=False, ncore=None)
        nx_m = data_to_measure.shape[2]
        nx_r = reprojections.shape[2]
        if nx_r != nx_m:
            start = (nx_r - nx_m) // 2
            reprojections = reprojections[:, :, start:start + nx_m]
        tomo.finalReprojections = reprojections

    if plot:
        angles_deg = np.rad2deg(angles)
        order = np.argsort(angles_deg)
        ad = angles_deg[order]

        # --- central slice sinograms ---
        center_row = ny // 2
        sinogram_data = data_to_measure[:, center_row, :]  # (angles, x)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

        fig.suptitle(f'Sinogram Consistency  —  Combined RMSE={combined_rmse:.4f} px', fontsize=12)

        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1])
        ax_sino_data   = fig.add_subplot(gs[2, 0])
        ax_sino_reproj = fig.add_subplot(gs[2, 1])

        ax00.plot(ad, x_cm[order], '.', markersize=3, label='x_cm')
        ax00.plot(ad, x_fit[order], '-', linewidth=1.5, label='Sinusoid fit')
        ax00.set_ylabel('x center of mass (px)')
        ax00.set_title(f'Horizontal  RMSE={x_rmse:.4f} px')
        ax00.legend(markerscale=3)

        ax10.plot(ad, x_residuals[order], '.', markersize=3, color='tomato')
        ax10.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax10.set_xlabel('Angle (degrees)')
        ax10.set_ylabel('Residual (px)')

        ax01.plot(ad, y_cm[order], '.', markersize=3, color='steelblue', label='y_cm')
        ax01.axhline(y_cm.mean(), color='orange', linewidth=1.5, label='Mean')
        ax01.set_ylabel('y center of mass (px)')
        ax01.set_title(f'Vertical  RMSE={y_rmse:.4f} px')
        ax01.legend(markerscale=3)

        ax11.plot(ad, y_residuals[order], '.', markersize=3, color='tomato')
        ax11.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax11.set_xlabel('Angle (degrees)')
        ax11.set_ylabel('Residual (px)')

        # --- data sinogram ---
        sino_data_sorted = sinogram_data[order]
        vmin = sino_data_sorted.min()
        vmax = sino_data_sorted.max()

        im_d = ax_sino_data.imshow(
            sino_data_sorted, aspect='auto', vmin=vmin, vmax=vmax,
            extent=[0, nx, ad.max(), ad.min()]
        )
        ax_sino_data.set_title('Data Sinogram (central slice)')
        ax_sino_data.set_ylabel('Angle (deg)')
        ax_sino_data.set_xlabel('Detector pixel')
        plt.colorbar(im_d, ax=ax_sino_data)

        # --- reprojected sinogram (if available) ---
        if reprojections is not None:
            sino_reproj = reprojections[:, center_row, :]
            sino_reproj_sorted = sino_reproj[order]
            vmin_r = min(vmin, sino_reproj_sorted.min())
            vmax_r = max(vmax, sino_reproj_sorted.max())
            im_r = ax_sino_reproj.imshow(
                sino_reproj_sorted, aspect='auto', vmin=vmin_r, vmax=vmax_r,
                extent=[0, nx, ad.max(), ad.min()]
            )
            ax_sino_reproj.set_title('Reprojected Sinogram (central slice)')
            plt.colorbar(im_r, ax=ax_sino_reproj)
        else:
            ax_sino_reproj.text(
                0.5, 0.5,
                "No reprojections available.\nRun tomo.reconstruct() first.",
                ha='center', va='center', transform=ax_sino_reproj.transAxes, fontsize=9
            )
            ax_sino_reproj.set_title('Reprojected Sinogram (central slice)')
        ax_sino_reproj.set_ylabel('Angle (deg)')
        ax_sino_reproj.set_xlabel('Detector pixel')

        plt.tight_layout()
        plt.show()

    return combined_rmse, x_rmse, y_rmse, x_cm, y_cm

def reconstruction_sharpness_score(recon, plot=True, percentile_crop=5):
    """
    Quantifies 3D reconstruction sharpness using two per-slice metrics:

    1. Mean gradient magnitude  mean(|∇I|):
       Measures average edge strength across each slice. A well-aligned reconstruction
       has crisp boundaries between materials; misalignment smears those edges and
       reduces this value. Uses np.gradient (central finite differences).

    2. Variance of Laplacian  var(∇²I):
       Classic autofocus metric from computational photography. The Laplacian is a
       high-pass filter — sharp images have high variance in their high-frequency
       content; blurry/streaky images have low variance. More sensitive to fine
       structure than gradient magnitude alone.

    Scores are raw (not normalized) so they unambiguously increase when sharpness
    improves. Normalizing by mean intensity inverts the result because alignment
    concentrates attenuation into the correct voxels, raising the mean faster than
    the gradient — making a sharper reconstruction appear to score lower.

    Scores are only meaningful as a before/after comparison on the same dataset
    (same object, same intensity scale). Use percentile_crop to suppress ring
    artifacts or hot pixels that would otherwise dominate.

    The per-slice arrays let you identify which axial positions are worst-aligned
    (e.g., a dip in the middle could indicate a specific angular range is off).

    Parameters:
    - recon: 3D numpy array (nz, ny, nx) — the reconstruction volume (e.g. tomo.recon).
    - plot: If True, plots per-slice sharpness profiles for both metrics.
    - percentile_crop: Clips intensities to [p, 100-p] percentile before scoring —
      suppresses ring artifacts or edge streaks that would bias the score.

    Returns:
    - grad_score (float): Overall mean gradient magnitude — higher is sharper.
    - lap_score (float): Overall Laplacian variance — higher is sharper.
    - grad_per_slice (ndarray): Per-slice gradient scores, shape (nz,).
    - lap_per_slice (ndarray): Per-slice Laplacian variance scores, shape (nz,).
    """
    vol = recon.astype(np.float64)

    # Clip outliers (ring artifacts, hot pixels) that would inflate gradient/Laplacian
    if percentile_crop is not None:
        lo, hi = np.percentile(vol, [percentile_crop, 100 - percentile_crop])
        vol = np.clip(vol, lo, hi)

    nz = vol.shape[0]
    grad_per_slice = np.zeros(nz)
    lap_per_slice = np.zeros(nz)

    for i in range(nz):
        slc = vol[i]

        # Gradient magnitude: finite-difference ∂I/∂x and ∂I/∂y, then mean of magnitude
        gy, gx = np.gradient(slc)
        grad_per_slice[i] = np.mean(np.sqrt(gx**2 + gy**2))

        # Laplacian variance: scipy laplace = ∂²I/∂x² + ∂²I/∂y², then variance
        lap = sp.ndimage.laplace(slc)
        lap_per_slice[i] = lap.var()

    grad_score = float(grad_per_slice.mean())
    lap_score = float(lap_per_slice.mean())

    print(f"Reconstruction sharpness:")
    print(f"  Mean gradient magnitude: {grad_score:.6f}")
    print(f"  Laplacian variance:      {lap_score:.6f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(
            f'Reconstruction Sharpness  —  Gradient={grad_score:.4f}  Laplacian={lap_score:.4f}',
            fontsize=12
        )

        axes[0].plot(grad_per_slice, linewidth=1)
        axes[0].axhline(grad_score, color='r', linestyle='--', linewidth=1,
                        label=f'Mean = {grad_score:.4f}')
        axes[0].set_xlabel('Slice index')
        axes[0].set_ylabel('Mean |∇I| (normalized)')
        axes[0].set_title('Gradient Magnitude per Slice')
        axes[0].legend()

        axes[1].plot(lap_per_slice, linewidth=1, color='steelblue')
        axes[1].axhline(lap_score, color='r', linestyle='--', linewidth=1,
                        label=f'Mean = {lap_score:.4f}')
        axes[1].set_xlabel('Slice index')
        axes[1].set_ylabel('var(∇²I) (normalized)')
        axes[1].set_title('Laplacian Variance per Slice')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return grad_score, lap_score, grad_per_slice, lap_per_slice



def reprojection_consistency_score(tomo, plot=True, use_circ_mask=True):
    """
    Computes the Reprojection Consistency Score (RCS) — Metric 1 for alignment quality.

    For each angle, reprojects the 3D reconstruction and computes the Normalized
    Root-Mean-Squared Error (NRMSE) against the corresponding measured projection:

        NRMSE(θ) = ‖P_θ - P̂_θ‖₂ / ‖P_θ‖₂

        RCS = mean over all θ of NRMSE(θ)

    A lower score means the reconstruction is more self-consistent with the data.
    Typical ranges:
        < 0.10  →  excellent alignment
        0.10–0.20  →  acceptable
        0.20–0.35  →  moderate misalignment or noise
        > 0.35   →  poor alignment, likely reconstruction artifacts

    Parameters
    ----------
    tomo : tomoData
        A tomoData object that has already been aligned and reconstructed.
        Must have:
          - tomo.finalProjections  : ndarray (n_angles, ny, nx)
          - tomo.recon             : ndarray (nz, ny, nx)  — set by tomo.reconstruct()
          - tomo.ang               : ndarray (n_angles,) in radians
          - tomo.rotation_center   : float, set by tomo.reconstruct()
    plot : bool
        If True, produces two diagnostic plots:
          1. Per-angle NRMSE bar chart — reveals which angles are worst-aligned.
          2. Worst and best angle side-by-side: measured vs reprojection overlay.
    use_circ_mask : bool
        If True, applies a circular mask to the reconstruction before reprojecting,
        consistent with the circ_mask applied during reconstruction in this codebase.

    Returns
    -------
    rcs : float
        The scalar Reprojection Consistency Score (mean NRMSE across all angles).
    per_angle_nrmse : ndarray (n_angles,)
        The per-angle NRMSE values, useful for diagnosing which angles drive errors.
    reprojections : ndarray (n_angles, ny, nx)
        The synthetic reprojections of the reconstruction, for further inspection.
    """
    if not hasattr(tomo, 'recon') or tomo.recon is None:
        raise AttributeError(
            "tomo.recon is not set. Run tomo.reconstruct() before calling this function."
        )

    measured = tomo.finalProjections           # (n_angles, ny, nx)
    recon    = tomo.recon.copy()               # (nz, ny, nx)
    angles   = tomo.ang                        # (n_angles,) in radians
    center   = tomo.rotation_center
    n_angles = tomo.num_angles

    # Apply the same circular mask used during reconstruction so the
    # reprojections are computed on the same effective volume.
    if use_circ_mask:
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.99)

    nx_m = measured.shape[2]

    def _compute_reprojections():
        print("Computing reprojections of reconstruction...")
        raw = tomo.simulateProjections(recon=recon, emission=True, pad=False, ncore=None)
        # tomopy pads the reconstruction volume, so reprojections may be wider
        # than the measured projections. Crop the center columns to match.
        nx_r = raw.shape[2]
        if nx_r != nx_m:
            start = (nx_r - nx_m) // 2
            raw = raw[:, :, start:start + nx_m]
        return raw

    # Reuse cached reprojections only if they match the current projection width.
    if (hasattr(tomo, 'finalReprojections')
            and tomo.finalReprojections is not None
            and tomo.finalReprojections.shape[2] == nx_m):
        print("Reusing cached reprojections from tomo.finalReprojections.")
        reprojections = tomo.finalReprojections
    else:
        reprojections = _compute_reprojections()
        tomo.finalReprojections = reprojections

    # --- Compute per-angle NRMSE ---
    print("Computing per-angle NRMSE...")
    per_angle_nrmse = np.zeros(n_angles)

    for i in tqdm(range(n_angles), desc="NRMSE per angle"):
        meas = measured[i].astype(np.float64)
        reproj = reprojections[i].astype(np.float64)

        residual_norm = np.linalg.norm(meas - reproj)
        meas_norm     = np.linalg.norm(meas)

        # Guard against a zero-energy projection (e.g. blank frames)
        if meas_norm < 1e-12:
            per_angle_nrmse[i] = np.nan
        else:
            per_angle_nrmse[i] = residual_norm / meas_norm

    # Scalar RCS: mean over valid (non-NaN) angles
    valid_mask = ~np.isnan(per_angle_nrmse)
    rcs = float(np.mean(per_angle_nrmse[valid_mask]))

    # --- Console report ---
    worst_idx = int(np.nanargmax(per_angle_nrmse))
    best_idx  = int(np.nanargmin(per_angle_nrmse))
    print("\n─── Reprojection Consistency Score ───────────────────────")
    print(f"  RCS (mean NRMSE):   {rcs:.4f}")
    print(f"  Best  angle [{best_idx:>4}]:  NRMSE = {per_angle_nrmse[best_idx]:.4f}")
    print(f"  Worst angle [{worst_idx:>4}]:  NRMSE = {per_angle_nrmse[worst_idx]:.4f}")
    print(f"  Std across angles:  {np.nanstd(per_angle_nrmse):.4f}")
    if rcs < 0.10:
        verdict = "✓  Excellent — reconstruction is highly self-consistent with data."
    elif rcs < 0.20:
        verdict = "~  Acceptable — minor residual misalignment or noise present."
    elif rcs < 0.35:
        verdict = "⚠  Moderate — consider additional PMA iterations or check alignment."
    else:
        verdict = "✗  Poor — significant misalignment or reconstruction failure."
    print(f"  Verdict:  {verdict}")
    print("───────────────────────────────────────────────────────────\n")

    if not plot:
        return rcs, per_angle_nrmse, reprojections

    # --- Prepare shared quantities ---
    angle_deg = np.degrees(angles.ravel())
    order = np.argsort(angle_deg)
    ad = angle_deg[order]

    ny, nx = measured.shape[1], measured.shape[2]

    best_meas    = measured[best_idx]
    best_reproj  = reprojections[best_idx]
    worst_meas   = measured[worst_idx]
    worst_reproj = reprojections[worst_idx]

    # --- Build figure: bar chart (top), 2x2 image grid (bottom) ---
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.5, 1.5], hspace=0.4, wspace=0.3)

    fig.suptitle(f"Reprojection Consistency Score = {rcs:.4f}   |   {verdict}",
                 fontsize=10)

    ax_bar          = fig.add_subplot(gs[0, :])
    ax_best_orig    = fig.add_subplot(gs[1, 0])
    ax_best_reproj  = fig.add_subplot(gs[1, 1])
    ax_worst_orig   = fig.add_subplot(gs[2, 0])
    ax_worst_reproj = fig.add_subplot(gs[2, 1])

    # --- Plot 1: per-angle NRMSE bar chart ---
    colors = np.where(per_angle_nrmse > rcs + np.nanstd(per_angle_nrmse),
                      '#d62728', '#1f77b4')
    ax_bar.bar(angle_deg, per_angle_nrmse, color=colors,
               width=(angle_deg[1] - angle_deg[0]) * 0.9)
    ax_bar.axhline(rcs, color='black', linewidth=1.5, linestyle='--',
                   label=f'RCS = {rcs:.4f}')
    ax_bar.axhline(rcs + np.nanstd(per_angle_nrmse), color='red', linewidth=1,
                   linestyle=':', label='mean + 1σ')
    ax_bar.set_xlabel("Projection angle (degrees)")
    ax_bar.set_ylabel("NRMSE")
    ax_bar.set_title("Per-Angle Reprojection NRMSE\n(red bars = outlier angles)")
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(bottom=0)

    # --- Plot 2: 2x2 grid — best (top) and worst (bottom), original vs reprojection ---
    def _shared_clim(a, b):
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        return lo, hi

    best_vmin, best_vmax = _shared_clim(best_meas, best_reproj)
    worst_vmin, worst_vmax = _shared_clim(worst_meas, worst_reproj)

    kw = dict(aspect='auto', cmap='gray')
    tick_kw = dict(labelsize=7)

    def _set_img_axes(ax):
        ax.tick_params(axis='both', **tick_kw)
        ax.set_xlabel("Width (px)", fontsize=8)
        ax.set_ylabel("Height (px)", fontsize=8)

    im = ax_best_orig.imshow(best_meas, vmin=best_vmin, vmax=best_vmax, **kw)
    ax_best_orig.set_title(f"Best {angle_deg[best_idx]:.1f}° — Original\nNRMSE={per_angle_nrmse[best_idx]:.4f}", fontsize=9)
    _set_img_axes(ax_best_orig)
    plt.colorbar(im, ax=ax_best_orig, fraction=0.046, pad=0.04)

    im = ax_best_reproj.imshow(best_reproj, vmin=best_vmin, vmax=best_vmax, **kw)
    ax_best_reproj.set_title(f"Best {angle_deg[best_idx]:.1f}° — Reprojection", fontsize=9)
    _set_img_axes(ax_best_reproj)
    plt.colorbar(im, ax=ax_best_reproj, fraction=0.046, pad=0.04)

    im = ax_worst_orig.imshow(worst_meas, vmin=worst_vmin, vmax=worst_vmax, **kw)
    ax_worst_orig.set_title(f"Worst {angle_deg[worst_idx]:.1f}° — Original\nNRMSE={per_angle_nrmse[worst_idx]:.4f}", fontsize=9)
    _set_img_axes(ax_worst_orig)
    plt.colorbar(im, ax=ax_worst_orig, fraction=0.046, pad=0.04)

    im = ax_worst_reproj.imshow(worst_reproj, vmin=worst_vmin, vmax=worst_vmax, **kw)
    ax_worst_reproj.set_title(f"Worst {angle_deg[worst_idx]:.1f}° — Reprojection", fontsize=9)
    _set_img_axes(ax_worst_reproj)
    plt.colorbar(im, ax=ax_worst_reproj, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return rcs, per_angle_nrmse, reprojections


# ─────────────────────────────────────────────────────────────────────────────
# Fourier Shell Correlation (FSC) resolution metric
# ─────────────────────────────────────────────────────────────────────────────

def _fsc_compute_shells(vol1, vol2):
    """
    Core FSC computation via 3D FFT + vectorised bincount.

    Returns
    -------
    fsc : ndarray (max_r+1,)          — FSC value per radial shell
    n_vox : ndarray int (max_r+1,)    — voxels per shell (for σ thresholds)
    freqs : ndarray (max_r+1,)        — spatial frequency in cycles/pixel (0 … 0.5)
    """
    nz, ny, nx = vol1.shape

    F1 = np.fft.fftshift(np.fft.fftn(vol1.astype(np.float64)))
    F2 = np.fft.fftshift(np.fft.fftn(vol2.astype(np.float64)))

    # Radial distance from centre in pixel units (integer shells)
    def _centred_freq(n):
        return np.fft.fftshift(np.fft.fftfreq(n)) * n

    KZ, KY, KX = np.meshgrid(_centred_freq(nz), _centred_freq(ny), _centred_freq(nx), indexing='ij')
    r_flat = np.sqrt(KZ**2 + KY**2 + KX**2).ravel()

    max_r = min(nz, ny, nx) // 2
    shells = np.round(r_flat).astype(np.int32)
    valid = shells <= max_r
    shells = shells[valid]

    cross = (F1 * np.conj(F2)).real.ravel()[valid]
    p1    = (np.abs(F1)**2).ravel()[valid]
    p2    = (np.abs(F2)**2).ravel()[valid]

    n_vox  = np.bincount(shells, minlength=max_r + 1)
    num    = np.bincount(shells, weights=cross, minlength=max_r + 1)
    denom  = np.sqrt(
        np.bincount(shells, weights=p1, minlength=max_r + 1) *
        np.bincount(shells, weights=p2, minlength=max_r + 1)
    )
    fsc = np.where(denom > 0, num / denom, 0.0)

    # Shell k → frequency k / min_dim  (so shell max_r → 0.5 cycles/pixel)
    freqs = np.arange(max_r + 1) / min(nz, ny, nx)
    return fsc, n_vox, freqs


def _fsc_crossing(fsc, threshold, freqs):
    """
    Resolution at the *last* downward crossing of FSC below `threshold`.

    Uses linear interpolation between the last shell where FSC ≥ threshold
    and the first shell below it.  The "last above" convention avoids false
    crossings at low frequencies where the statistical thresholds can exceed 1.

    Parameters
    ----------
    threshold : float or ndarray matching fsc
    freqs : ndarray — spatial frequency per shell (cycles/pixel)

    Returns
    -------
    resolution_px : float or None   — resolution in pixels; None if FSC never
                                      drops below threshold within measured range.
    """
    t = threshold if isinstance(threshold, np.ndarray) else np.full(len(fsc), float(threshold))

    # Skip DC shell (index 0)
    above = fsc[1:] >= t[1:]
    if not above.any():
        return None

    last_above = int(np.where(above)[0][-1]) + 1  # back to original indexing
    if last_above + 1 >= len(fsc):
        return None  # FSC stays above threshold all the way to Nyquist

    f1, f2 = fsc[last_above], fsc[last_above + 1]
    t1, t2 = t[last_above], t[last_above + 1]
    q1, q2 = freqs[last_above], freqs[last_above + 1]

    delta = (f1 - t1) - (f2 - t2)
    crossing_freq = q1 + (f1 - t1) / delta * (q2 - q1) if abs(delta) > 1e-12 else q2
    return 1.0 / crossing_freq if crossing_freq > 1e-9 else None


def fourier_shell_correlation(tomo, algorithm='gridrec', plot=True,
                               smooth_sigma=0.0, apply_circ_mask=True):
    """
    Estimate reconstruction resolution using the gold-standard half-dataset
    Fourier Shell Correlation (FSC) method.

    The tilt series is split into two interleaved halves (even / odd angle
    indices).  Each half is independently reconstructed and the two volumes
    are compared in 3D Fourier space shell-by-shell to produce the FSC curve.

    Three threshold criteria are reported:
      • FSC = 0.5   — traditional fixed threshold
      • FSC = 0.143 — modern gold-standard (equivalent to the 0.5 criterion
                       in X-ray crystallography; Rosenthal & Henderson 2003)
      • 3σ          — statistical criterion 3 / √N_k  (treat as optimistic for
                       ptychographic data where per-projection noise is correlated)

    .. note::
        For limited-angle datasets with a missing wedge the FSC averages over
        all 3D Fourier directions, blending well-resolved in-plane directions
        with the poorly-resolved beam direction.  The reported value is a
        direction-averaged upper bound on structural reproducibility.

    Parameters
    ----------
    tomo : tomoData
    algorithm : str
        Reconstruction algorithm passed to tomopy.recon for the two
        half-datasets.  'gridrec' is fast; 'art' gives better quality.
    plot : bool
    smooth_sigma : float
        Standard deviation (in shells) of a Gaussian used to smooth the FSC
        curve before threshold detection.  0 = no smoothing.
    apply_circ_mask : bool
        Apply the same circular mask used during full reconstruction.

    Returns
    -------
    fsc : ndarray (n_shells,)
        Raw (unsmoothed) FSC curve.
    resolutions : dict
        Resolution in pixels at each threshold; None if FSC never drops below.
    freqs : ndarray (n_shells,)
        Spatial frequency in cycles/pixel for each shell.
    """
    projs  = tomo.finalProjections
    angles = tomo.ang
    center = tomo.rotation_center
    n      = tomo.num_angles

    even_idx = np.arange(0, n, 2)
    odd_idx  = np.arange(1, n, 2)

    print(f"\nFourier Shell Correlation  (algorithm={algorithm})")
    print(f"  Half 1 (even angles): {len(even_idx)} projections")
    print(f"  Half 2 (odd  angles): {len(odd_idx)}  projections")

    print("  Reconstructing half 1 …")
    r1 = tomopy.recon(projs[even_idx], angles[even_idx],
                      center=center, algorithm=algorithm, sinogram_order=False)
    print("  Reconstructing half 2 …")
    r2 = tomopy.recon(projs[odd_idx],  angles[odd_idx],
                      center=center, algorithm=algorithm, sinogram_order=False)

    if apply_circ_mask:
        r1 = tomopy.circ_mask(r1, axis=0, ratio=0.99)
        r2 = tomopy.circ_mask(r2, axis=0, ratio=0.99)

    print("  Computing FSC …")
    fsc, n_vox, freqs = _fsc_compute_shells(r1, r2)

    # Statistical thresholds
    n_safe     = np.maximum(n_vox, 1)
    three_sigma = 3.0 / np.sqrt(n_safe)
    half_bit    = (0.2071 + 1.9102 / np.sqrt(n_safe)) / (1.2071 + 0.9102 / np.sqrt(n_safe))

    # Optionally smooth the curve for threshold detection (not applied to n_vox)
    fsc_smooth = fsc
    if smooth_sigma > 0:
        fsc_smooth = gaussian_filter1d(fsc, sigma=smooth_sigma)

    resolutions = {
        'FSC=0.5':   _fsc_crossing(fsc_smooth, 0.5,         freqs),
        'FSC=0.143': _fsc_crossing(fsc_smooth, 0.143,        freqs),
        '3-sigma':   _fsc_crossing(fsc_smooth, three_sigma,  freqs),
    }

    # ── Console report ────────────────────────────────────────────────────────
    nz, ny, nx = r1.shape
    nyquist_px = 2.0  # Nyquist = 2 pixels by definition
    print("\n─── Fourier Shell Correlation ────────────────────────────────")
    print(f"  Volume shape (half-map): {r1.shape}  |  Nyquist limit: {nyquist_px:.1f} px")
    print(f"  {'Threshold':<20} {'Freq (cyc/px)':>15} {'Resolution (px)':>16}")
    print(f"  {'─'*53}")
    for name, res in resolutions.items():
        if res is None:
            print(f"  {name:<20} {'> Nyquist':>15} {'(not reached)':>16}")
        else:
            freq = 1.0 / res
            print(f"  {name:<20} {freq:>15.4f} {res:>16.2f}")
    print("──────────────────────────────────────────────────────────────\n")

    if plot:
        _fsc_plot(fsc, fsc_smooth, freqs, three_sigma, half_bit, resolutions, r1, r2)

    return fsc, resolutions, freqs


def _fsc_plot(fsc_raw, fsc_smooth, freqs, three_sigma, half_bit, resolutions, r1, r2):
    """Plot FSC curve with thresholds and central-slice half-map comparison."""
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.3)
    ax_fsc   = fig.add_subplot(gs[0, :])
    ax_half1 = fig.add_subplot(gs[1, 0])
    ax_half2 = fig.add_subplot(gs[1, 1])

    # ── FSC curve ─────────────────────────────────────────────────────────────
    shell_freqs = freqs[1:]  # skip DC
    ax_fsc.plot(shell_freqs, fsc_raw[1:], color='#aaaaaa', linewidth=1.0,
                label='FSC (raw)', alpha=0.7)
    ax_fsc.plot(shell_freqs, fsc_smooth[1:], color='#1f77b4', linewidth=2.0,
                label='FSC (smoothed)' if not np.array_equal(fsc_raw, fsc_smooth) else 'FSC')
    ax_fsc.plot(shell_freqs, three_sigma[1:], color='#9467bd', linewidth=1.2,
                linestyle=':', label='3σ threshold')
    ax_fsc.plot(shell_freqs, half_bit[1:], color='#8c564b', linewidth=1.0,
                linestyle=':', label='Half-bit threshold', alpha=0.6)

    threshold_styles = {
        'FSC=0.5':   (0.5,   '#d62728', '--'),
        'FSC=0.143': (0.143, '#2ca02c', '--'),
        '3-sigma':   (None,  '#9467bd', ':'),
    }
    for name, (val, color, ls) in threshold_styles.items():
        if val is not None:
            ax_fsc.axhline(val, color=color, linewidth=1.2, linestyle=ls,
                           label=f'{name} = {val}', alpha=0.8)
        res = resolutions.get(name)
        if res is not None and res > 0:
            freq_cross = 1.0 / res
            if freq_cross <= freqs[-1]:
                ax_fsc.axvline(freq_cross, color=color, linewidth=1.0,
                               linestyle=':', alpha=0.6)
                ax_fsc.annotate(f'{res:.1f} px',
                                xy=(freq_cross, 0.02),
                                xytext=(freq_cross + 0.005, 0.08),
                                fontsize=7.5, color=color,
                                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax_fsc.set_xlim(0, freqs[-1])
    ax_fsc.set_ylim(-0.05, 1.05)
    ax_fsc.set_xlabel('Spatial frequency (cycles / pixel)', fontsize=10)
    ax_fsc.set_ylabel('FSC', fontsize=10)
    ax_fsc.set_title('Fourier Shell Correlation\n(half-dataset gold-standard)', fontsize=10)
    ax_fsc.legend(fontsize=7.5, loc='upper right')
    ax_fsc.grid(True, alpha=0.25)

    # Secondary x-axis: resolution in pixels
    ax_top = ax_fsc.twiny()
    ax_top.set_xlim(ax_fsc.get_xlim())
    tick_freqs = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    tick_freqs = tick_freqs[tick_freqs <= freqs[-1]]
    ax_top.set_xticks(tick_freqs)
    ax_top.set_xticklabels([f'{1/f:.1f}' for f in tick_freqs], fontsize=7)
    ax_top.set_xlabel('Resolution (pixels)', fontsize=8)

    # ── Central-slice comparison ───────────────────────────────────────────────
    nz = r1.shape[0]
    mid = nz // 2
    slice1 = r1[mid]
    slice2 = r2[mid]
    vmin = min(slice1.min(), slice2.min())
    vmax = max(slice1.max(), slice2.max())
    kw = dict(cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')

    ax_half1.imshow(slice1, **kw)
    ax_half1.set_title('Half 1 (even angles)\ncentral slice', fontsize=9)
    ax_half1.axis('off')

    im = ax_half2.imshow(slice2, **kw)
    ax_half2.set_title('Half 2 (odd angles)\ncentral slice', fontsize=9)
    ax_half2.axis('off')
    plt.colorbar(im, ax=ax_half2, fraction=0.046, pad=0.04)

    plt.suptitle('FSC Resolution Estimation', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()









    