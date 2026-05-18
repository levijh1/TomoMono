"""
Cross-correlation alignment (XCA).

Aligns projections by maximizing pairwise cross-correlation between
consecutive (or rolling-median) projections, with shift accumulation
to avoid compounding interpolation error.
"""

import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

from gpu import xp
from helperFunctions import subpixel_shift


def compute_grad_image(image):
    """Gradient magnitude of a 2D image, dispatched to GPU when available."""
    arr = xp.asarray(image)
    gy, gx = xp.gradient(arr)
    # hypot avoids allocating gx², gy², and their sum as separate temporaries
    result = xp.hypot(gx, gy)
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
    - yROI_Range (list): [start, end] y-crop for correlation region. If None, defaults to full height.
    - xROI_Range (list): [start, end] x-crop for correlation region. If None, defaults to full width.
    - isFull360 (bool): If True, also correlates last image against first to close the loop.
    - num_images_for_median (int): If set, use a rolling median of the last K images as
      the reference instead of just the previous image. More robust to noisy projections.
    - upsample_factor (int): Passed to phase_cross_correlation for sub-pixel accuracy.
      upsample_factor=1 gives pixel-accurate shifts; higher values (e.g. 10, 100) give
      finer sub-pixel precision at increased compute cost.
    - downsample (int): Downscale factor applied before shift detection.
      Speeds up correlation on large images; detected shifts are rescaled back automatically.
      Use 1 (default) to skip downsampling.
    - use_grad (bool): If True, replace each reference image with its gradient magnitude before
      computing the cross-correlation. Gradient images are edge-enhanced, which can give
      more accurate shifts when features have strong edges but low overall contrast.
    """
    if xROI_Range is None:
        xROI_Range = [0, tomo.workingProjections.shape[2]]
    if yROI_Range is None:
        yROI_Range = [0, tomo.workingProjections.shape[1]]

    roi_str = f"ROI y={yROI_Range} x={xROI_Range}"
    ds_str = f"{downsample}x downsample" if downsample > 1 else "full resolution"
    grad_str = "gradient mode" if use_grad else "intensity mode"
    print(f"Cross-Correlation Alignment  [{ds_str} | {roi_str} | {grad_str}]")

    n = tomo.num_angles
    K = num_images_for_median if (num_images_for_median is not None and num_images_for_median > 1) else None

    def _crop(img):
        return img[yROI_Range[0]:yROI_Range[1], xROI_Range[0]:xROI_Range[1]]

    def _downsample(img):
        if downsample == 1:
            return img
        return zoom(img, 1.0 / downsample)

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
            _plot_xca_overlay(
                *_plot_data, n=n, downsample=downsample,
                xROI_Range=xROI_Range, yROI_Range=yROI_Range,
                iteration=iteration + 1, _downsample_fn=_downsample,
            )

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


def _plot_xca_overlay(mov_img, ref_img, y_sh, x_sh,
                      *, n, downsample, xROI_Range, yROI_Range,
                      iteration, _downsample_fn):
    """XCA diagnostic plot: reference, moving image, and a red/cyan overlay."""
    mov_ds = _downsample_fn(mov_img)
    ref_ds = _downsample_fn(ref_img)
    print(f"  Projection {n//2} shift: y={y_sh:.4f} px, x={x_sh:.4f} px")
    vmin = min(ref_ds.min(), mov_ds.min())
    vmax = max(ref_ds.max(), mov_ds.max())

    def _norm01(img):
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + 1e-9)

    # Red = reference, Cyan (G+B) = moving; misaligned edges show as red/cyan fringing
    overlay = np.stack([_norm01(ref_ds), _norm01(mov_ds), _norm01(mov_ds)], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Cross-Correlation Alignment — Iteration {iteration}  |  "
        f"Proj {n//2} shift: y={y_sh:.4f}, x={x_sh:.4f} px"
    )

    xlabel = f"X (pixels, {downsample}x downsampled)" if downsample > 1 else "X (pixels)"
    ylabel = f"Y (pixels, {downsample}x downsampled)" if downsample > 1 else "Y (pixels)"

    im0 = axes[0].imshow(ref_ds, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Projection {n//2 - 1} (reference)")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mov_ds, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Projection {n//2} (being aligned)")
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay, aspect='auto')
    axes[2].set_title("Overlay (red = ref, cyan = moving)")
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel(ylabel)

    # ROI rectangle in downsampled-image coordinates
    roi_x0 = xROI_Range[0] / downsample
    roi_y0 = yROI_Range[0] / downsample
    roi_w  = (xROI_Range[1] - xROI_Range[0]) / downsample
    roi_h  = (yROI_Range[1] - yROI_Range[0]) / downsample
    for ax in axes[:2]:
        ax.add_patch(plt.Rectangle(
            (roi_x0, roi_y0), roi_w, roi_h,
            linewidth=2, edgecolor='red', facecolor='none',
        ))
    axes[2].add_patch(plt.Rectangle(
        (roi_x0, roi_y0), roi_w, roi_h,
        linewidth=2, edgecolor='white', facecolor='none',
    ))
    plt.tight_layout()
    plt.show()
