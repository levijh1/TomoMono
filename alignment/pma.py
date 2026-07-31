"""
Projection Matching Alignment (PMA).

Iterates: reconstruct → forward-project → measure per-projection shift
(phase cross-correlation or Lucas-Kanade optical flow) → apply shifts.
Supports multi-scale alignment, ROI cropping, and matching preprocessing.
"""

import numpy as np
from tqdm import tqdm
import tomopy
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

from gpu import xp, torch, gaussian_filter as _gaussian_filter
from helperFunctions import subpixel_shift
from alignment.cross_correlate import compute_grad_image


def _highpass(img, sigma):
    """High-pass an image by subtracting its Gaussian-blurred self (GPU-dispatched)."""
    arr = xp.asarray(img, dtype=xp.float64)
    result = arr - _gaussian_filter(arr, sigma)
    return result.get() if xp is not np else result


def _fourier_gradients(img):
    """Spatial derivatives (dy, dx) computed in Fourier space — smoother than finite differences."""
    ny, nx = img.shape
    arr = xp.asarray(img)
    F = xp.fft.fft2(arr)
    ux = xp.fft.fftfreq(nx).reshape(1, -1)
    uy = xp.fft.fftfreq(ny).reshape(-1, 1)
    dy = xp.real(xp.fft.ifft2(2j * np.pi * uy * F))
    dx = xp.real(xp.fft.ifft2(2j * np.pi * ux * F))
    if xp is not np:
        return dy.get(), dx.get()
    return dy, dx


def _pma_reconstruct(projs, ang, center, algorithm, ratio=0.98):
    """Reconstruct one PMA iteration's volume (ASTRA GPU or tomopy CPU), circle-masked."""
    if algorithm.endswith("CUDA"):
        if torch is None:
            raise ValueError("GPU requested but torch is unavailable.")
        options = {'proj_type': 'cuda', 'method': algorithm, 'num_iter': 400, 'extra_options': {}}
        recon = tomopy.recon(projs, ang, center=center, algorithm=tomopy.astra, options=options, ncore=1)
    else:
        recon = tomopy.recon(projs, ang, center=center, algorithm=algorithm, sinogram_order=False)
    return tomopy.circ_mask(recon, axis=0, ratio=ratio)


def _margin_pixels(recon_margin, downsample_factor):
    """
    Resolve ``recon_margin`` (full-resolution px, scalar or (y, x)) to a
    (my, mx) pair of non-negative margins in downsampled pixels.

    A requested margin smaller than one downsampled pixel is rounded up to 1 so
    coarse levels still get a buffer; a margin of exactly 0 disables padding.
    """
    if np.isscalar(recon_margin):
        my_full = mx_full = float(recon_margin)
    else:
        my_full, mx_full = (float(v) for v in recon_margin)
    if my_full < 0 or mx_full < 0:
        raise ValueError(f"recon_margin must be non-negative, got {recon_margin}")

    def _scale(m):
        if m == 0:
            return 0
        return max(1, int(round(m / downsample_factor)))

    return _scale(my_full), _scale(mx_full)


def _cross_correlation_shift(ref, mov, upsample_factor):
    """Estimate the (dy, dx) translation between two images by phase cross-correlation."""
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample_factor)
    return float(shift[0]), float(shift[1])


def _preprocess_for_matching(img, sigma=2):
    """High-pass and unit-variance an image so matching keys on edges, not on overall brightness."""
    img = img.astype(np.float64)
    img = img - gaussian_filter(img, sigma)  # high-pass
    img /= (np.std(img) + 1e-8)
    return img


def _optical_flow_shift(proj, reproj_img, sigma):
    """
    Estimate a global (dy, dx) translation using a Lucas-Kanade formulation.

    Solves the 2x2 least-squares system over the whole image to find the single
    rigid shift that best explains the intensity difference between the measured
    projection and its reprojection. The result is a (dy, dx) pair — the image
    is still shifted rigidly, same as phase cross-correlation.

    This is NOT the same as standalone optical_flow_align (legacy.py), which
    computes a dense per-pixel displacement field and deforms the image non-rigidly
    via warp(). Here, "optical flow" refers only to the shift estimation technique.
    """
    r_hp = _highpass(proj - reproj_img, sigma)

    dy_grad, dx_grad = _fourier_gradients(reproj_img)
    hp_dy = _highpass(dy_grad, sigma)
    hp_dx = _highpass(dx_grad, sigma)

    A11 = np.sum(hp_dy * hp_dy)
    A22 = np.sum(hp_dx * hp_dx)
    A12 = np.sum(hp_dy * hp_dx)

    b1 = np.sum(hp_dy * r_hp)
    b2 = np.sum(hp_dx * r_hp)

    det = A11 * A22 - A12 * A12 + 1e-8

    dy = ( A22 * b1 - A12 * b2) / det
    dx = (-A12 * b1 + A11 * b2) / det

    return float(dy), float(dx)


def projection_matching_alignment(
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
        recon_margin=32,
        shift_method='cross_correlation',
        of_sigma=2.0,
        smooth_sigma=None,
        plot=False,

        use_highpass_filter=False,
        matching_sigma=2,
        use_grad=False,

        max_step=0.5,
        stepRatio=1,

        recon_crop_ratio = 0.98
        ):
    """
    Projection Matching Alignment (PMA).

    Each iteration: reconstruct the 3D volume → forward-project at each angle →
    measure the per-projection shift between measured and reprojected images →
    apply a rigid subpixel shift to each projection.

    Multi-scale strategy (levels / scale):
        ``levels`` controls how many resolution levels to run. At level L > 0,
        projections are downsampled by ``scale**L`` before reconstructing and
        measuring shifts. The coarse levels capture large alignment errors cheaply;
        the fine level (level 0, always run at full resolution) refines to
        sub-pixel accuracy. Shifts measured at a coarse level are scaled back up
        before being added to the accumulated shift. Example: levels=2, scale=2
        runs one pass at 2x downsampled then one pass at full resolution.

    recon_margin:
        Extra border, in full-resolution pixels, added around the ROI before
        reconstructing. Each iteration reconstructs on this padded window and
        forward-projects it, then crops both the measured and reprojected
        images back to the exact ROI before measuring shifts. This keeps
        reconstruction edge artifacts (the circ_mask boundary and truncation
        artifacts from the cropped sinogram) out of the compared region, where
        they would otherwise appear in the reprojection but not the measurement
        and bias the shift estimate. Accepts a scalar or a (y_margin, x_margin)
        pair; use 0 to reconstruct on the ROI exactly. The margin is clipped to
        the available frame, so an ROI touching a frame edge gets a smaller
        margin on that side. With no ROI specified there is no surrounding data
        to pad with, so the margin has no effect.

    shift_method:
        'cross_correlation' (default) — phase cross-correlation via skimage.
        'optical_flow' — Lucas-Kanade shift estimation: solves the 2x2 least-
            squares system to find the single global (dy, dx) that best explains
            the image difference. Both methods produce a rigid translation that
            is applied with subpixel_shift. Neither deforms the image pixel-by-pixel.

        NOTE: 'optical_flow' here is entirely different from the standalone
        optical_flow_align() in legacy.py, which computes a dense per-pixel
        displacement field using TV-L1 and deforms the image non-rigidly via
        warp(). That function changes the spatial structure of each projection;
        this shift_method only estimates how far to translate it.

    Other parameters:
    - tomo: Tomography object with .workingProjections and .tracked_shifts.
      Shifts land in workingProjections/tracked_shifts; call make_updates_shift() to commit.
    - max_iterations / iterations_per_level: iteration budget, overall or per level.
    - tolerance (float): stop a level once its average shift drops below this (full-res px).
    - algorithm (str): reconstruction algorithm used each iteration ('art', 'SIRT_CUDA', ...).
    - xROI_Range / yROI_Range (list or None): restrict reconstruction and matching to
      this window. The x range must contain the rotation axis.
    - isPhaseData / standardize (bool): standardize projections (and reprojections)
      to zero mean and unit variance before matching.
    - upsample_factor (int): sub-pixel precision for 'cross_correlation'.
    - of_sigma (float): high-pass sigma for 'optical_flow' shift estimation.
    - smooth_sigma (float or None): smooth the per-angle shifts across angle index,
      suppressing per-projection jitter in favour of smooth drift.
    - use_highpass_filter / matching_sigma: high-pass both images before matching.
    - use_grad (bool): match on gradient magnitude images instead of intensity.
    - max_step (float): clip on each iteration's per-projection shift, for stability.
    - stepRatio (float): fraction of the computed shift applied per iteration (damping).
    - recon_crop_ratio (float): circular mask radius applied to each iteration's volume.
    - plot (bool): show measured / reprojection / difference for one angle on the
      first iteration of each level.
    """
    grad_str = " | gradient mode" if use_grad else ""
    preprocess_str = " | highpass filter" if use_highpass_filter else ""
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
            scaled_projs = zoom(
                current_projs,
                (1, 1.0 / downsample_factor, 1.0 / downsample_factor)
            ).astype(np.float32)
            del current_projs
        else:
            scaled_projs = current_projs

        scaled_center = tomo.rotation_center / downsample_factor
        level_shifts = np.zeros((tomo.num_angles, 2), dtype=np.float64)
        level_snapshot = scaled_projs.copy()

        # ROI setup: validate bounds and compute downsampled indices
        H, W = scaled_projs.shape[1:]
        _xr = xROI_Range
        _yr = yROI_Range
        if _xr is not None or _yr is not None:
            if _xr is None:
                _xr = [0, scaled_projs.shape[2] * downsample_factor]
            if _yr is None:
                _yr = [0, scaled_projs.shape[1] * downsample_factor]
            x0_ds = int(_xr[0]) // downsample_factor
            x1_ds = int(_xr[1]) // downsample_factor
            y0_ds = int(_yr[0]) // downsample_factor
            y1_ds = int(_yr[1]) // downsample_factor
            print(f"Using ROI: x={_xr}, y={_yr} (downsampled by {downsample_factor}x)")
            print(f"H and W values are {H} and {W}")
            print(f"Downsample ROI bounds are x={x0_ds} to {x1_ds}, y={y0_ds} to {y1_ds}")
            assert 0 <= x0_ds < x1_ds <= W
            assert 0 <= y0_ds < y1_ds <= H

            # The rotation axis need not sit at the ROI midpoint. Its position
            # within the cropped ROI is simply the full-frame rotation center
            # minus the ROI's left edge, which is valid for any xROI. This keeps
            # PMA working even when find_center_vo returns a slightly inconsistent
            # rotation center relative to the chosen xROI.
            roi_x_center = (x0_ds + x1_ds) / 2.0
            roi_center = scaled_center - x0_ds
            if abs(roi_x_center - scaled_center) > 1.0:
                print(
                    f"Note: xROI is not centered on the rotation axis "
                    f"(scaled_center={scaled_center:.1f}, xROI midpoint={roi_x_center:.1f}, "
                    f"diff={abs(roi_x_center - scaled_center):.2f} px). Using rotation center "
                    f"offset within ROI (roi_center={roi_center:.2f})."
                )
            # Guard against a rotation axis that falls outside the ROI entirely,
            # which would make the reconstruction geometry meaningless.
            assert 0 <= roi_center <= (x1_ds - x0_ds), (
                f"Rotation center (scaled_center={scaled_center:.1f}) falls outside xROI "
                f"[{x0_ds}, {x1_ds}] (downsampled). Widen xROI_Range to include the rotation axis."
            )
            roi_active = True
        else:
            roi_active = False
            x0_ds, x1_ds, y0_ds, y1_ds = 0, W, 0, H

        # Reconstruction margin: reconstruct from a window slightly larger than
        # the ROI, then crop measured/reprojected images back to the ROI before
        # measuring shifts. Reconstructions carry artifacts at the boundary of
        # their support (the circ_mask edge plus truncation artifacts from the
        # cropped sinogram), and those artifacts land in the reprojections but
        # not in the measured data, biasing the shift estimate. Pushing them
        # outside the compared region removes that bias. The margin is clipped
        # to the available frame, so an ROI already touching an edge simply gets
        # a smaller (or zero) margin on that side.
        my_ds, mx_ds = _margin_pixels(recon_margin, downsample_factor)
        rx0 = max(0, x0_ds - mx_ds)
        rx1 = min(W, x1_ds + mx_ds)
        ry0 = max(0, y0_ds - my_ds)
        ry1 = min(H, y1_ds + my_ds)

        # Offsets of the ROI inside the (larger) reconstruction window.
        ix0, ix1 = x0_ds - rx0, x1_ds - rx0
        iy0, iy1 = y0_ds - ry0, y1_ds - ry0

        # Rotation center is expressed relative to the reconstruction window.
        recon_center = scaled_center - rx0

        if (rx0, rx1, ry0, ry1) != (x0_ds, x1_ds, y0_ds, y1_ds):
            print(
                f"Reconstructing on padded window x={rx0}-{rx1}, y={ry0}-{ry1} "
                f"(margin y={my_ds}, x={mx_ds} ds-px; ROI at x offset {ix0}, y offset {iy0}); "
                f"recon_center={recon_center:.2f}"
            )

        for k in tqdm(range(n_iters), desc=f'PMA Level {level} iterations'):
            # Crop projections to the padded ROI window before reconstruction so
            # the volume and forward-projection both operate on the smaller
            # domain, which is faster than reconstructing the full volume.
            window_projs = scaled_projs[:, ry0:ry1, rx0:rx1]

            recon  = _pma_reconstruct(window_projs, tomo.ang, recon_center, algorithm, recon_crop_ratio)
            reproj = tomo.simulateProjections(recon=recon, pad=False, center=recon_center)
            del recon

            # Discard the margin: compare only the requested ROI, where the
            # reprojection is free of reconstruction-boundary artifacts.
            recon_projs = window_projs[:, iy0:iy1, ix0:ix1]
            reproj = reproj[:, iy0:iy1, ix0:ix1]

            if standardize:
                reproj = (reproj - np.mean(reproj)) / np.std(reproj)

            dy = np.zeros(tomo.num_angles)
            dx = np.zeros(tomo.num_angles)

            plot_idx = tomo.num_angles // 2
            plot_ref_raw = plot_mov_raw = None

            for i in range(tomo.num_angles):
                # reproj and recon_projs are already ROI-sized; no further crop needed
                ref_roi = reproj[i]
                mov_roi = recon_projs[i]

                if use_highpass_filter:
                    ref_roi = _preprocess_for_matching(ref_roi, matching_sigma)
                    mov_roi = _preprocess_for_matching(mov_roi, matching_sigma)

                if use_grad:
                    ref_roi = compute_grad_image(ref_roi)
                    mov_roi = compute_grad_image(mov_roi)

                if shift_method == 'optical_flow':
                    dy[i], dx[i] = _optical_flow_shift(mov_roi, ref_roi, of_sigma)
                elif shift_method == 'cross_correlation':
                    dy[i], dx[i] = _cross_correlation_shift(ref_roi, mov_roi, upsample_factor)
                else: #Return error
                    raise ValueError(f"Unknown shift_method '{shift_method}'")

                if plot and k == 0 and i == plot_idx:
                    plot_ref_raw, plot_mov_raw = ref_roi.copy(), mov_roi.copy()

            if plot and k == 0 and plot_ref_raw is not None:
                _plot_pma_diff(plot_mov_raw, plot_ref_raw,
                               level=level, iteration=k + 1,
                               plot_idx=plot_idx, roi_active=roi_active)

            dy -= np.mean(dy)
            dx -= np.mean(dx)

            if smooth_sigma:
                dy = gaussian_filter1d(dy, sigma=smooth_sigma)
                dx = gaussian_filter1d(dx, sigma=smooth_sigma)

            dy = np.clip(dy, -max_step, max_step)
            dx = np.clip(dx, -max_step, max_step)

            dy *= stepRatio
            dx *= stepRatio

            level_shifts[:, 0] += dy
            level_shifts[:, 1] += dx

            for i in range(tomo.num_angles):
                scaled_projs[i] = subpixel_shift(
                    level_snapshot[i],
                    level_shifts[i, 0],
                    level_shifts[i, 1]
                )

            shift_magnitudes = np.sqrt(dy**2 + dx**2)
            avg_shift = np.mean(shift_magnitudes)
            max_shift = np.max(shift_magnitudes)

            clamp_flag = " [CLAMPED]" if np.isclose(max_shift/stepRatio, max_step) else ""

            print(f"Iteration {k+1}: avg shift = {downsample_factor * avg_shift:.4f} px, "
                  f"max shift = {downsample_factor * max_shift:.4f} px{clamp_flag}")

            if downsample_factor * avg_shift < tolerance:
                print(f"  Convergence at level {level} after {k+1} iterations.")
                break

        pma_shifts += level_shifts * downsample_factor

    for i in range(tomo.num_angles):
        tomo.workingProjections[i] = subpixel_shift(
            original[i],
            pma_shifts[i, 0],
            pma_shifts[i, 1]
        )

    tomo.tracked_shifts += pma_shifts
    print("\nPMA complete.")


def _plot_pma_diff(measured_img, reproj_img, *, level, iteration, plot_idx, roi_active):
    """PMA diagnostic plot: measured, reprojection, and signed difference."""
    vmin = min(measured_img.min(), reproj_img.min())
    vmax = max(measured_img.max(), reproj_img.max())
    diff = measured_img - reproj_img
    abs_max = float(np.abs(diff).max()) or 1.0

    roi_label = " (ROI)" if roi_active else ""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"PMA Level {level} — Iteration {iteration} | Projection {plot_idx}{roi_label}")

    im0 = axes[0].imshow(measured_img, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("Measured projection"); axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(reproj_img, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title("Reprojection"); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap='bwr', aspect='auto', vmin=-abs_max, vmax=abs_max)
    axes[2].set_title("Difference (Measured − Reprojection)"); axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


