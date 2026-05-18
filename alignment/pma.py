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
    arr = xp.asarray(img, dtype=xp.float64)
    result = arr - _gaussian_filter(arr, sigma)
    return result.get() if xp is not np else result


def _fourier_gradients(img):
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


def _preprocess_for_matching(img, sigma=2):
    img = img.astype(np.float64)
    img = img - gaussian_filter(img, sigma)  # high-pass
    img /= (np.std(img) + 1e-8)
    return img


def _optical_flow_shift(proj, reproj_img, sigma):
    """Lucas-Kanade optical flow with the full 2x2 system on high-pass-filtered images."""
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
    Projection Matching Alignment with ROI-based alignment (non-destructive cropping).
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
        _xr = xROI_Range
        _yr = yROI_Range
        if _xr is not None or _yr is not None:
            if _xr is None:
                _xr = [0, scaled_projs.shape[2] * downsample_factor]
            if _yr is None:
                _yr = [0, scaled_projs.shape[1] * downsample_factor]
            H, W = scaled_projs.shape[1:]
            x0_ds = int(_xr[0]) // downsample_factor
            x1_ds = int(_xr[1]) // downsample_factor
            y0_ds = int(_yr[0]) // downsample_factor
            y1_ds = int(_yr[1]) // downsample_factor
            print(f"Using ROI: x={_xr}, y={_yr} (downsampled by {downsample_factor}x)")
            print(f"H and W values are {H} and {W}")
            print(f"Downsample ROI bounds are x={x0_ds} to {x1_ds}, y={y0_ds} to {y1_ds}")
            assert 0 <= x0_ds < x1_ds <= W
            assert 0 <= y0_ds < y1_ds <= H

            # The xROI must be centered at the rotation center so the rotation axis
            # stays at the ROI midpoint. This guarantees roi_center = (x1-x0)/2 and
            # avoids a lateral offset in the smaller reconstruction.
            roi_x_center = (x0_ds + x1_ds) / 2.0
            assert abs(roi_x_center - scaled_center) <= 1.0, (
                f"xROI must be centered at the rotation center (scaled_center={scaled_center:.1f}), "
                f"but xROI midpoint is at {roi_x_center:.1f} (diff={abs(roi_x_center - scaled_center):.2f} px). "
                f"Adjust xROI_Range to [{int(scaled_center * downsample_factor) - (_xr[1] - _xr[0]) // 2}, "
                f"{int(scaled_center * downsample_factor) + (_xr[1] - _xr[0]) // 2}]."
            )
            roi_active = True
            # Because xROI is centered on the rotation axis, roi_center is simply the ROI midpoint.
            roi_center = (x1_ds - x0_ds) / 2.0
        else:
            roi_active = False
            roi_center = scaled_center

        for k in tqdm(range(n_iters), desc=f'PMA Level {level} iterations'):
            # Crop projections to ROI before reconstruction so the volume and
            # forward-projection both operate on the smaller ROI domain, which
            # is faster than reconstructing the full volume and cropping after.
            if roi_active:
                recon_projs = scaled_projs[:, y0_ds:y1_ds, x0_ds:x1_ds]
            else:
                recon_projs = scaled_projs

            recon  = _pma_reconstruct(recon_projs, tomo.ang, roi_center, algorithm, recon_crop_ratio)
            reproj = tomo.simulateProjections(recon=recon, pad=False, center=roi_center)
            del recon

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

                if plot and k == 0 and i == plot_idx:
                    plot_ref_raw, plot_mov_raw = ref_roi.copy(), mov_roi.copy()

                if use_matching_preprocess:
                    ref_roi = _preprocess_for_matching(ref_roi, matching_sigma)
                    mov_roi = _preprocess_for_matching(mov_roi, matching_sigma)

                if use_grad:
                    ref_roi = compute_grad_image(ref_roi)
                    mov_roi = compute_grad_image(mov_roi)

                if shift_method == 'optical_flow':
                    dy[i], dx[i] = _optical_flow_shift(mov_roi, ref_roi, of_sigma)
                else:
                    dy[i], dx[i] = _cross_correlation_shift(ref_roi, mov_roi, upsample_factor)

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


# Backwards-compatibility alias for the previous CAPS naming.
PMA = projection_matching_alignment
