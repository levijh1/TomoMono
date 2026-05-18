"""Reconstruction sharpness metric — per-slice gradient and Laplacian variance."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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

    if percentile_crop is not None:
        lo, hi = np.percentile(vol, [percentile_crop, 100 - percentile_crop])
        vol = np.clip(vol, lo, hi)

    nz = vol.shape[0]
    grad_per_slice = np.zeros(nz)
    lap_per_slice = np.zeros(nz)

    for i in range(nz):
        slc = vol[i]
        gy, gx = np.gradient(slc)
        grad_per_slice[i] = np.mean(np.sqrt(gx**2 + gy**2))
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
            fontsize=12,
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
