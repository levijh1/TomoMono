"""Reprojection Consistency Score (RCS) — per-angle NRMSE between measured and reprojected."""

import numpy as np
from tqdm import tqdm
import tomopy
import matplotlib.pyplot as plt
from matplotlib import gridspec


def reprojection_consistency_score(tomo, plot=True, use_circ_mask=True,
                                   normalize_method='none'):
    """
    Computes the Reprojection Consistency Score (RCS) — Metric 1 for alignment quality.

    For each angle, reprojects the 3D reconstruction and computes the Normalized
    Root-Mean-Squared Error (NRMSE) against the corresponding measured projection.
    Three normalization modes handle algorithms (e.g. SVMBIR) that require rescaled
    or sign-inverted input:

        'none'   (default):
            Raw NRMSE: ‖P_θ − P̂_θ‖ / ‖P_θ‖
            Requires measured and reprojected to share the same scale/sign convention.

        'zscore':
            Z-score each image to zero mean and unit std before comparing:
            NRMSE(θ) = ‖z(P_θ) − z(P̂_θ)‖ / ‖z(P_θ)‖
            Scale- and offset-invariant.  Mathematically sqrt(2(1−r)) where r is
            the per-angle Pearson correlation, so it measures structural similarity
            independent of absolute values or sign.

        'affine':
            Fit the optimal (a, b) to map P̂_θ → P_θ via least squares, then:
            NRMSE(θ) = ‖P_θ − (a·P̂_θ + b)‖ / ‖P_θ − mean(P_θ)‖
            Most robust: handles sign inversions (SVMBIR on inverted phase data),
            arbitrary scale differences, and DC offsets all at once.
            Recommended when comparing reconstructions across different algorithms.

        RCS = mean over all θ of NRMSE(θ)

    A lower score means the reconstruction is more self-consistent with the data.
    Typical ranges (all modes):
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
          When normalize_method != 'none', the image panels show the
          normalized/aligned versions so the visual comparison matches the score.
    use_circ_mask : bool
        If True, applies a circular mask to the reconstruction before reprojecting,
        consistent with the circ_mask applied during reconstruction in this codebase.
    normalize_method : {'none', 'zscore', 'affine'}
        Controls scale normalization before computing NRMSE.  See above.

    Returns
    -------
    rcs : float
        The scalar Reprojection Consistency Score (mean NRMSE across all angles).
    per_angle_nrmse : ndarray (n_angles,)
        The per-angle NRMSE values, useful for diagnosing which angles drive errors.
    reprojections : ndarray (n_angles, ny, nx)
        The synthetic reprojections of the reconstruction, for further inspection.
    """
    _valid_methods = ('none', 'zscore', 'affine')
    if normalize_method not in _valid_methods:
        raise ValueError(f"normalize_method must be one of {_valid_methods}, "
                         f"got {normalize_method!r}")
    if not hasattr(tomo, 'recon') or tomo.recon is None:
        raise AttributeError(
            "tomo.recon is not set. Run tomo.reconstruct() before calling this function."
        )

    measured = tomo.finalProjections
    angles   = tomo.ang
    n_angles = tomo.num_angles

    nx_m = measured.shape[2]

    if (hasattr(tomo, 'finalReprojections')
            and tomo.finalReprojections is not None
            and tomo.finalReprojections.shape[2] == nx_m):
        print("Reusing cached reprojections from tomo.finalReprojections.")
        reprojections = tomo.finalReprojections
    else:
        if use_circ_mask:
            recon = tomopy.circ_mask(tomo.recon.copy(), axis=0, ratio=0.99)
        else:
            recon = tomo.recon

        print("Computing reprojections of reconstruction...")
        raw = tomo.simulateProjections(recon=recon, emission=True, pad=False, ncore=None)
        del recon

        # tomopy pads the reconstruction volume, so reprojections may be wider
        # than the measured projections. Crop the center columns to match.
        nx_r = raw.shape[2]
        if nx_r != nx_m:
            start = (nx_r - nx_m) // 2
            raw = raw[:, :, start:start + nx_m]
        reprojections = raw
        tomo.finalReprojections = reprojections

    def _zscore(x):
        mu, sigma = x.mean(), x.std()
        return (x - mu) / sigma if sigma > 1e-12 else (x - mu)

    def _affine_align(reproj, meas):
        """Return reproj rescaled to best match meas via least-squares (a*reproj + b)."""
        A = np.column_stack([reproj.ravel(), np.ones(reproj.size)])
        coeff = np.linalg.lstsq(A, meas.ravel(), rcond=None)[0]
        return coeff[0] * reproj + coeff[1]

    print(f"Computing per-angle NRMSE (normalize_method={normalize_method!r})...")
    per_angle_nrmse = np.zeros(n_angles)
    _plot_pairs = {}

    for i in tqdm(range(n_angles), desc="NRMSE per angle"):
        meas   = measured[i].astype(np.float64)
        reproj = reprojections[i].astype(np.float64)

        if normalize_method == 'zscore':
            meas_std, reproj_std = meas.std(), reproj.std()
            if meas_std < 1e-12 or reproj_std < 1e-12:
                per_angle_nrmse[i] = np.nan
                continue
            m_n = _zscore(meas)
            r_n = _zscore(reproj)
            denom = np.linalg.norm(m_n)
            per_angle_nrmse[i] = np.linalg.norm(m_n - r_n) / denom if denom > 1e-12 else np.nan
            if plot:
                _plot_pairs[i] = (m_n, r_n)

        elif normalize_method == 'affine':
            reproj_aligned = _affine_align(reproj, meas)
            denom = np.linalg.norm(meas - meas.mean())
            per_angle_nrmse[i] = np.linalg.norm(meas - reproj_aligned) / denom if denom > 1e-12 else np.nan
            if plot:
                _plot_pairs[i] = (meas, reproj_aligned)

        else:  # 'none'
            meas_norm = np.linalg.norm(meas)
            per_angle_nrmse[i] = np.linalg.norm(meas - reproj) / meas_norm if meas_norm > 1e-12 else np.nan
            if plot:
                _plot_pairs[i] = (meas, reproj)

    valid_mask = ~np.isnan(per_angle_nrmse)
    rcs = float(np.mean(per_angle_nrmse[valid_mask]))

    worst_idx = int(np.nanargmax(per_angle_nrmse))
    best_idx  = int(np.nanargmin(per_angle_nrmse))
    print("\n─── Reprojection Consistency Score ───────────────────────")
    print(f"  normalize_method:   {normalize_method!r}")
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

    _plot_rcs_diagnostics(
        per_angle_nrmse, angles, rcs, verdict,
        best_idx, worst_idx, _plot_pairs, normalize_method,
    )

    return rcs, per_angle_nrmse, reprojections


def _plot_rcs_diagnostics(per_angle_nrmse, angles, rcs, verdict,
                          best_idx, worst_idx, _plot_pairs, normalize_method):
    """RCS diagnostic plot: bar chart + 2x2 image grid (best/worst angles)."""
    angle_deg = np.degrees(angles.ravel())

    best_meas,  best_reproj  = _plot_pairs[best_idx]
    worst_meas, worst_reproj = _plot_pairs[worst_idx]

    _norm_label = {
        'none':   'Original',
        'zscore': 'Z-scored',
        'affine': 'Affine-aligned reprojection',
    }[normalize_method]

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.5, 1.5], hspace=0.4, wspace=0.3)

    fig.suptitle(f"Reprojection Consistency Score = {rcs:.4f}   |   {verdict}",
                 fontsize=10)

    ax_bar          = fig.add_subplot(gs[0, :])
    ax_best_orig    = fig.add_subplot(gs[1, 0])
    ax_best_reproj  = fig.add_subplot(gs[1, 1])
    ax_worst_orig   = fig.add_subplot(gs[2, 0])
    ax_worst_reproj = fig.add_subplot(gs[2, 1])

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

    def _shared_clim(a, b):
        return min(a.min(), b.min()), max(a.max(), b.max())

    best_vmin, best_vmax = _shared_clim(best_meas, best_reproj)
    worst_vmin, worst_vmax = _shared_clim(worst_meas, worst_reproj)

    kw = dict(aspect='auto', cmap='gray')
    tick_kw = dict(labelsize=7)

    def _set_img_axes(ax):
        ax.tick_params(axis='both', **tick_kw)
        ax.set_xlabel("Width (px)", fontsize=8)
        ax.set_ylabel("Height (px)", fontsize=8)

    _meas_label = "Measured (z-scored)" if normalize_method == 'zscore' else "Measured"
    _reproj_label = _norm_label if normalize_method != 'none' else "Reprojection"

    im = ax_best_orig.imshow(best_meas, vmin=best_vmin, vmax=best_vmax, **kw)
    ax_best_orig.set_title(f"Best {angle_deg[best_idx]:.1f}° — {_meas_label}\nNRMSE={per_angle_nrmse[best_idx]:.4f}", fontsize=9)
    _set_img_axes(ax_best_orig)
    plt.colorbar(im, ax=ax_best_orig, fraction=0.046, pad=0.04)

    im = ax_best_reproj.imshow(best_reproj, vmin=best_vmin, vmax=best_vmax, **kw)
    ax_best_reproj.set_title(f"Best {angle_deg[best_idx]:.1f}° — {_reproj_label}", fontsize=9)
    _set_img_axes(ax_best_reproj)
    plt.colorbar(im, ax=ax_best_reproj, fraction=0.046, pad=0.04)

    im = ax_worst_orig.imshow(worst_meas, vmin=worst_vmin, vmax=worst_vmax, **kw)
    ax_worst_orig.set_title(f"Worst {angle_deg[worst_idx]:.1f}° — {_meas_label}\nNRMSE={per_angle_nrmse[worst_idx]:.4f}", fontsize=9)
    _set_img_axes(ax_worst_orig)
    plt.colorbar(im, ax=ax_worst_orig, fraction=0.046, pad=0.04)

    im = ax_worst_reproj.imshow(worst_reproj, vmin=worst_vmin, vmax=worst_vmax, **kw)
    ax_worst_reproj.set_title(f"Worst {angle_deg[worst_idx]:.1f}° — {_reproj_label}", fontsize=9)
    _set_img_axes(ax_worst_reproj)
    plt.colorbar(im, ax=ax_worst_reproj, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
