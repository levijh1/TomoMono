"""
Fourier Shell Correlation (FSC) resolution metric.

Estimates 3D reconstruction resolution by splitting the tilt series into
two interleaved halves, independently reconstructing each, and comparing
the volumes shell-by-shell in 3D Fourier space.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import tomopy
import matplotlib.pyplot as plt
from matplotlib import gridspec

from gpu import torch, svmbir


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

    # float32 yields complex64 FFTs (8 bytes/voxel vs 16 for float64/complex128).
    F1 = np.fft.fftshift(np.fft.fftn(vol1.astype(np.float32)))
    F2 = np.fft.fftshift(np.fft.fftn(vol2.astype(np.float32)))

    # Radial distance from centre using broadcasting — avoids three full-size meshgrids.
    def _centred_freq(n):
        return np.fft.fftshift(np.fft.fftfreq(n)) * n

    kz = _centred_freq(nz)
    ky = _centred_freq(ny)
    kx = _centred_freq(nx)
    r = np.sqrt(
        kz[:, None, None]**2 + ky[None, :, None]**2 + kx[None, None, :]**2
    )

    max_r = min(nz, ny, nx) // 2
    shells = np.round(r).astype(np.int32)
    del r
    valid = shells <= max_r
    shells = shells[valid]

    # Flatten once and index to avoid creating full-size intermediate arrays
    valid_flat = valid.ravel()
    F1_ravel = F1.ravel()[valid_flat]
    F2_ravel = F2.ravel()[valid_flat]
    del F1, F2

    cross = np.real(F1_ravel * np.conj(F2_ravel))
    p1    = np.abs(F1_ravel)**2
    p2    = np.abs(F2_ravel)**2
    del F1_ravel, F2_ravel

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
    Resolution at the *first* downward crossing of FSC below `threshold`.

    Uses linear interpolation between the last shell where FSC ≥ threshold
    *before that first crossing* and the shell below it.  The first-crossing
    convention is the standard FSC resolution criterion: once the FSC drops
    below the threshold the reconstruction is no longer reliable, and any
    recovery at higher frequencies (noise correlation / reconstruction
    artifacts) must not extend the reported resolution.

    To avoid spurious crossings at the lowest frequencies — where the
    statistical thresholds (3σ, half-bit) can exceed 1.0 while the FSC has
    not yet risen — the search starts from the first shell where FSC is
    actually above the threshold.

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

    # Skip shell 0 (DC) and start looking only once the FSC is above threshold,
    # so low-frequency statistical thresholds > 1 don't trigger a false crossing.
    above = fsc[1:] >= t[1:]
    if not above.any():
        return None
    first_above = int(np.where(above)[0][0]) + 1

    # First shell at or after first_above where FSC drops below threshold.
    below = np.where(fsc[first_above:] < t[first_above:])[0]
    if len(below) == 0:
        return None  # FSC stays above threshold all the way to Nyquist
    cross = first_above + int(below[0])  # first shell below; cross-1 is last above

    i0 = cross - 1
    f1, f2 = fsc[i0], fsc[cross]
    t1, t2 = t[i0], t[cross]
    q1, q2 = freqs[i0], freqs[cross]

    delta = (f1 - t1) - (f2 - t2)
    crossing_freq = q1 + (f1 - t1) / delta * (q2 - q1) if abs(delta) > 1e-12 else q2
    return 1.0 / crossing_freq if crossing_freq > 1e-9 else None


def _fsc_recon_half(projs, angles, center, center_offset, algorithm, min_constraint=None):
    """Reconstruct one FSC half-dataset using the same dispatch as tomoData.reconstruct."""
    recon_kwargs = {}
    if min_constraint is not None:
        recon_kwargs['min_constraint'] = min_constraint

    if algorithm.endswith('CUDA'):
        if torch is None:
            raise ValueError("GPU algorithm requested but CUDA is not available.")
        options = {
            'proj_type': 'cuda',
            'method': algorithm,
            'num_iter': 400,
            'extra_options': {},
        }
        return tomopy.recon(projs, angles, center=center,
                            algorithm=tomopy.astra, options=options, ncore=1,
                            **recon_kwargs)
    elif algorithm == 'svmbir':
        if svmbir is None:
            raise ImportError("svmbir is not installed.")
        return svmbir.recon(projs, angles, center_offset=center_offset, verbose=1)
    else:
        return tomopy.recon(projs, angles, center=center,
                            algorithm=algorithm, sinogram_order=False,
                            **recon_kwargs)


def fourier_shell_correlation(tomo, algorithm='gridrec', plot=True,
                              smooth_sigma=0.0, apply_circ_mask=True,
                              min_constraint=None):
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
        Reconstruction algorithm.  Supports the same values as
        tomoData.reconstruct(): CPU algorithms ('gridrec', 'sirt', 'art', …),
        GPU ASTRA algorithms ('SIRT_CUDA', 'FBP_CUDA', …), and 'svmbir'.
    plot : bool
    smooth_sigma : float
        Standard deviation (in shells) of a Gaussian used to smooth the FSC
        curve before threshold detection.  0 = no smoothing.
    apply_circ_mask : bool
        Apply the same circular mask used during full reconstruction.
    min_constraint : float or None
        If given, passed to tomopy.recon() as a lower-bound floor on voxel
        values after each iteration.  Has no effect for the 'svmbir' algorithm.
        Default None (no constraint).

    Returns
    -------
    fsc : ndarray (n_shells,)
        Raw (unsmoothed) FSC curve.
    resolutions : dict
        Resolution in pixels at each threshold; None if FSC never drops below.
    freqs : ndarray (n_shells,)
        Spatial frequency in cycles/pixel for each shell.
    """
    projs         = tomo.finalProjections
    angles        = tomo.ang
    center        = tomo.rotation_center
    center_offset = getattr(tomo, 'center_offset', 0.0)
    n             = tomo.num_angles

    even_idx = np.arange(0, n, 2)
    odd_idx  = np.arange(1, n, 2)

    print(f"\nFourier Shell Correlation  (algorithm={algorithm})")
    print(f"  Half 1 (even angles): {len(even_idx)} projections")
    print(f"  Half 2 (odd  angles): {len(odd_idx)}  projections")

    print("  Reconstructing half 1 …")
    r1 = _fsc_recon_half(projs[even_idx], angles[even_idx], center, center_offset, algorithm,
                         min_constraint=min_constraint)
    print("  Reconstructing half 2 …")
    r2 = _fsc_recon_half(projs[odd_idx],  angles[odd_idx],  center, center_offset, algorithm,
                         min_constraint=min_constraint)

    if apply_circ_mask:
        r1 = tomopy.circ_mask(r1, axis=0, ratio=0.99)
        r2 = tomopy.circ_mask(r2, axis=0, ratio=0.99)

    print("  Computing FSC …")
    fsc, n_vox, freqs = _fsc_compute_shells(r1, r2)

    n_safe     = np.maximum(n_vox, 1)
    three_sigma = 3.0 / np.sqrt(n_safe)
    half_bit    = (0.2071 + 1.9102 / np.sqrt(n_safe)) / (1.2071 + 0.9102 / np.sqrt(n_safe))

    fsc_smooth = fsc
    if smooth_sigma > 0:
        fsc_smooth = gaussian_filter1d(fsc, sigma=smooth_sigma)

    resolutions = {
        'half-bit':  _fsc_crossing(fsc_smooth, half_bit,     freqs),
        'FSC=0.5':   _fsc_crossing(fsc_smooth, 0.5,         freqs),
        'FSC=0.143': _fsc_crossing(fsc_smooth, 0.143,        freqs),
        '3-sigma':   _fsc_crossing(fsc_smooth, three_sigma,  freqs),
    }

    half_shape = r1.shape
    nyquist_px = 2.0
    print("\n─── Fourier Shell Correlation ────────────────────────────────")
    print(f"  Volume shape (half-map): {half_shape}  |  Nyquist limit: {nyquist_px:.1f} px")
    print(f"  {'Threshold':<20} {'Freq (cyc/px)':>15} {'Resolution (px)':>16}")
    print(f"  {'─'*53}")
    for name, res in resolutions.items():
        marker = " *" if name == 'half-bit' else ""
        if res is None:
            print(f"  {name:<20} {'> Nyquist':>15} {'(not reached)':>16}{marker}")
        else:
            freq = 1.0 / res
            print(f"  {name:<20} {freq:>15.4f} {res:>16.2f}{marker}")
    print("  * default threshold")
    print("──────────────────────────────────────────────────────────────\n")

    if plot:
        _fsc_plot(fsc, fsc_smooth, freqs, three_sigma, half_bit, resolutions, r1, r2)

    del r1, r2

    return fsc, resolutions, freqs


def _fsc_plot(fsc_raw, fsc_smooth, freqs, three_sigma, half_bit, resolutions, r1, r2):
    """Plot FSC curve with thresholds and central-slice half-map comparison."""
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.3)
    ax_fsc   = fig.add_subplot(gs[0, :])
    ax_half1 = fig.add_subplot(gs[1, 0])
    ax_half2 = fig.add_subplot(gs[1, 1])

    shell_freqs = freqs[1:]
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

    ax_top = ax_fsc.twiny()
    ax_top.set_xlim(ax_fsc.get_xlim())
    tick_freqs = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    tick_freqs = tick_freqs[tick_freqs <= freqs[-1]]
    ax_top.set_xticks(tick_freqs)
    ax_top.set_xticklabels([f'{1/f:.1f}' for f in tick_freqs], fontsize=7)
    ax_top.set_xlabel('Resolution (pixels)', fontsize=8)

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
