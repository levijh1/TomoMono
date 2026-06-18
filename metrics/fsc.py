"""
Fourier Shell Correlation (FSC) resolution metric.

Estimates 3D reconstruction resolution by splitting the tilt series into
two interleaved halves, independently reconstructing each, and comparing
the volumes shell-by-shell in 3D Fourier space.

This is the **best metric of reconstruction quality** in this toolkit. Because
the two half-volumes are reconstructed from disjoint data, their agreement is a
genuine, noise-independent measure of how much real structure was recovered, and
the crossing frequency is a physical resolution in pixels (or nm). Use FSC to
judge how good a *reconstruction* is. (For judging *alignment* quality, use
``reprojection_consistency_score`` instead.)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import tomopy
import matplotlib.pyplot as plt

from gpu import torch, svmbir


def _pad_taper_3d(vol, pad_length, taper_length):
    """
    Pad a 3D volume on all six faces, then taper its edges smoothly to zero.

    The padding replicates the existing edge voxel values outward by
    ``pad_length`` voxels per face (``np.pad(mode='edge')``).  A separable
    Hann (raised-cosine) taper then ramps the outermost ``taper_length``
    voxels of every axis from full value down to zero at the outer edge.

    Removing the sharp real-space support boundary this way suppresses the
    correlated high-frequency ringing that otherwise leaks into both half-maps
    and makes the FSC curve dip then rise back toward 1.0 at high shells (cf.
    van Heel & Schatz 2005).  The replicated-edge pad keeps that taper from
    eating into the reconstructed object, and the extra extent zero-fills the
    FFT grid for finer radial-shell sampling.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    pad_length : int     — voxels of edge-replicated padding added per face.
    taper_length : int   — width (voxels) of the raised-cosine roll-off to zero
                           at each edge of the padded volume.

    Returns
    -------
    ndarray (nz + 2*pad_length, …)  — padded, tapered, float32.
    """
    vol = vol.astype(np.float32)
    if pad_length and pad_length > 0:
        vol = np.pad(vol, pad_length, mode='edge')

    if taper_length and taper_length > 0:
        window = np.ones(1, dtype=np.float32)
        for axis, n in enumerate(vol.shape):
            t = min(int(taper_length), n // 2)
            w = np.ones(n, dtype=np.float32)
            if t > 0:
                # Ascending half of a Hann window: 0 at the outer edge → 1 by
                # depth t.  sin^2 is equivalent to the raised cosine 0.5(1-cos).
                ramp = np.sin(0.5 * np.pi * np.arange(t, dtype=np.float32) / t) ** 2
                w[:t] = ramp
                w[n - t:] = ramp[::-1]
            shape_b = [1] * vol.ndim
            shape_b[axis] = n
            window = window * w.reshape(shape_b)
        vol = vol * window

    return vol


def _soft_circ_mask(shape, ratio=0.99, taper=0.1):
    """
    Soft-edged circular field-of-view mask in the (axis-1, axis-2) plane,
    broadcast over axis 0.

    Drop-in replacement for tomopy.circ_mask's *hard* edge: a raised-cosine
    roll-off of relative width `taper` (as a fraction of the mask radius)
    between the inner and outer radius.  The hard mask edge is byte-for-byte
    identical in both half-maps, so its ringing correlates near-perfectly and
    inflates the FSC at high frequency; the soft edge removes that.

    Parameters
    ----------
    shape : (nz, ny, nx)
    ratio : float   — outer mask radius as a fraction of min(ny, nx)/2.
    taper : float   — width of the cosine roll-off as a fraction of the radius.
    """
    nz, ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy = np.arange(ny) - cy
    xx = np.arange(nx) - cx
    rr = np.sqrt(yy[:, None]**2 + xx[None, :]**2)

    r_out = ratio * min(ny, nx) / 2.0
    r_in  = r_out * (1.0 - taper)
    mask = np.ones((ny, nx), dtype=np.float32)
    edge = (rr > r_in) & (rr <= r_out)
    mask[edge] = 0.5 * (1.0 + np.cos(np.pi * (rr[edge] - r_in) / (r_out - r_in)))
    mask[rr > r_out] = 0.0
    return mask[None, :, :]


def _fsc_compute_shells(vol1, vol2):
    """
    Core FSC computation via 3D FFT + vectorised bincount.

    Edge tapering (pad+taper, see _pad_taper_3d) is expected to be applied by
    the caller before this function, so the volumes can be passed straight into
    the FFT here.

    Returns
    -------
    fsc : ndarray (max_r+1,)          — FSC value per radial shell
    n_vox : ndarray int (max_r+1,)    — voxels per shell (for σ thresholds)
    freqs : ndarray (max_r+1,)        — spatial frequency in cycles/pixel (0 … 0.5)
    """
    nz, ny, nx = vol1.shape

    v1 = vol1.astype(np.float32)
    v2 = vol2.astype(np.float32)

    # float32 yields complex64 FFTs (8 bytes/voxel vs 16 for float64/complex128).
    F1 = np.fft.fftshift(np.fft.fftn(v1))
    F2 = np.fft.fftshift(np.fft.fftn(v2))
    del v1, v2

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


def _show_mask_slices(before_slices, r1_masked, shape):
    """Orthogonal slices of half-map 1 before and after the soft circular mask."""
    nz, ny, nx = shape
    iz, iy, ix = nz // 2, ny // 2, nx // 2

    after_slices = (
        r1_masked[iz, :, :],
        r1_masked[:, iy, :],
        r1_masked[:, :, ix],
    )
    slice_names = [
        f'Axial  (Z={iz})',
        f'Coronal  (Y={iy})',
        f'Sagittal  (X={ix})',
    ]

    all_before = np.concatenate([s.ravel() for s in before_slices])
    vmin, vmax = float(np.percentile(all_before, 2)), float(np.percentile(all_before, 98))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (sl, name) in enumerate(zip(before_slices, slice_names)):
        axes[0, col].imshow(sl, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis('off')
    for col, sl in enumerate(after_slices):
        axes[1, col].imshow(sl, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, col].axis('off')

    fig.text(0.02, 0.73, 'Before mask', va='center', rotation='vertical', fontsize=11)
    fig.text(0.02, 0.27, 'After mask',  va='center', rotation='vertical', fontsize=11)
    fig.suptitle('Soft Circular Mask — Orthogonal Slices (half-map 1)', fontsize=12)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    plt.show()


def _square_crop_xy(vol):
    """
    Crop XY plane to the largest square inscribed in the circular FOV.

    For a circular mask of diameter min(ny, nx) the inscribed square has
    side = diameter / sqrt(2).  The crop boundary lies entirely inside the
    circle, so there is no masked-edge discontinuity and no correlated
    high-frequency artifact in the FSC curve.
    """
    nz, ny, nx = vol.shape
    side = int(min(ny, nx) / np.sqrt(2))
    if side % 2 != 0:
        side -= 1
    cy, cx = ny // 2, nx // 2
    hy = hx = side // 2
    return vol[:, cy - hy : cy + hy, cx - hx : cx + hx]


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
                              smooth_sigma=2.0, crop_mode='soft_circle',
                              pad_length=50, taper_length=None,
                              min_constraint=None,
                              threshold='half-bit', pixel_size_nm=None):
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
        curve before threshold detection.  0 = no smoothing.  Defaults to 2.0
        so the reported crossing is stable against single-shell noise dips.
    crop_mode : str or None
        Controls how each half-map is prepared before the FSC:

        ``'soft_circle'`` (default)
            Apply a soft-edged (raised-cosine) circular mask that tapers to
            zero at the FOV boundary.  Avoids the correlated high-frequency
            ringing a hard mask edge would inject into both half-maps.  The
            pad+taper step (see ``pad_length``) is still applied afterward.

        ``'square'``
            Crop the XY plane to the largest square inscribed in the circular
            FOV (side = min(ny,nx) / √2).  The crop boundary lies entirely
            inside the circle, so no masking discontinuity exists and the FSC
            curve is free of masked-edge artifacts.  No soft filter is applied.

        ``None``
            No masking or cropping.  Use only when the reconstruction already
            fills the volume without a circular support constraint.
    pad_length : int
        Voxels of edge-replicated padding added to all six faces of each
        half-map, applied *after* any ``crop_mode`` masking/cropping.  The
        outer edges are then tapered to zero (see ``taper_length``), which
        removes the sharp real-space support boundary and suppresses the
        high-frequency FSC rise-back artifact (replacing the former Hann
        apodization).  Default 50.  Set to 0 to disable padding/tapering.
    taper_length : int or None
        Width (voxels) of the raised-cosine roll-off to zero at each edge of
        the padded volume.  When None (default) it equals ``pad_length`` so the
        taper occupies exactly the replicated-edge pad strip; pass a smaller
        value to keep an inner band of the pad at full edge value, or a larger
        value to let the taper extend into the reconstructed object.
    min_constraint : float or None
        If given, passed to tomopy.recon() as a lower-bound floor on voxel
        values after each iteration.  Has no effect for the 'svmbir' algorithm.
        Default None (no constraint).
    threshold : str
        Resolution criterion used for the plot and the console table.
        Options: ``'half-bit'`` (default), ``'FSC=0.5'``, ``'FSC=0.143'``,
        ``'3-sigma'``.  All four are still computed and returned; only the
        selected one is highlighted in the figure.
    pixel_size_nm : float or None
        Physical pixel size in nanometres.  When provided, the top x-axis of
        the plot shows resolution in nm and the console table converts to nm.

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

    _before = None
    if crop_mode == 'soft_circle':
        soft_mask = _soft_circ_mask(r1.shape, ratio=0.99, taper=0.1)
        _iz, _iy, _ix = r1.shape[0] // 2, r1.shape[1] // 2, r1.shape[2] // 2
        _before = (r1[_iz, :, :].copy(), r1[:, _iy, :].copy(), r1[:, :, _ix].copy())
        r1 = r1 * soft_mask
        r2 = r2 * soft_mask
    elif crop_mode == 'square':
        orig_ny, orig_nx = r1.shape[1], r1.shape[2]
        _iz, _iy, _ix = r1.shape[0] // 2, r1.shape[1] // 2, r1.shape[2] // 2
        _before = (r1[_iz, :, :].copy(), r1[:, _iy, :].copy(), r1[:, :, _ix].copy())
        r1 = _square_crop_xy(r1)
        r2 = _square_crop_xy(r2)
        print(f"  Square crop: XY {orig_ny}×{orig_nx} → {r1.shape[1]}×{r1.shape[2]} px "
              f"(inscribed square, no masking)")
    elif crop_mode is not None:
        raise ValueError(f"crop_mode must be 'soft_circle', 'square', or None; got {crop_mode!r}")

    # Pad each half-map by replicating edge voxels, then taper the outer edges
    # to zero (Hann roll-off).  Done after any crop_mode masking/cropping; this
    # removes the sharp real-space support boundary that causes the
    # high-frequency FSC rise-back artifact, and zero-fills the FFT grid.
    if pad_length and pad_length > 0:
        tlen = pad_length if taper_length is None else taper_length
        r1 = _pad_taper_3d(r1, pad_length, tlen)
        r2 = _pad_taper_3d(r2, pad_length, tlen)
        print(f"  Pad+taper: +{pad_length} px/face (edge-replicated), "
              f"{tlen} px Hann taper to zero → half-map shape {r1.shape}")

    # Show the half-map slices after masking *and* pad+taper, so the display
    # reflects exactly what goes into the FFT.
    if _before is not None:
        _show_mask_slices(_before, r1, r1.shape)

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
        marker = " *" if name == threshold else ""
        if res is None:
            print(f"  {name:<20} {'> Nyquist':>15} {'(not reached)':>16}{marker}")
        else:
            freq = 1.0 / res
            print(f"  {name:<20} {freq:>15.4f} {res:>16.2f}{marker}")
    print(f"  * selected threshold ({threshold})")
    print("──────────────────────────────────────────────────────────────\n")

    if plot:
        _fsc_plot(fsc, fsc_smooth, freqs, three_sigma, half_bit, resolutions,
                  threshold, pixel_size_nm)

    del r1, r2

    return fsc, resolutions, freqs


def _fsc_plot(fsc_raw, fsc_smooth, freqs, three_sigma, half_bit, resolutions,
              threshold='half-bit', pixel_size_nm=None):
    """Plot FSC curve with the selected threshold and crossing frequency."""
    _thresh_options = {
        'half-bit':  (half_bit,    'half-bit criteria'),
        'FSC=0.5':   (0.5,         'FSC = 0.5'),
        'FSC=0.143': (0.143,       'FSC = 0.143'),
        '3-sigma':   (three_sigma, '3σ criteria'),
    }
    thresh_val, thresh_label = _thresh_options.get(threshold, _thresh_options['half-bit'])

    fig, ax = plt.subplots(figsize=(8, 5))
    shell_freqs = freqs[1:]
    smoothed = not np.array_equal(fsc_raw, fsc_smooth)

    ax.plot(shell_freqs, fsc_raw[1:], color='blue', linewidth=1.5, label='FSC')
    if smoothed:
        ax.plot(shell_freqs, fsc_smooth[1:], color='red', linewidth=2.0,
                label='Smoothed FSC')

    if isinstance(thresh_val, np.ndarray):
        ax.plot(shell_freqs, thresh_val[1:], color='black', linewidth=1.2,
                linestyle='--', label=thresh_label)
    else:
        ax.axhline(thresh_val, color='black', linewidth=1.2, linestyle='--',
                   label=thresh_label)

    res = resolutions.get(threshold)
    if res is not None and res > 0:
        freq_cross = 1.0 / res
        if freq_cross <= freqs[-1]:
            ax.axvline(freq_cross, color='green', linewidth=1.2, linestyle=':',
                       label=f'{freq_cross:.3f} pixel⁻¹')
            if pixel_size_nm is not None:
                res_text = f'Resolution: {res * pixel_size_nm:.0f} nm'
            else:
                res_text = f'Resolution: {res:.1f} px'
            ax.text(0.25, 0.45, res_text, transform=ax.transAxes, fontsize=11)

    ax.set_xlim(0, freqs[-1])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Spatial frequency (pixel⁻¹)', fontsize=11)
    ax.set_ylabel('Correlation', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    if pixel_size_nm is not None:
        ax_top.set_xlabel('Resolution (nm)', fontsize=11)
        cands_nm = np.array([500, 300, 200, 150, 100, 75, 60, 50, 40, 30])
        cand_freqs = pixel_size_nm / cands_nm
        ok = (cand_freqs > freqs[1]) & (cand_freqs <= freqs[-1])
        ax_top.set_xticks(cand_freqs[ok])
        ax_top.set_xticklabels([str(n) for n in cands_nm[ok]], fontsize=9)
    else:
        ax_top.set_xlabel('Resolution (pixels)', fontsize=11)
        cand_freqs = np.array([0.05, 0.1, 0.2, 0.25, 0.33, 0.5])
        ok = cand_freqs <= freqs[-1]
        ax_top.set_xticks(cand_freqs[ok])
        ax_top.set_xticklabels([f'{1/f:.0f}' for f in cand_freqs[ok]], fontsize=9)

    plt.tight_layout()
    plt.show()
