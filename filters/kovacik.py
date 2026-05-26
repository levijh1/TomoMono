"""
Kovacik soft Fourier angular filter for missing-wedge ray artifacts.

Reference: Kovacik et al. (2014) J. Struct. Biol. 186, 141-152.
"""

import numpy as np
import matplotlib.pyplot as plt

from gpu import cp


def kovacik_filter(tomo, tilt_max=None,
                   mwr_length=20, mwr_order=4, mwr_wmin=0.2,
                   cs_order=4, cs_cutoff=10,
                   plot=False, plotSlice=None, only_slice=None):
    """
    Apply the Kovacik soft Fourier angular filter to ``tomo.recon`` in-place.

    Suppresses missing-wedge ray artifacts by smoothing the sharp Fourier-space
    transition between the sampled data region and the missing wedge. The
    filter Omega_A = max(MWR, CS) consists of:

      MWR (Missing Wedge Ramps): Butterworth ramps adjacent to the boundaries
        of the highest-tilt projections, transitioning from 1.0 (interior) to
        ``mwr_wmin`` (at the boundary).

      CS (Central Stripe): a Butterworth ramp along the kx-axis that protects
        low spatial frequencies from attenuation.

    Always filters from the original recon stashed in ``tomo._recon_pre_kovacik``
    so repeated calls do not compound.

    Parameters
    ----------
    tomo : tomoData
    tilt_max : float, optional
        Maximum tilt angle in radians (full range is ±tilt_max). Defaults to
        ``max(tomo.ang)``.
    mwr_length : int
        MWR half-width in pixels. Larger values give a smoother ramp.
    mwr_order : int
        Butterworth order for the MWR (2 or 4 recommended).
    mwr_wmin : float
        Minimum filter weight at the tilt boundary (0–1). Lower values suppress
        artifacts more aggressively.
    cs_order : int
        Butterworth order for the central stripe.
    cs_cutoff : int
        Half-power radius (pixels) of the central stripe. Equals 1 at kz=0,
        falls to 0.5 at |kz| = cs_cutoff.
    plot : bool
        If True, display before/after slices, before/after FFTs, the filter,
        and the difference image.
    plotSlice : int, optional
        Z-slice index to display in diagnostic plots; defaults to the middle.
    only_slice : int or tuple of (int, int), optional
        If specified, only filter this single slice (int) or range of slices
        (tuple: start_idx, end_idx inclusive). Useful for parameter tuning
        before filtering the full volume. Default None filters all slices.
    """
    if tilt_max is None:
        tilt_max = np.max(tomo.ang)
    if tomo.recon is None:
        raise ValueError("No reconstruction found. Run reconstruct() first.")

    # Always filter from the original recon so repeated calls don't stack.
    if not hasattr(tomo, '_recon_pre_kovacik') or tomo._recon_pre_kovacik is None:
        tomo._recon_pre_kovacik = tomo.recon.copy()
    source = tomo._recon_pre_kovacik

    nz, ny, nx = source.shape
    tilt_max_rad = tilt_max

    # Parse only_slice parameter to determine slice range to filter.
    if only_slice is not None:
        if isinstance(only_slice, int):
            z_start, z_end = only_slice, only_slice
        elif isinstance(only_slice, (tuple, list)) and len(only_slice) == 2:
            z_start, z_end = only_slice[0], only_slice[1]
        else:
            raise ValueError("only_slice must be an int or (start, end) tuple")
        z_indices = np.arange(z_start, z_end + 1)
    else:
        z_indices = np.arange(nz)

    if only_slice is not None:
        plotSlice = z_start  # Always show the filtered slice(s) when only_slice is specified
    elif plot is True and plotSlice is None:
        plotSlice = nz // 2  # Default to middle slice when filtering entire volume

    # Build the 2D angular filter in fftfreq coordinate order.
    # Each XY slice recon[z, :, :] is the reconstruction plane; the missing
    # wedge shows up in the KX-KY Fourier plane of these slices.
    kx_1d = np.fft.fftfreq(nx) * nx
    ky_1d = np.fft.fftfreq(ny) * ny
    kx, ky = np.meshgrid(kx_1d, ky_1d)

    R = np.sqrt(kx ** 2 + ky ** 2)

    # Fold into the first quadrant — real-space XY slices have a Hermitian FFT,
    # so the filter has the same 4-fold symmetry.
    angle_from_x = np.arctan2(np.abs(ky), np.abs(kx))

    # MWR: perpendicular pixel distance from each point to the nearest tilt-
    # boundary line. Inside the data region this is R * sin(tilt_max - angle);
    # outside, max() clamps to 0 → weight = mwr_wmin.
    dist_to_boundary = R * np.sin(np.maximum(tilt_max_rad - angle_from_x, 0.0))

    safe_length = max(mwr_length, 1e-6)
    t = (dist_to_boundary / safe_length) ** (2 * mwr_order)
    mwr = mwr_wmin + (1.0 - mwr_wmin) * t / (1.0 + t)
    mwr[angle_from_x > tilt_max_rad] = mwr_wmin

    safe_cutoff = max(cs_cutoff, 1e-6)
    cs = 1.0 / (1.0 + (np.abs(ky) / safe_cutoff) ** (2 * cs_order))

    angular_filter = np.maximum(mwr, cs)

    # Apply vectorized over Z using rfft2 (real input → half-spectrum).
    # rfft2 output is (nz, ny, nx//2+1) — half the complex memory of fft2.
    # The filter is real and even-symmetric, so irfft2 is equivalent to
    # real(ifft2) for this multiplication.
    if only_slice is not None:
        print(f"Applying Kovacik filter to slice(s) {z_start}–{z_end}...")
    else:
        print("Applying Kovacik filter...")

    filt_r = angular_filter[:, :nx // 2 + 1]

    # Start with a copy of the source (to preserve non-filtered slices).
    filtered = source.copy()

    # Filter only the selected slices.
    to_filter = source[z_indices]
    if cp is not None:
        src_gpu = cp.asarray(to_filter)
        filt_gpu = cp.asarray(filt_r)
        rfft_gpu = cp.fft.rfft2(src_gpu, axes=(1, 2))
        filt_result = cp.asnumpy(
            cp.fft.irfft2(rfft_gpu * filt_gpu[np.newaxis], s=(ny, nx), axes=(1, 2))
        ).astype(source.dtype, copy=False)
    else:
        rfft = np.fft.rfft2(to_filter, axes=(1, 2))
        filt_result = np.fft.irfft2(
            rfft * filt_r[np.newaxis], s=(ny, nx), axes=(1, 2)
        ).astype(source.dtype, copy=False)

    filtered[z_indices] = filt_result
    tomo.recon = filtered

    if plot:
        _plot_kovacik_diagnostics(source, filtered, angular_filter, plotSlice)

    print("Kovacik filter applied.")


def _plot_kovacik_diagnostics(source, filtered, angular_filter, plotSlice):
    """Kovacik filter diagnostic plot: before/after slices, FFTs, filter, and difference."""
    cz = plotSlice
    orig_slice = source[cz, :, :]
    filt_slice = filtered[cz, :, :]
    diff_slice = filt_slice - orig_slice

    fft_orig = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(orig_slice))))
    fft_filt = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(filt_slice))))
    filter_display = np.fft.fftshift(angular_filter)

    vmin, vmax = np.percentile(orig_slice, [1, 99])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Kovacik Filter — Central Slice Diagnostics", fontsize=14, y=1.01)

    axes[0, 0].imshow(orig_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Central Slice — Before"); axes[0, 0].axis("off")

    axes[0, 1].imshow(filt_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Central Slice — After"); axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(diff_slice, cmap="bwr",
                                vmin=-np.abs(diff_slice).max(),
                                vmax=np.abs(diff_slice).max())
    axes[0, 2].set_title("Difference (After − Before)"); axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    fft_vmin, fft_vmax = np.percentile(np.concatenate([fft_orig.ravel(), fft_filt.ravel()]), [1, 99])
    axes[1, 0].imshow(fft_orig, cmap="inferno", vmin=fft_vmin, vmax=fft_vmax)
    axes[1, 0].set_title("FFT Magnitude — Before (log)"); axes[1, 0].axis("off")

    axes[1, 1].imshow(fft_filt, cmap="inferno", vmin=fft_vmin, vmax=fft_vmax)
    axes[1, 1].set_title("FFT Magnitude — After (log)"); axes[1, 1].axis("off")

    im_flt = axes[1, 2].imshow(filter_display, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("Kovacik Filter (Fourier space)"); axes[1, 2].axis("off")
    fig.colorbar(im_flt, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
