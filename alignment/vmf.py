"""
Vertical mass-fluctuation alignment (VMF).

Aligns projections vertically using the per-row mass profile,
windowed and (optionally) gradient-transformed to reject cut-off boundary
artifacts. Robust reference is the mean profile across all angles.
"""

import numpy as np
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

from helperFunctions import subpixel_shift


def vertical_mass_fluctuation_align(
    tomo,
    tolerance=0.0,
    max_iterations=10,
    y_range=None,
    upsample_factor=50,
    window='hanning',     # 'hanning', 'soft_roi', or None — suppresses cut-off boundary artifacts
    roi_sigma=0.3,        # Gaussian half-width as fraction of frame height (only for 'soft_roi')
    use_gradient=True,    # Differentiate mass profile — sensitive to internal features, ignores bulk cut-off
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

            # 3. Intensity Normalization — makes the sum independent of beam intensity
            # fluctuations. Use mean-abs for gradient profiles since they can be signed.
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

        # Robust reference: the average fluctuation across all angles
        ref = np.mean(profiles, axis=0)

        shifts_y = np.zeros(n)
        for i in range(n):
            shift, error, diffphase = phase_cross_correlation(
                ref[:, np.newaxis],
                profiles[i][:, np.newaxis],
                upsample_factor=upsample_factor,
            )
            shifts_y[i] = shift[0]

        # Subtract the mean shift to keep the volume centered in the FOV
        shifts_y -= np.mean(shifts_y)

        shifts_y *= stepRatio

        for i in range(n):
            tomo.workingProjections[i] = subpixel_shift(snapshot[i], shifts_y[i], 0)
            tomo.tracked_shifts[i, 0] += shifts_y[i]

        avg_delta = np.mean(np.abs(shifts_y))
        print(f"  Iteration {iteration + 1}: Mean Correction = {avg_delta:.4f} px")

        if plot:
            _plot_vmf_profiles(win_to_plot, ref, profiles,
                               window=window, iteration=iteration + 1, n=n)

        if avg_delta < tolerance:
            print(f"  Converged.")
            break


def _plot_vmf_profiles(win, ref_profile, profiles, *, window, iteration, n):
    """VMF diagnostic plot: window, reference profile, and a sample projection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"VMF Alignment — Iteration {iteration}")

    axes[0].plot(win)
    axes[0].set_title(f"Window Profile ({window if window else 'none'})")
    axes[0].set_xlabel("Pixel (vertical)"); axes[0].set_ylabel("Weight")

    axes[1].plot(ref_profile)
    axes[1].set_title("Overall Reference Profile\n(mean fluctuation across all angles)")
    axes[1].set_xlabel("Pixel (vertical)"); axes[1].set_ylabel("Fluctuation")

    second_idx = min(1, n - 1)
    axes[2].plot(profiles[second_idx])
    axes[2].set_title(f"Processed Profile — Projection {second_idx}")
    axes[2].set_xlabel("Pixel (vertical)"); axes[2].set_ylabel("Fluctuation")

    plt.tight_layout()
    plt.show()
