"""Sinogram consistency score — Helgason-Ludwig conditions on the center of mass."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tomopy


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

        sino_data_sorted = sinogram_data[order]
        vmin = sino_data_sorted.min()
        vmax = sino_data_sorted.max()

        im_d = ax_sino_data.imshow(
            sino_data_sorted, aspect='auto', vmin=vmin, vmax=vmax,
            extent=[0, nx, ad.max(), ad.min()],
        )
        ax_sino_data.set_title('Data Sinogram (central slice)')
        ax_sino_data.set_ylabel('Angle (deg)')
        ax_sino_data.set_xlabel('Detector pixel')
        plt.colorbar(im_d, ax=ax_sino_data)

        if reprojections is not None:
            sino_reproj = reprojections[:, center_row, :]
            sino_reproj_sorted = sino_reproj[order]
            vmin_r = min(vmin, sino_reproj_sorted.min())
            vmax_r = max(vmax, sino_reproj_sorted.max())
            im_r = ax_sino_reproj.imshow(
                sino_reproj_sorted, aspect='auto', vmin=vmin_r, vmax=vmax_r,
                extent=[0, nx, ad.max(), ad.min()],
            )
            ax_sino_reproj.set_title('Reprojected Sinogram (central slice)')
            plt.colorbar(im_r, ax=ax_sino_reproj)
        else:
            ax_sino_reproj.text(
                0.5, 0.5,
                "No reprojections available.\nRun tomo.reconstruct() first.",
                ha='center', va='center', transform=ax_sino_reproj.transAxes, fontsize=9,
            )
            ax_sino_reproj.set_title('Reprojected Sinogram (central slice)')
        ax_sino_reproj.set_ylabel('Angle (deg)')
        ax_sino_reproj.set_xlabel('Detector pixel')

        plt.tight_layout()
        plt.show()

    return combined_rmse, x_rmse, y_rmse, x_cm, y_cm
