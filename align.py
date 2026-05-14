if __name__ == '__main__':
    import time
    import sys
    import os
    import tomoDataClass
    from datetime import datetime
    from scipy.ndimage import zoom
    from helperFunctions import DualLogger, convert_to_tiff
    from alignment_methods import reprojection_consistency_score

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = True
    saveToFile = True
    saveRecon = True
    recon_alg = "SIRT_CUDA"

    # -------------------------
    # SETUP: Timing & Logging
    # -------------------------
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("timestamp:", timestamp)

    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')

    print("Running Staged Alignment Script (4x → 2x → 1x)")
    try:
        import cupy as cp
        if cp.is_available():
            print("CuPy GPU available — array operations will use GPU")
    except ImportError:
        cp = None

    # -------------------------
    # DATA IMPORT
    # -------------------------
    import h5py
    import numpy as np
    filename = "/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5"

    with h5py.File(filename) as hf:
        data = hf['data'][...]
        angles = hf['angles'][...]
    angles = angles * np.pi / 180
    angles = angles - np.mean(angles)

    drop_idx = [19, 26]
    data = np.delete(data, drop_idx, axis=0)
    angles = np.delete(angles, drop_idx, axis=0)
    print(f"Loaded data shape: {data.shape}, angles: {len(angles)}")

    num_angles = data.shape[0]

    # -------------------------
    # OUTPUT DIRECTORIES
    # -------------------------
    aligned_dir = "/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25"
    sinogram_dir = os.path.join(aligned_dir, "sinograms")
    recon_dir = "reconstructions/APSbeamtime_Oct25"
    shifts_dir = os.path.join(aligned_dir, "shifts")
    os.makedirs(sinogram_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(shifts_dir, exist_ok=True)

    scale_info = None
    run_results = []

    def save_outputs(tomo, label):
        final = tomo.get_finalProjections()
        if cp is not None and isinstance(final, cp.ndarray):
            final_np = cp.asnumpy(final)
        elif hasattr(final, 'get') and not isinstance(final, np.ndarray):
            final_np = final.get()
        else:
            final_np = np.asarray(final)

        if saveToFile:
            proj_path = f"{aligned_dir}/{label}_aligned_{timestamp}.tif"
            convert_to_tiff(final, proj_path, scale_info)
            print(f"Saved projections: {proj_path}")

        mid_row = final_np.shape[1] // 2
        sino = final_np[:, mid_row, :]
        sino_path = f"{sinogram_dir}/{label}_sinogram_{timestamp}.png"
        plt.imsave(sino_path, sino, cmap='gray')
        print(f"Saved sinogram: {sino_path}")

        if saveRecon:
            recon_start = time.time()
            tomo.reconstruct(algorithm=recon_alg, snr_db=None)
            recon_elapsed = time.time() - recon_start
            print(f"Reconstruction done in {recon_elapsed / 60:.1f} min")
            recon_path = f"{recon_dir}/{label}_recon_{timestamp}.tif"
            convert_to_tiff(tomo.get_recon(), recon_path, scale_info)
            print(f"Saved reconstruction: {recon_path}")
        else:
            recon_elapsed = 0.0

        rcs, _, _ = reprojection_consistency_score(tomo, plot=False)
        _, fsc_resolutions, _ = tomo.fourier_shell_correlation(algorithm=recon_alg, plot=False)
        fsc_res = fsc_resolutions.get('FSC=0.143')
        fsc_str = f"{fsc_res:.4f}" if fsc_res is not None else "N/A"
        print(f"[{label}]  RCS = {rcs:.4f}  |  FSC = {fsc_str} px")
        return rcs, fsc_res, recon_elapsed

    # =========================================================
    # STAGE 1: 4x DOWNSAMPLED — Full XCA + PMA alignment
    # =========================================================
    print(f"\n{'=' * 60}")
    print("  STAGE 1: 4x downsampled")
    print(f"{'=' * 60}")
    DS = 4
    data_4x = zoom(data, (1, 1 / DS, 1 / DS), order=1)
    print(f"Shape: {data_4x.shape}")

    tomo4x = tomoDataClass.tomoData(data_4x, angles)
    tomo4x.normalize(isPhaseData=True)

    align_start = time.time()

    tomo4x.cross_correlate_align(
        tolerance=0, maxShiftTolerance=0, max_iterations=5, stepRatio=0.8,
        downsample=4, use_grad=True,
        yROI_Range=None, xROI_Range=None, isFull360=False,
    )
    tomo4x.center_projections()

    tomo4x.cross_correlate_align(
        tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.8,
        downsample=2, use_grad=True,
        yROI_Range=[0, tomo4x.workingProjections.shape[1] - 50],
        xROI_Range=[62, tomo4x.workingProjections.shape[2] - 62],
        isFull360=False,
    )

    tomo4x.cross_correlate_align(
        tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.8,
        downsample=1, use_grad=True,
        yROI_Range=[0, tomo4x.workingProjections.shape[1] - 75],
        xROI_Range=[62, tomo4x.workingProjections.shape[2] - 62],
        isFull360=False,
    )

    maxWidth_4x = num_angles // DS
    xROI_4x = [
        tomo4x.workingProjections.shape[2] // 2 - maxWidth_4x,
        tomo4x.workingProjections.shape[2] // 2 + maxWidth_4x,
    ]
    yROI_4x = [0, tomo4x.workingProjections.shape[1] - 75]
    print(f"PMA: xROI={xROI_4x}, yROI={yROI_4x}")
    tomo4x.PMA(
        levels=1, scale=1, iterations_per_level=[10],
        tolerance=0.0, algorithm=recon_alg, standardize=False, use_grad=True,
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8, plot=False,
        xROI_Range=xROI_4x, yROI_Range=yROI_4x,
    )

    # Save cumulative shifts before committing (used to seed 2x stage)
    shifts_4x = tomo4x.tracked_shifts.copy()
    np.save(f"{shifts_dir}/shifts_4x_{timestamp}.npy", shifts_4x)
    print(f"Saved 4x shifts: max |y|={np.abs(shifts_4x[:,0]).max():.2f} px, max |x|={np.abs(shifts_4x[:,1]).max():.2f} px")

    tomo4x.make_updates_shift()
    align_elapsed_4x = time.time() - align_start
    print(f"4x alignment done in {align_elapsed_4x / 60:.1f} min")

    rcs_4x, fsc_4x, recon_elapsed_4x = save_outputs(tomo4x, "cfg_4xds")
    run_results.append(("4x", "cfg_4xds", rcs_4x, fsc_4x, align_elapsed_4x, recon_elapsed_4x))

    del data_4x, tomo4x

    # =========================================================
    # STAGE 2: 2x DOWNSAMPLED — Apply scaled 4x shifts + PMA
    # =========================================================
    print(f"\n{'=' * 60}")
    print("  STAGE 2: 2x downsampled")
    print(f"{'=' * 60}")
    DS = 2
    data_2x = zoom(data, (1, 1 / DS, 1 / DS), order=1)
    print(f"Shape: {data_2x.shape}")

    tomo2x = tomoDataClass.tomoData(data_2x, angles)
    tomo2x.reset_workingProjections()
    tomo2x.normalize(isPhaseData=True)

    align_start = time.time()

    # Seed from 4x shifts scaled up by 2
    tomo2x.tracked_shifts = (shifts_4x * 2).copy()
    tomo2x.make_updates_shift()
    tomo2x.workingProjections = np.copy(tomo2x.finalProjections)

    maxWidth_2x = num_angles // DS
    xROI_2x = [
        tomo2x.workingProjections.shape[2] // 2 - maxWidth_2x,
        tomo2x.workingProjections.shape[2] // 2 + maxWidth_2x,
    ]
    yROI_2x = [0, tomo2x.workingProjections.shape[1] - 150]
    print(f"PMA: xROI={xROI_2x}, yROI={yROI_2x}")
    tomo2x.PMA(
        levels=1, scale=1, iterations_per_level=[5],
        tolerance=0.0, algorithm=recon_alg, standardize=False,
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8, plot=False,
        xROI_Range=xROI_2x, yROI_Range=yROI_2x,
    )

    # Save PMA correction before committing (needed to seed 1x stage)
    pma_correction_2x = tomo2x.tracked_shifts.copy()
    shifts_2x_total = shifts_4x * 2 + pma_correction_2x
    np.save(f"{shifts_dir}/shifts_2x_total_{timestamp}.npy", shifts_2x_total)
    print(f"Saved 2x total shifts: max |y|={np.abs(shifts_2x_total[:,0]).max():.2f} px, max |x|={np.abs(shifts_2x_total[:,1]).max():.2f} px")

    tomo2x.make_updates_shift()
    align_elapsed_2x = time.time() - align_start
    print(f"2x alignment done in {align_elapsed_2x / 60:.1f} min")

    rcs_2x, fsc_2x, recon_elapsed_2x = save_outputs(tomo2x, "cfg_2xds")
    run_results.append(("2x", "cfg_2xds", rcs_2x, fsc_2x, align_elapsed_2x, recon_elapsed_2x))

    del data_2x, tomo2x

    # =========================================================
    # STAGE 3: FULL RESOLUTION — Apply scaled 2x shifts + PMA
    # =========================================================
    print(f"\n{'=' * 60}")
    print("  STAGE 3: Full resolution")
    print(f"{'=' * 60}")
    print(f"Shape: {data.shape}")

    tomo1x = tomoDataClass.tomoData(data, angles)
    tomo1x.reset_workingProjections()
    tomo1x.normalize(isPhaseData=True)

    align_start = time.time()

    # Seed from 2x total shifts scaled up by 2
    tomo1x.tracked_shifts = (shifts_2x_total * 2).copy()
    tomo1x.make_updates_shift()
    tomo1x.workingProjections = np.copy(tomo1x.finalProjections)

    W_1x = tomo1x.workingProjections.shape[2]
    maxWidth_1x = num_angles  # num_angles // 1, consistent with 4x and 2x stages
    xROI_1x = [
        max(0, W_1x // 2 - maxWidth_1x),
        min(W_1x, W_1x // 2 + maxWidth_1x),
    ]
    yROI_1x = [0, tomo1x.workingProjections.shape[1] - 300]
    print(f"PMA: xROI={xROI_1x}, yROI={yROI_1x}")
    tomo1x.PMA(
        levels=1, scale=1, iterations_per_level=[5],
        tolerance=0.0, algorithm=recon_alg, standardize=False,
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8, plot=False,
        xROI_Range=xROI_1x, yROI_Range=yROI_1x,
    )

    tomo1x.make_updates_shift()
    align_elapsed_1x = time.time() - align_start
    print(f"1x alignment done in {align_elapsed_1x / 60:.1f} min")

    rcs_1x, fsc_1x, recon_elapsed_1x = save_outputs(tomo1x, "cfg_fullres")
    run_results.append(("1x", "cfg_fullres", rcs_1x, fsc_1x, align_elapsed_1x, recon_elapsed_1x))

    # -------------------------
    # SUMMARY
    # -------------------------
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for ds, label, rcs, fsc_res, align_e, recon_e in run_results:
        fsc_str = f"{fsc_res:.4f}" if fsc_res is not None else "N/A"
        print(f"  {label}:  RCS = {rcs:.4f}  |  FSC = {fsc_str} px  |  align {align_e / 60:.1f} min  |  recon {recon_e / 60:.1f} min")
    print(f"{'─' * 60}")
    print(f"  Total runtime: {total_elapsed / 60:.1f} min ({total_elapsed:.0f} s)")
    print(f"{'=' * 60}")
