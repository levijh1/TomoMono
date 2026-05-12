if __name__ == '__main__':
    import time
    import math
    import sys
    import os
    import tomoDataClass
    from datetime import datetime
    from scipy.ndimage import zoom
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift, degree_to_positiveRadians
    from alignment_methods import reprojection_consistency_score

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = True           # Set to True to enable logging output to a file
    saveToFile = True     # Set to True to save aligned projection data to a TIFF file
    saveRecon = True    # Set to True to save reconstructed volume to a TIFF file
    recon_alg = "SIRT_CUDA"

    # -------------------------
    # SETUP: Timing & Logging
    # -------------------------
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("timestamp:", timestamp)

    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')

    print("Running Image Registration Script")
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

    def tomo_data(file, redo_align=False):
        try:
            with h5py.File(file) as hf:
                projs = hf['data'][...]
                angles = hf['angles'][...]
        except KeyError:
            with h5py.File(file) as hf:
                projs = hf['data'][...]
                angles = hf['angles'][...]
        angles = angles * np.pi / 180
        # Shift angles to be centered around 0
        angles = angles - np.mean(angles)
        return projs, angles

    data, angles = tomo_data(filename, redo_align=True)
    print("data shape is: ", data.shape)
    print("angles shape is: ", len(angles))

    # Drop corrupted projections/angles at indices 19 and 26
    drop_idx = [19, 26]
    data = np.delete(data, drop_idx, axis=0)
    angles = np.delete(angles, drop_idx, axis=0)
    print(f"Dropped indices {drop_idx}. New data shape: {data.shape}, angles: {len(angles)}")

    scale_info = None
    run_results = []  # (downsample_factor, label, rcs, fsc_res, align_elapsed, recon_elapsed)

    # -------------------------
    # OUTPUT DIRECTORIES
    # -------------------------
    aligned_dir = "/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25"
    sinogram_dir = os.path.join(aligned_dir, "sinograms")
    recon_dir = "reconstructions/APSbeamtime_Oct25"
    os.makedirs(sinogram_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # -------------------------
    # RUN EACH CONFIGURATION
    # -------------------------
    configs = [
        (4, "cfg_4xds"),
        (2, "cfg_2xds"),
        (1, "cfg_fullres"),
    ]

    for DS, saveName in configs:
        print(f"\n{'=' * 60}")
        print(f"  Run: {saveName}  ({DS}x spatial downsample)")
        print(f"{'=' * 60}")

        if DS > 1:
            data_ds = zoom(data, (1, 1 / DS, 1 / DS), order=1)
        else:
            data_ds = data
        print(f"Working shape: {data_ds.shape}")

        tomo = tomoDataClass.tomoData(data_ds, angles)

        savePath = f"{aligned_dir}/{saveName}_aligned_{timestamp}.tif"
        print("Output:", savePath)

        # -------------------------
        # ALIGNMENT
        # -------------------------
        tomo.reset_workingProjections(x_size=None, y_size=None)
        tomo.normalize(isPhaseData=True)

        align_start = time.time()

        # XCA Pass 1 — coarse; no ROI
        # downsample scaled so effective resolution matches DS=4 notebook baseline:
        #   DS=4 → 4, DS=2 → 8, DS=1 → 16
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=5, stepRatio=0.8,
            downsample=4 * (4 // DS), use_grad=True,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        tomo.center_projections()

        # XCA Pass 2 — medium; DS-scaled ROI
        #   DS=4 → 2, DS=2 → 4, DS=1 → 8
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=10, stepRatio=0.8,
            downsample=2 * (4 // DS), use_grad=True,
            yROI_Range=[0, tomo.workingProjections.shape[1] - (200 // DS)],
            xROI_Range=[250 // DS, tomo.workingProjections.shape[2] - (250 // DS)],
            isFull360=False,
        )

        # XCA Pass 3 — fine; DS-scaled ROI
        #   DS=4 → 1, DS=2 → 2, DS=1 → 4
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=10, stepRatio=0.8,
            downsample=1 * (4 // DS), use_grad=True,
            yROI_Range=[0, tomo.workingProjections.shape[1] - (300 // DS)],
            xROI_Range=[250 // DS, tomo.workingProjections.shape[2] - (250 // DS)],
            isFull360=False,
        )

        # PMA — levels=3 for DS=4, +1 per halving of DS (4→5 for DS=1)
        pma_levels = 3 + int(round(math.log2(4 / DS)))
        maxWidth = tomo.num_angles // DS
        xROI_pma = [
            tomo.workingProjections.shape[2] // 2 - maxWidth,
            tomo.workingProjections.shape[2] // 2 + maxWidth,
        ]
        yROI_pma = [0, tomo.workingProjections.shape[1] - (300 // DS)]
        print(f"PMA: levels={pma_levels}, xROI={xROI_pma}, yROI={yROI_pma}")
        tomo.PMA(
            levels=pma_levels, scale=2,
            iterations_per_level=[5] * pma_levels,
            tolerance=0.01, algorithm=recon_alg, standardize=False,
            shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8, plot=False,
            xROI_Range=xROI_pma, yROI_Range=yROI_pma,
        )

        tomo.make_updates_shift()
        align_elapsed = time.time() - align_start
        print(f"Alignment done in {align_elapsed / 60:.1f} min ({align_elapsed:.0f} s)")

        # -------------------------
        # SAVE ALIGNED PROJECTIONS
        # -------------------------
        if saveToFile:
            convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)

        # -------------------------
        # SAVE MIDDLE-SLICE SINOGRAM PNG
        # -------------------------
        final = tomo.get_finalProjections()
        mid_row = final.shape[1] // 2
        sino = final[:, mid_row, :]
        if cp is not None and isinstance(sino, cp.ndarray):
            sino = cp.asnumpy(sino)
        elif hasattr(sino, 'get') and not isinstance(sino, np.ndarray):
            sino = sino.get()
        sinogram_path = f"{sinogram_dir}/{saveName}_sinogram_{timestamp}.png"
        plt.imsave(sinogram_path, sino, cmap='gray')
        print(f"Saved middle-slice sinogram: {sinogram_path}")

        # -------------------------
        # RECONSTRUCT
        # -------------------------
        recon_start = time.time()
        tomo.reconstruct(algorithm=recon_alg, snr_db=None)
        recon_elapsed = time.time() - recon_start
        print(f"Reconstruction done in {recon_elapsed / 60:.1f} min ({recon_elapsed:.0f} s)")

        if saveRecon:
            recon_path = f"{recon_dir}/{saveName}_recon_{timestamp}.tif"
            convert_to_tiff(tomo.get_recon(), recon_path, scale_info)

        # -------------------------
        # QUALITY METRICS
        # -------------------------
        rcs, _, _ = reprojection_consistency_score(tomo, plot=False)
        fsc_res = tomo.fourier_shell_correlation(algorithm=recon_alg, plot=False)

        print(f"\n  [{saveName}]  RCS = {rcs:.4f}  |  FSC resolution = {fsc_res:.4f}")
        print(f"  Alignment: {align_elapsed / 60:.1f} min  |  Reconstruction: {recon_elapsed / 60:.1f} min")
        run_results.append((DS, saveName, rcs, fsc_res, align_elapsed, recon_elapsed))

    # -------------------------
    # SUMMARY
    # -------------------------
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for ds, label, rcs, fsc_res, align_elapsed, recon_elapsed in run_results:
        print(f"  {label}:  RCS = {rcs:.4f}  |  FSC = {fsc_res:.4f}  |  align {align_elapsed / 60:.1f} min  |  recon {recon_elapsed / 60:.1f} min")
    print(f"{'─' * 60}")
    print(f"  Total runtime: {total_elapsed / 60:.1f} min ({total_elapsed:.0f} s)")
    print(f"{'=' * 60}")
