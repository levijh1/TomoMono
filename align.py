if __name__ == '__main__':
    import time
    import sys
    import os
    import tomoDataClass
    from datetime import datetime
    from scipy.ndimage import zoom
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift, degree_to_positiveRadians, runwidget
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
    run_results = []  # (downsample_factor, label, rcs, elapsed_s)

    # -------------------------
    # OUTPUT DIRECTORIES
    # -------------------------
    aligned_dir = "/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25"
    sinogram_dir = os.path.join(aligned_dir, "sinograms")
    recon_dir = "reconstructions/APSbeamtime_Oct25"
    os.makedirs(sinogram_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # -------------------------
    # PER-CONFIG ALIGNMENT RECIPES
    # -------------------------
    # Each entry is (DOWNSAMPLE_SPATIAL, label, recipe_fn).
    # recipe_fn(tomo, DS) runs the XCA passes (with a make_updates_shift + crop_center
    # inserted after the first XCA pass) and the PMA stage. Final make_updates_shift
    # is called by the main loop.

    def recipe_4xds(tomo, DS):
        # XCA Pass 1 - coarse (downsample=4)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=4, use_grad=True,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        tomo.make_updates_shift()
        x_size = tomo.workingProjections.shape[2] - (500 // DS)
        tomo.crop_center(new_x=x_size, new_y=None)

        # XCA Pass 2 - medium
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=2, use_grad=True,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 3 - full resolution
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=1, use_grad=True,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # PMA - 3 levels, scale=2, of_sigma=2.0, stepRatio=0.8
        tomo.PMA(
            levels=3, scale=2, iterations_per_level=[5, 5, 5],
            tolerance=0.01, algorithm=recon_alg, standardize=False,
            shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8, plot=False,
        )

    def recipe_2xds(tomo, DS):
        # XCA Pass 1 - coarse (downsample=8)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=8, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        tomo.make_updates_shift()
        x_size = tomo.workingProjections.shape[2] - (500 // DS)
        tomo.crop_center(new_x=x_size, new_y=None)

        # XCA Pass 2 (downsample=4)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=4, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 3 (downsample=2)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=2, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 4 - full resolution
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.8,
            downsample=1, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # PMA - 3 levels, scale=4, of_sigma=4.0, stepRatio=0.9
        tomo.PMA(
            levels=3, scale=4, iterations_per_level=[8, 5, 3],
            tolerance=0.01, algorithm=recon_alg, standardize=False,
            shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9, plot=False,
        )

    def recipe_fullres(tomo, DS):
        # XCA Pass 1 - very coarse (downsample=16)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.85,
            downsample=16, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        tomo.make_updates_shift()
        x_size = tomo.workingProjections.shape[2] - (500 // DS)
        tomo.crop_center(new_x=x_size, new_y=None)

        # XCA Pass 2 (downsample=8)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.85,
            downsample=8, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 3 (downsample=4)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.85,
            downsample=4, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 4 (downsample=2)
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.85,
            downsample=2, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # XCA Pass 5 - full resolution
        tomo.cross_correlate_align(
            tolerance=0.001, maxShiftTolerance=0.5, max_iterations=15, stepRatio=0.85,
            downsample=1, use_grad=False,
            yROI_Range=None, xROI_Range=None, isFull360=False,
        )
        # PMA - 3 levels, scale=4, of_sigma=8.0, stepRatio=0.9
        tomo.PMA(
            levels=3, scale=4, iterations_per_level=[10, 6, 4],
            tolerance=0.01, algorithm=recon_alg, standardize=False,
            shift_method='optical_flow', of_sigma=8.0, stepRatio=0.9, plot=False,
        )

    configs = [
        (4, "cfg_4xds",    recipe_4xds),
        (2, "cfg_2xds",    recipe_2xds),
        (1, "cfg_fullres", recipe_fullres),
    ]

    # -------------------------
    # RUN EACH CONFIGURATION
    # -------------------------
    for DOWNSAMPLE_SPATIAL, saveName, recipe in configs:
        run_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"  Run: {saveName}  ({DOWNSAMPLE_SPATIAL}x spatial downsample)")
        print(f"{'=' * 60}")

        if DOWNSAMPLE_SPATIAL > 1:
            data_ds = zoom(data, (1, 1 / DOWNSAMPLE_SPATIAL, 1 / DOWNSAMPLE_SPATIAL), order=1)
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

        recipe(tomo, DOWNSAMPLE_SPATIAL)

        tomo.make_updates_shift()

        runwidget(tomo.get_finalProjections())

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
        # RECONSTRUCT + RCS
        # -------------------------
        tomo.reconstruct(algorithm=recon_alg, snr_db=None)

        if saveRecon:
            recon_path = f"{recon_dir}/{saveName}_recon_{timestamp}.tif"
            convert_to_tiff(tomo.get_recon(), recon_path, scale_info)

        rcs, _, _ = reprojection_consistency_score(tomo, plot=False)
        run_elapsed = time.time() - run_start

        print(f"\n  [{saveName}]  RCS = {rcs:.4f}  |  time = {run_elapsed / 60:.1f} min ({run_elapsed:.0f} s)")
        run_results.append((DOWNSAMPLE_SPATIAL, saveName, rcs, run_elapsed))

    # -------------------------
    # SUMMARY
    # -------------------------
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for ds, label, rcs, elapsed in run_results:
        print(f"  {label}:  RCS = {rcs:.4f}  |  {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    print(f"{'─' * 60}")
    print(f"  Total runtime: {total_elapsed / 60:.1f} min ({total_elapsed:.0f} s)")
    print(f"{'=' * 60}")
