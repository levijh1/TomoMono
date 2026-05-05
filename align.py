if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from datetime import datetime
    from scipy.ndimage import zoom
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift, degree_to_positiveRadians, runwidget
    from alignment_methods import reprojection_consistency_score

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
    start_time = time.time()  # Start execution timer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Generate timestamp for filenames
    print("timestamp:", timestamp)

    if log:
        # Redirect stdout/stderr to both console and log file
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')

    print("Running Image Registration Script")
    try:
        import cupy as cp
        if cp.is_available():
            print("CuPy GPU available — array operations will use GPU")
    except ImportError:
        pass

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
        return projs, angles

    data, angles = tomo_data(filename, redo_align=True)
    print("data shape is: ", data.shape)
    print("angles shape is: ", len(angles))

    scale_info = None
    run_results = []  # (downsample_factor, rcs, elapsed_s)

    # -------------------------
    # DOWNSAMPLE SWEEP
    # -------------------------
    for DOWNSAMPLE_SPATIAL in [4, 2, 1]:
        run_start = time.time()
        label = f"ds{DOWNSAMPLE_SPATIAL}x" if DOWNSAMPLE_SPATIAL > 1 else "full_res"
        saveName = f"cfg59_{label}"

        print(f"\n{'=' * 60}")
        print(f"  Run: {saveName}  ({DOWNSAMPLE_SPATIAL}x spatial downsample)")
        print(f"{'=' * 60}")

        if DOWNSAMPLE_SPATIAL > 1:
            data_ds = zoom(data, (1, 1 / DOWNSAMPLE_SPATIAL, 1 / DOWNSAMPLE_SPATIAL), order=1)
        else:
            data_ds = data
        print(f"Working shape: {data_ds.shape}")

        tomo = tomoDataClass.tomoData(data_ds, angles)

        savePath = f"/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/{saveName}_aligned_{timestamp}.tif"
        print("Output:", savePath)

        # -------------------------
        # ALIGNMENT STRATEGY — config 59 (best from hyperparameter search)
        # xca_3p_grad_sr08 + pma_2lev_sr08_sig2
        # -------------------------

        # Full-width first pass
        tomo.reset_workingProjections(x_size=None, y_size=None)
        tomo.normalize(isPhaseData=True)

        tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.8,
                                   yROI_Range=None, xROI_Range=None, isFull360=False, downsample=4, use_grad=True)

        # Commit shifts, then crop (500 px at full-res → scaled by downsample)
        tomo.make_updates_shift()
        x_size = data_ds.shape[2] - (500 // DOWNSAMPLE_SPATIAL)
        tomo.crop_center(new_x=x_size, new_y=None)

        tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.8,
                                   yROI_Range=None, xROI_Range=None, isFull360=False, downsample=2, use_grad=True)
        tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=5,  stepRatio=0.8,
                                   yROI_Range=None, xROI_Range=None, isFull360=False, downsample=1, use_grad=True)

        tomo.PMA(tolerance=0, algorithm=recon_alg, plot=False,
                 levels=2, scale=4, iterations_per_level=[5, 5],
                 shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8)
        tomo.make_updates_shift()

        runwidget(tomo.get_finalProjections())

        # -------------------------
        # SAVE ALIGNED PROJECTIONS
        # -------------------------
        if saveToFile:
            convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)

        # -------------------------
        # RECONSTRUCT + RCS
        # -------------------------
        tomo.reconstruct(algorithm=recon_alg, snr_db=None)

        if saveRecon:
            recon_path = f"reconstructions/APSbeamtime_Oct25/{saveName}_recon_{timestamp}.tif"
            convert_to_tiff(tomo.get_recon(), recon_path, scale_info)

        rcs, _, _ = reprojection_consistency_score(tomo, plot=False)
        run_elapsed = time.time() - run_start

        print(f"\n  [{saveName}]  RCS = {rcs:.4f}  |  time = {run_elapsed / 60:.1f} min ({run_elapsed:.0f} s)")
        run_results.append((DOWNSAMPLE_SPATIAL, label, rcs, run_elapsed))

    # -------------------------
    # SUMMARY
    # -------------------------
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for ds, label, rcs, elapsed in run_results:
        print(f"  cfg59_{label}:  RCS = {rcs:.4f}  |  {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    print(f"{'─' * 60}")
    print(f"  Total runtime: {total_elapsed / 60:.1f} min ({total_elapsed:.0f} s)")
    print(f"{'=' * 60}")
