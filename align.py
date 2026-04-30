if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from datetime import datetime
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift, degree_to_positiveRadians, runwidget

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = True           # Set to True to enable logging output to a file
    saveToFile = True     # Set to True to save aligned projection data to a TIFF file
    reconstruct = True     # Set to True to save the reconstruction to a TIFF file
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
    # DATA IMPORT (EXAMPLE FOR REAL DATA)
    # -------------------------
    # Uncomment the following lines to use experimental projection data from a TIFF file:
    # tif_file = "alignedProjections/aligned_manually_3_3_25.tif"
    # obj, scale_info = convert_to_numpy(tif_file)
    # print(obj.shape)

    #Importing data from Taylor Buckway h5 file (APS data)
    import h5py
    import numpy as np
    # filename = "/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5"
    filename = "/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5"


    def tomo_data(file,redo_align=False):
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

    tomo = tomoDataClass.tomoData(data, angles)
    scale_info = None

    # # -------------------------
    # # ALIGNMENT INSTRUCTIONS
    # # -------------------------

    
    # """
    # Alignment Options (defined in alignment_methods.py):
    # - cross_correlate_align
    # - rotate_correlate_align
    # - vertical_mass_fluctuation_align
    # - tomopy_align            # TomoPy’s implementation of joint reprojection alignment (PMA)
    # - PMA                     # Custom projection matching algorithm
    # - optical_flow_align
    # - center_projections
    # """

    print("Starting alignment")

    saveName = "1stRun"

    savePath = f"/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/{saveName}_aligned_{timestamp}.tif"
    
    print("\n\nCreating aligned projections:", savePath)

    # -------------------------
    # ALIGNMENT STRATEGY
    # -------------------------
    # Choose and configure alignment algorithm below:
    tomo.reset_workingProjections(x_size=data.shape[2]-500, y_size=data.shape[1])
    tomo.normalize(isPhaseData=True)

    #Best XC params from sweep:
    # Coarse passes: stepRatio=0.9 (stable global convergence)
    tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=20, stepRatio=0.9, yROI_Range=None, xROI_Range=None, isFull360=False, downsample=16, use_grad=True)
    tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=20, stepRatio=0.9, yROI_Range=None, xROI_Range=None, isFull360=False, downsample=8, use_grad=True)
    tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.9, yROI_Range=None, xROI_Range=None, isFull360=False, downsample=4, use_grad=True)
    tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=10, stepRatio=0.9, yROI_Range=None, xROI_Range=None, isFull360=False, downsample=2, use_grad=True)
    tomo.cross_correlate_align(tolerance=0, maxShiftTolerance=0, max_iterations=5, stepRatio=0.8, yROI_Range=None, xROI_Range=None, isFull360=False, downsample=1, use_grad=True)

    tomo.vertical_mass_fluctuation_align(tolerance=0, max_iterations=5, y_range=None, sigma=None, smooth_sigma=1.0, window='soft_roi', roi_sigma=0.3, use_gradient=True, plot=False)

    tomo.PMA(max_iterations=5, tolerance=0, algorithm=recon_alg, levels=3, scale=4, iterations_per_level=[5,5,2], shift_method='optical_flow', of_sigma=3.0, plot=False)
    tomo.make_updates_shift()

    runwidget(tomo.get_finalProjections())
    # -------------------------
    # SAVE RESULTS (Optional)
    # -------------------------
    if saveToFile:
        convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)
    if reconstruct:
        tomo.reconstruct(algorithm=recon_alg, snr_db=None)
        recon_path = f"reconstructions/APSbeamtime_Oct25/{saveName}_recon_{timestamp}.tif"
        convert_to_tiff(tomo.get_recon(), recon_path, scale_info)




