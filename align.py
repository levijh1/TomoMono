if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from datetime import datetime
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift
    import tomopy

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = True           # Set to True to enable logging output to a file
    saveToFile = True     # Set to True to save aligned projection data to a TIFF file

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
    filename = r"/home/ljh79/TomoMono/data/poly_tomo_128.hdf5"

    indicesToRemove = [0,1,2,3,4,5,53,60,67,69,82,90,178,183,186, 209, 218, 219, 221, 224, 254, 258, 276, 278, 304, 306] + list(range(308,325)) ##Look out for 200, 

    with h5py.File(filename, "r") as f:
        data = np.array(f["/data"])
        angles = list(f["/angles"])

    data[0] = subpixel_shift(data[0], 0, 400)

    #removing images with holder in the way
    print("data shape is: ", data.shape)
    print("angles shape is: ", len(angles))
    # for index in reversed(indicesToRemove):
    #     print(index)
    data = np.delete(data, indicesToRemove, axis = 0)
    angles = np.delete(angles, indicesToRemove, axis = 0)
    print("data shape after removing is: ", data.shape)
    print("angles shape after removing is: ", len(angles))

    tomo = tomoDataClass.tomoData(data, angles)
    scale_info = None

    # # -------------------------
    # # DATA IMPORT (EXAMPLE FOR SIMULATED DATA): Tomopy Simulated Projections (Shepp-Logan Phantom)
    # # -------------------------
    # numAngles = 800
    # shepp3d = tomopy.shepp3d(size=128)  # Generate 3D Shepp-Logan phantom
    # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)  # Define projection angles
    # obj = tomopy.project(shepp3d, ang, pad=False)  # Create projection data
    # tomo = tomoDataClass.tomoData(obj)  # Wrap projections in tomoData class
    # scale_info = None
    # tomo.jitter(maxShift=5)  # Add random misalignment to simulate experimental shifts

    # -------------------------
    # ALIGNMENT INSTRUCTIONS
    # -------------------------
    """
    Alignment Options (defined in alignment_methods.py):
    - cross_correlate_align
    - rotate_correlate_align
    - vertical_mass_fluctuation_align
    - tomopy_align            # TomoPy’s implementation of joint reprojection alignment (PMA)
    - PMA                     # Custom projection matching algorithm
    - optical_flow_align
    - center_projections
    """






    print("Starting alignment")

    savePath = f"alignedProjections/APSbeamtime1/aligned_XC_{timestamp}.tif"
    print("\n\nCreating aligned projections:", savePath)

    # Ensure alignment begins from original, unmodified projections
    tomo.reset_workingProjections(x_size = data.shape[2], y_size=data.shape[1])

    # -------------------------
    # ALIGNMENT STRATEGY
    # -------------------------
    # Choose and configure alignment algorithm below:
    tomo.cross_correlate_align(tolerance=0.01, max_iterations = 10, yROI_Range=[30, -30], xROI_Range=[0, tomo.workingProjections.shape[2]], maxShiftTolerance=5)
    tomo.cross_correlate_align(tolerance=0.01, max_iterations = 10, yROI_Range=[30, -30], xROI_Range=[760, -760], maxShiftTolerance=5)
    # tomo.PMA(max_iterations = 10, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = tomo.workingProjections.shape[1]-40, crop_bottom_center_x = 1000, isPhaseData = True)
    tomo.center_projections()
    tomo.make_updates_shift()

    # Apply the computed shifts to original data to finalize alignment
    tomo.make_updates_shift()

    # -------------------------
    # SAVE RESULTS (Optional)
    # -------------------------
    if saveToFile:
        convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)





















    print("Starting alignment")

    savePath = f"alignedProjections/APSbeamtime1/aligned_XC&PMA_{timestamp}.tif"
    print("\n\nCreating aligned projections:", savePath)

    # Ensure alignment begins from original, unmodified projections
    tomo.reset_workingProjections(x_size = data.shape[2], y_size=data.shape[1])

    # -------------------------
    # ALIGNMENT STRATEGY
    # -------------------------
    # Choose and configure alignment algorithm below:
    tomo.cross_correlate_align(tolerance=0.01, max_iterations = 10, yROI_Range=[30, -30], xROI_Range=[0, tomo.workingProjections.shape[2]], maxShiftTolerance=5)
    tomo.cross_correlate_align(tolerance=0.01, max_iterations = 10, yROI_Range=[30, -30], xROI_Range=[760, -760], maxShiftTolerance=5)
    tomo.PMA(max_iterations = 10, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = tomo.workingProjections.shape[1]-40, crop_bottom_center_x = 1000, isPhaseData = True)
    tomo.center_projections()
    tomo.make_updates_shift()

    # Apply the computed shifts to original data to finalize alignment
    tomo.make_updates_shift()

    # -------------------------
    # SAVE RESULTS (Optional)
    # -------------------------
    if saveToFile:
        convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)

    # -------------------------
    # EXECUTION TIME REPORTING
    # -------------------------
    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")

    # Restore original stdout/stderr if logging was enabled
    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


