if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from datetime import datetime
    from helperFunctions import DualLogger, convert_to_tiff
    import tomopy

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = False           # Set to True to enable logging output to a file
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
    num_remove = 0   #images with the holder at the end to remove
    with h5py.File(filename, "r") as f:
        data = np.array(f["/object"])
        angles = list(f["/angles"])
    print("data shape is: ", data.shape)
    print("angles shape is: ", angles.shape)
    #removing images with holder in the way
    if num_remove > 0:
        data = data[:-num_remove]
        angles = angles[:-num_remove]
    tomo = tomoDataClass.tomoData(data)
    tomo.makeScriptProjMovie()
    sys.exit(0) 

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

    savePath = f"alignedProjections/aligned_manuallyPrepped_PMA_{timestamp}.tif"
    print("\n\nCreating aligned projections:", savePath)

    # Ensure alignment begins from original, unmodified projections
    tomo.reset_workingProjections()

    # -------------------------
    # ALIGNMENT STRATEGY
    # -------------------------
    # Choose and configure alignment algorithm below:
    tomo.cross_correlate_align(tolerance=0.01, max_iterations = 20)
    tomo.PMA(max_iterations = 15, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = 500, crop_bottom_center_x = 750)
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


