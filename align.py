if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    import argparse
    from helperFunctions import DualLogger, subpixel_shift
    from tqdm import tqdm
    from scipy.ndimage import shift
    import numpy as np
    import tomopy
    import matplotlib.pyplot as plt


    # Configuration flags
    log = False  # Enable logging to file
    saveToFile = True  # Enable saving data to file

    # Start the timer for execution duration tracking
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Timestamp for file naming
    print("timestamp: ", timestamp)


    # Setup logging if enabled
    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')


    print("Running Image Registration Script")


    # Import foam data
    # tif_file = "data/fullTomoReconstructions_3_3_25.tif"
    tif_file = "alignedProjections/aligned_manually_3_3_25.tif"
    obj, scale_info = convert_to_numpy(tif_file)

    print(obj.shape)
    tomo = tomoDataClass.tomoData(obj)

    # #Import model data
    # numAngles = 800
    # shepp3d = tomopy.shepp3d(size=128)
    # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)
    # obj = tomopy.project(shepp3d, ang, pad=False)
    # tomo = tomoDataClass.tomoData(obj)
    # tomo.jitter()

    #Actually align data
    """ Choose whatever alignment algorithms you want to use. Options include:
    - cross_correlate_align
    - rotate_correlate_align
    - vertical_mass_fluctuation_align
    - tomopy_align (joint reprojection algorithm)
    - PMA (my own projection matching alignment algorithm)
    - optical_flow_align
    - center_projections"""

    print("Starting alignment")

    name = f"alignedProjections/aligned_manuallyPrepped_PMA_{timestamp}.tif"
    print("Creating aligned Projections: ", name)
    tomo.reset_workingProjections()
    tomo.PMA(max_iterations = 15, tolerance=0.01, algorithm="SIRT_CUDA", crop_bottom_center_y = 500, crop_bottom_center_x = 750)
    tomo.center_projections()
    tomo.make_updates_shift()
    convert_to_tiff(tomo.get_finalProjections(), name, scale_info)
    
    name = f"alignedProjections/aligned_manuallyPrepped_XCA&PMA_{timestamp}.tif"
    print("Creating aligned Projections: ", name)
    tomo.reset_workingProjections()
    tomo.cross_correlate_align(tolerance=0.3, max_iterations = 20)
    tomo.PMA(max_iterations = 15, tolerance=0.01, algorithm="SIRT_CUDA", crop_bottom_center_y = 500, crop_bottom_center_x = 750)
    tomo.center_projections()
    tomo.make_updates_shift()
    convert_to_tiff(tomo.get_finalProjections(), name, scale_info)


    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

