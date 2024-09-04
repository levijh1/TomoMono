if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    # import torch
    import argparse
    from helperFunctions import DualLogger, subpixel_shift
    from tqdm import tqdm
    from scipy.ndimage import shift
    import numpy as np


    # Configuration flags
    log = True  # Enable logging to file
    saveToFile = True  # Enable saving data to file

    # Start the timer for execution duration tracking
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Timestamp for file naming
    print("timestamp: ", timestamp)


    # Setup logging if enabled
    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')


    print("Running Image Registration Script")

    # # Check for GPU availability
    # if torch.cuda.is_available():
    #     print("GPU is available")


    # Import foam data
    numAngles = 800
    tif_file = "data/fullTomoReconstructions_8_28_24.tif"
    obj, scale_info = convert_to_numpy(tif_file)
    obj = obj[0:numAngles]
    print(obj.shape)
    tomo = tomoDataClass.tomoData(obj)
    tomo.crop_center(900,550)



    #Start tracking shifts made by each alignment algorithm to apply shifts all at once at the end
    tomo.track_shifts() 

    #Actually align data
    """ Choose whatever alignment algorithms you want to use. Options include:
    - cross_correlate_align
    - rotate_correlate_align
    - vertical_mass_fluctuation_align
    - tomopy_align (joint reprojection algorithm)
    - optical_flow_align
    - center_projections"""
    
    ##TODO: Try with out optical flow
    ##TODO: Try with vertical mass fluct instead of rotate_correlation

    for alg in ['sirt', 'tv']:
        print("Starting alignment")
        tomo.cross_correlate_align()
        tomo.vertical_mass_fluctuation_align(5)
        # tomo.rotate_correlate_align()
        tomo.center_projections()
        tomo.tomopy_align(iterations = 5, alg = alg)
        tomo.optical_flow_align()
        tomo.center_projections()
        # print(tomo.tracked_shifts)

        # #Apply changes on unchanged projections
        # for m in tqdm(range(tomo.num_angles), desc='Center projections'):
        #     tomo.originalProjections[m] = subpixel_shift(tomo.originalProjections[m], tomo.tracked_shifts[m,0], tomo.tracked_shifts[m,1])
        # tomo.projections = tomo.originalProjections

        # #Save the aligned data
        if saveToFile:
            convert_to_tiff(tomo.get_projections(), f"alignedProjections/aligned_iterateVMF_{alg}_{timestamp}.tif", scale_info)
            # np.save(f'shift_values/shiftValues_{timestamp}.npy', tomo.tracked_shifts)


    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

