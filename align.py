if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    import torch
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


    # Import foam data
    tif_file = "data/fullTomoReconstructions_8_28_24.tif"
    obj, scale_info = convert_to_numpy(tif_file)
    # obj = obj[0:numAngles]
    # obj = obj[::10]
    print(obj.shape)
    tomo = tomoDataClass.tomoData(obj)
    # tomo.crop_center(900,550)


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

    for alg in ['sirt']:
        print("Starting alignment")

        # #Base Case
        tomo.reset_workingProjections()
        tomo.track_shifts()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_baseCase_noFilter_noOptFlow_{timestamp}.tif", scale_info)


        # #Base Case with Filter
        tomo.reset_workingProjections()
        tomo.track_shifts()
        tomo.bilateralFilter()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_baseCase_Filter_noOptFlow_{timestamp}.tif", scale_info)



        # #Rotational Alignment
        tomo.reset_workingProjections()
        tomo.track_shifts()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.make_updates_shift()

        tomo.rotate_correlate_align()
        tomo.make_updates_rotate()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()

        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_rotate_noFilter_noOptFlow_{timestamp}.tif", scale_info)


        # #Rotational Alignment with Filter
        tomo.reset_workingProjections()
        tomo.track_shifts()
        tomo.bilateralFilter()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.make_updates_shift()

        tomo.rotate_correlate_align()
        tomo.make_updates_rotate()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()

        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_rotate_Filter_noOptFlow_{timestamp}.tif", scale_info)



        # #Unrotate (without filter)
        tomo.reset_workingProjections()
        tomo.track_shifts()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.make_updates_shift()

        tomo.rotate_correlate_align()
        tomo.make_updates_rotate()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()

        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()

        tomo.unrotate()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_unRotate_noFilter_noOptFlow_{timestamp}.tif", scale_info)


        # #Unrotate (with filter)
        tomo.reset_workingProjections()
        tomo.track_shifts()
        tomo.bilateralFilter()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()
        tomo.make_updates_shift()

        tomo.rotate_correlate_align()
        tomo.make_updates_rotate()

        tomo.cross_correlate_align(tolerance=0.5, max_iterations=15)
        tomo.center_projections()

        tomo.tomopy_align(iterations = 15, alg = alg)
        tomo.make_updates_shift()
        # tomo.optical_flow_align()

        tomo.unrotate()
        tomo.makeScriptProjMovie()
        convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_unRotate_Filter_noOptFlow_{timestamp}.tif", scale_info)



        # #Save the aligned data
        # if saveToFile:
            # convert_to_tiff(tomo.get_finalProjections(), f"alignedProjections/aligned_iterateVMF_optFlow_{alg}_{timestamp}.tif", scale_info)
            # np.save(f'shift_values/shiftValues_{timestamp}.npy', tomo.tracked_shifts)


    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

