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
    import tomopy
    import numpy as np


    # Configuration flags
    log = True  # Enable logging to file
    saveToFile = True  # Enable saving data to file

    # Start the timer for execution duration tracking
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Timestamp for file naming

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run reconstruction algorithms.')
    parser.add_argument('--algorithms', nargs='+', help='List of algorithms to use for reconstruction', required=False,
                        #  default=[['art', 'bart','fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh', 'gpu', 'svmbir']])
                        # default = ['FP_CUDA', 'BP_CUDA', "FBP_CUDA", "SIRT_CUDA", "SART_CUDA", "CGLS_CUDA", "EM_CUDA"])
                        default = ["svmbir"])

    args = parser.parse_args()
    algorithms = args.algorithms

    # Setup logging if enabled
    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono{timestamp}.txt', 'w')








    print("Running TomoMono Script!")
    print(f"Algorithms: {algorithms}")

    # Check for GPU availability
    if torch.cuda.is_available():
        print("GPU is available")

    # # Importing data

    # # The following commented-out code block is for importing and processing example data from Tomopy
    # numAngles = 800
    # shepp3d = tomopy.shepp3d(size=128)
    # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)
    # obj = tomopy.project(shepp3d, ang, pad=False)
    # tomo = tomoDataClass.tomoData(obj, numAngles)
    # tomo.jitter()  # Apply jitter to the data

    # Import foam data
    numAngles = 800
    tif_file = "data/fullTomoReconstructions2.tif"
    obj, scale_info = convert_to_numpy(tif_file)
    obj = obj[0:numAngles]
    print(obj.shape)
    tomo = tomoDataClass.tomoData(obj)
    tomo.crop_center(900,550)




    # # Alignment Process
    print("Starting alignment")
    tomo.track_shifts()
    tomo.cross_correlate_align()
    tomo.vertical_mass_fluctuation_align()
    tomo.tomopy_align(iterations = 10)
    tomo.optical_flow_align()
    tomo.center_projections()
    print(tomo.tracked_shifts)
    tomo.makeScriptProjMovie()

    for m in tqdm(range(tomo.num_angles), desc='Center projections'):
        tomo.originalProjections[m] = subpixel_shift(tomo.originalProjections[m], tomo.tracked_shifts[m,0], tomo.tracked_shifts[m,1])
    tomo.projections = tomo.originalProjections
    tomo.makeScriptProjMovie()

    # tomo.projections = tomo.data
    # tomo.crop_center(900,550)

    # tomo.makeScriptProjMovie()
    # #Save the aligned data
    if saveToFile:
        convert_to_tiff(tomo.get_projections(), f"alignedProjections/aligned_foamTomo{timestamp}.tif", scale_info)
        np.save(f'shiftValues_{timestamp}.npy', tomo.tracked_shifts)





    # # Use pre-aligned data to reconstruct
    # prealigned_tif_file = "alignedProjections/aligned_foamTomo20240718-165515.tif"
    # obj, scale_info = convert_to_numpy(prealigned_tif_file)
    # tomo = tomoDataClass.tomoData(obj)


    # # # Reconstruction Process
    # print("Reconstructing")
    # tomo.crop_bottom_center(402, 512)
    # tomo.makeScriptProjMovie()
    # for alg in algorithms:
    #     try:
    #         tomo.reconstruct(algorithm=alg)
    #     except Exception as e:
    #         print(f"Failed to reconstruct using {alg}: {e}")
    #         continue
    #     if saveToFile:
    #         convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}.tif", scale_info)
    # tomo.normalize()
    # for alg in algorithms:
    #     try:
    #         tomo.reconstruct(algorithm=alg)
    #     except Exception as e:
    #         print(f"Failed to reconstruct using {alg}: {e}")
    #         continue
    #     if saveToFile:
    #         convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_normalized_{timestamp}_{alg}.tif", scale_info)

    # ###DELETE LATER
    # recon, recon_scale_info = convert_to_numpy(tomo.recon)
    # num = tomo.num_angles
    # ang = tomopy.angles(num, 0, 360*num/800) #First angle should theoretically be the same no matter how many angles I project through
    # tomopyProjections = tomopy.project(tomo, ang)
    # shiftedProjections = np.zeros_like(tomopyProjections)
    # for m in tqdm(range(tomo.num_angles), desc='Shift projections from volume'):
    #     shiftedProjections[m] = subpixel_shift(tomopyProjections[m], -tomo.tracked_shifts[m,0], -tomo.tracked_shifts[m,1])
    # newTomo = tomoDataClass.tomoData(shiftedProjections)
    # newTomo.makeScriptProjMovie()
    # tomo.makeScriptProjMovie()


    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

