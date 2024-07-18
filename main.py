if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    import torch
    import tomopy
    import matplotlib.pyplot as plt
    import os

    # Configuration flags
    log = False  # Enable logging to file
    saveToFile = True  # Enable saving data to file

    # Start the timer for execution duration tracking
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Timestamp for file naming

    # Setup logging if enabled
    if log:
        log_file = open(f'logs/output_tomoMono{timestamp}.txt', 'w')
        sys.stdout = log_file  # Redirect standard output to log file
        sys.stderr = log_file  # Redirect standard error to log file

    # Check for GPU availability
    if torch.cuda.is_available():
        print("GPU is available")




    # # The following commented-out code block is for importing and processing model data
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
    tomo.crop(900,550)
    # tomo.makeScriptProjMovie()





    # Alignment Process
    print("Starting alignment")
    tomo.cross_correlate_align()
    tomo.rotate_correlate_align()
    tomo.tomopy_align(iterations = 10)
    tomo.optical_flow_align()
    # tomo.makeScriptProjMovie()

    # #Save the aligned data
    if saveToFile:
        convert_to_tiff(tomo.get_projections(), f"alignedProjections/aligned_foamTomo{timestamp}.tif", scale_info)
        # convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon{timestamp}.tif", scale_info)





    # # Use pre-aligned data to reconstruct
    # tif_file = "alignedProjections/aligned_foamTomo20240709-152238.tif"
    # obj, scale_info = convert_to_numpy(tif_file)
    # tomo = tomoDataClass.tomoData(obj)


    #Reconstruction Process
    print("Reconstructing")
    # tomo.normalize()
    algorithms = ['art', 'bart','fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh', 'gpu', 'svmbir']
    for alg in algorithms:
        try:
            tomo.reconstruct(algorithm=alg)
        except Exception as e:
            print(f"Failed to reconstruct using {alg}: {e}")
            continue
        if saveToFile:
            convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_notNormalized_{timestamp}_{alg}.tif", scale_info)



    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

