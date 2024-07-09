if __name__ == '__main__':
    import time  # Import the time module
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    import torch
    import tomopy
    import matplotlib.pyplot as plt

    log = False
    saveToFile = True

    if log:
        log_file = open('output_tomoMono.txt', 'w')
        sys.stdout = log_file
        sys.stderr = log_file

    if torch.cuda.is_available():
        print("GPU is available")

    #Start the timer
    start_time = time.time()

    # #Import model data
    # numAngles = 400
    # shepp3d = tomopy.shepp3d(size=256)
    # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)
    # obj = tomopy.project(shepp3d, ang, pad=False)
    # tomo = tomoDataClass.tomoData(obj, numAngles)
    # tomo.jitter()

    #Import foam data
    numAngles = 800
    tif_file = "data/cropped_fullTomoReconstructions2.tif"
    obj = convert_to_numpy(tif_file)[::20]
    print(obj.shape)
    tomo = tomoDataClass.tomoData(obj)

    #Alignment Process
    print("Starting allignment")
    tomo.crossCorrelateAlign()
    # tomo.makeScriptProjMovie()
    # tomo.tomopyAlign(iterations = 3)
    # tomo.opticalFlowAlign()

    # tif_file = "data/aligned_foamTomo.tif"
    # obj = convert_to_numpy(tif_file)
    # tomo = tomoDataClass.tomoData(obj)


    #Show sinogram
    plt.imshow(tomo.get_projections()[:,tomo.imageSize[1]//2,:])
    plt.show()


    # # #Reconstruction Process
    # print("Reconstructing")
    # tomo.normalize()
    # tomo.recon()
    # tomo.makeScriptReconMovie()

    # # #Save the aligned data
    if saveToFile:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        proj = tomo.get_projections()
        recon = tomo.get_recon()
        convert_to_tiff(proj, f"alignedProjections/aligned_foamTomo{timestamp}.tif")
        convert_to_tiff(recon, f"reconstructions/foamRecon{timestamp}.tif")

    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

