if __name__ == '__main__':
    import time  # Import the time module
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff

    log_file = open('output_tomoMono_align.txt', 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    #Start the timer
    start_time = time.time()

    # #Import model data
    # numAngles = ?
    # shepp3d = tomopy.shepp3d(size=128)
    # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=180)
    # obj = tomopy.project(shepp3d, ang, pad=False)
    # tomo = tomoDataClass.tomoData(obj)
    # tomo.jitter()





    #Import foam data
    numAngles = 20
    tif_file = "cropped_fullTomoReconstructions2.tif"
    obj = convert_to_numpy(tif_file)[:numAngles]
    tomo = tomoDataClass.tomoData(obj)

    tomo.crop(400,400)

    #Alignment Process
    print("Starting allignment")
    # tomo.tomopyAlign(iterations = 3)
    tomo.makeScriptMovie()

    #Save the aligned data
    convert_to_tiff(tomo.get_projections, "aligned_foamTomo.tif")

    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
