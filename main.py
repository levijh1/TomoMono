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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run reconstruction algorithms.')
    parser.add_argument('--algorithms', nargs='+', help='List of algorithms to use for reconstruction', required=False,
                        #  default=[['art', 'bart','fbp', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh', 'gpu', 'svmbir']])
                        # default = ['FP_CUDA', 'BP_CUDA', "FBP_CUDA", "SIRT_CUDA", "SART_CUDA", "CGLS_CUDA", "EM_CUDA"])
                        default = ["FBP_CUDA", "SIRT_CUDA"])
                        # default = ['art'])


    args = parser.parse_args()
    algorithms = args.algorithms

    # Setup logging if enabled
    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono{timestamp}.txt', 'w')








    print("Running TomoMono 3D Reconstruction Script!")
    print(f"Algorithms: {algorithms}")

    # Check for GPU availability
    if torch.cuda.is_available():
        print("GPU is available")







    # # Reconstruction Process
    # Use pre-aligned data to reconstruct
    prealigned_tif_file = "alignedProjections/aligned_foamTomo20240730-164635.tif" #Without rotational alignment
    # prealigned_tif_file = "alignedProjections/aligned_foamTomo20240718-165515.tif"

    obj, scale_info = convert_to_numpy(prealigned_tif_file)
    tomo = tomoDataClass.tomoData(obj)
    tomo.center_projections()

    print("Reconstructing")
    for alg in algorithms:
        snr = None
        try:
            tomo.reconstruct(algorithm=alg, snr_db = snr)
        except Exception as e:
            print(f"Failed to reconstruct using {alg}: {e}")
            continue
        if saveToFile:
            if snr == None:
                convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}.tif", scale_info)
            else:
                convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}_snr{snr}.tif", scale_info)
        
            print("Reconstructing")

    tomo.crop_bottom_center(400, 550)
    for alg in algorithms:
        snr = None
        try:
            tomo.reconstruct(algorithm=alg, snr_db = snr)
        except Exception as e:
            print(f"Failed to reconstruct using {alg}: {e}")
            continue
        if saveToFile:
            if snr == None:
                convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}.tif", scale_info)
            else:
                convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}_snr{snr}.tif", scale_info)

    # #Optional SNR Test
    # for alg in algorithms:
    #     snr = None
    #     try:
    #         tomo.reconstruct(algorithm=alg, snr_db = snr)
    #     except Exception as e:
    #         print(f"Failed to reconstruct using {alg}: {e}")
    #         continue
    #     if saveToFile:
    #         if snr == None:
    #             convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}.tif", scale_info)
    #         else:
    #             convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{timestamp}_{alg}_snr{snr}.tif", scale_info)


    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

