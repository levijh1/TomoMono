if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from tiffConverter import convert_to_numpy, convert_to_tiff
    from datetime import datetime
    import torch
    import argparse
    from helperFunctions import DualLogger
    from tqdm import tqdm
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
                        default = ["SIRT_CUDA", "tv", "svmbir"])
                        # default = ['SIRT_CUDA'])
    args = parser.parse_args()
    algorithms = args.algorithms

    #Or just specify the exact algorithms you want to use here
    # algorithms = ['SIRT_CUDA', 'svmbir']

    # Setup logging if enabled
    if log:
        sys.stdout = DualLogger(f'logs/output_tomoMono{timestamp}.txt', 'w')








    print("Running TomoMono 3D Reconstruction Script!")
    print(f"Algorithms: {algorithms}")

    # Check for GPU availability
    if torch.cuda.is_available():
        print("GPU is available")


    # # Reconstruction Process
    alignedTifFileLocations = [("baseCase", "alignedProjections/aligned_baseCase_Filter_XCtip_20241004-100555.tif")
                               # ,("rotate", "alignedProjections/aligned_rotate_Filter_XCtip_20241004-100555.tif")
                               # ,("unrotate", "alignedProjections/aligned_unRotate_Filter_XCtip_20241004-100555.tif")
                              ]
    for case, prealigned_tif_file in alignedTifFileLocations:
        print("Reconstructing")
        algorithms = ["SIRT_CUDA", "svmbir"]
        for alg in algorithms:
            obj, scale_info = convert_to_numpy(prealigned_tif_file)
            tomo = tomoDataClass.tomoData(obj)
            tomo.center_projections()
            tomo.crop_bottom_center(500, 750)
            if alg == "SIRT_CUDA":
                try:
                    alg_start_time = time.time()
                    tomo.reconstruct(algorithm=alg, snr_db = None)
                    alg_end_time = time.time()
                    if saveToFile:
                        convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_NotNormalized_{case}_{timestamp}_{alg}.tif", scale_info)
                        
                    # tomo.normalize()
                    # tomo.reconstruct(algorithm=alg, snr_db = None)
                    # if saveToFile:
                    #     convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_Normalized_{case}_{timestamp}_{alg}.tif", scale_info)
                    print(f"{alg} Algorithm completed reconstruction in {alg_end_time - alg_start_time} seconds."
                except Exception as e:
                    print(f"Failed to reconstruct using {alg}: {e}")
                    continue

            elif alg == "svmbir":
                # tomo.normalize()
                # for snr in [20,25,30,35,40]:
                #     try:
                #         tomo.reconstruct(algorithm=alg, snr_db = snr)
                #         if saveToFile:
                #             convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_Normalized_{timestamp}_{alg}{snr}.tif", scale_info)
                #     except Exception as e:
                #         print(f"Failed to reconstruct using {alg}: {e}")
                #         continue
                tomo.normalize()
                try:
                    alg_start_time = time.time()
                    tomo.reconstruct(algorithm=alg, snr_db = None)
                    alg_end_time = time.time()
                    if saveToFile:
                        convert_to_tiff(tomo.get_recon(), f"reconstructions/foamRecon_Normalized_{timestamp}_{alg}{snr}.tif", scale_info)
                    print(f"{alg} Algorithm completed reconstruction in {alg_end_time - alg_start_time} seconds."
                except Exception as e:
                    print(f"Failed to reconstruct using {alg}: {e}")
                    continue



    # End the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Script completed in {end_time - start_time} seconds.")

    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

