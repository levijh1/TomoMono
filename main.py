if __name__ == '__main__':
    # ===================== Imports =====================
    import time
    import sys
    import torch
    from datetime import datetime
    import tomoDataClass
    from helperFunctions import DualLogger, convert_to_numpy, convert_to_tiff

    # ================== Configuration ==================
    log = False            # Enable logging to file
    saveToFile = True      # Enable saving data to file
    algorithms = ['SIRT_CUDA', 'svmbir']  # Algorithms to use

    # =============== Logging Setup =====================
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if log:
        sys.stdout = DualLogger(f'logs/time_test_output_tomoMono{timestamp}.txt', 'w')

    # ============== Runtime Info =======================
    print("Running TomoMono 3D Reconstruction Script!")
    print(f"Algorithms: {algorithms}")
    if torch.cuda.is_available():
        print("GPU is available")

    # ============ Input File Locations ================
    alignedTifFileLocations = [
        ("manualWithXCA&PMA", "alignedProjections/aligned_manuallyPrepped_XCA&PMA_20250522-145702.tif"),
        # ("manualWithPMA", "alignedProjections/aligned_manuallyPrepped_PMA_20250522-145702.tif"),
    ]

    # ============= Start Timer =========================
    start_time = time.time()

    # ============ Reconstruction Loop ==================
    for case, prealigned_tif_file in alignedTifFileLocations:
        print(f"\nReconstructing case: {case}")

        for alg in algorithms:
            print(f"  Using algorithm: {alg}")

            # Load and preprocess data
            obj, scale_info = convert_to_numpy(prealigned_tif_file)
            tomo = tomoDataClass.tomoData(obj)
            tomo.center_projections()
            tomo.crop_bottom_center(500, 750)

            # Normalize for specific algorithms
            if alg in ["SIRT_CUDA", "svmbir"]:
                tomo.normalize()
                isNormalized = "Normalized"
            else:
                isNormalized = "Raw"

            try:
                alg_start_time = time.time()
                tomo.reconstruct(algorithm=alg, snr_db=None)
                alg_end_time = time.time()

                # Save reconstruction
                if saveToFile:
                    out_path = f"reconstructions/foamRecon_{isNormalized}_{case}_{timestamp}_{alg}.tif"
                    convert_to_tiff(tomo.get_recon(), out_path, scale_info)

                print(f"    Completed in {alg_end_time - alg_start_time:.2f} seconds.")
            except Exception as e:
                print(f"    Failed to reconstruct using {alg}: {e}")

    # =============== End Timer =========================
    end_time = time.time()
    print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

    # ============ Restore Logging ======================
    if log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
