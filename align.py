if __name__ == '__main__':
    import time
    import sys
    import tomoDataClass
    from datetime import datetime
    from helperFunctions import DualLogger, convert_to_tiff, subpixel_shift, degree_to_positiveRadians, runwidget
    import tomopy
    import matplotlib.pyplot as plt
    import matplotlib


    # ### Matplotlib widget
    # from matplotlib.widgets import Slider
    # def runwidget(m):
    #     """Makes a movie of a list of images (3D array) that is good for running in a script"""
    #     fig, ax = plt.subplots(figsize=(9, 6))
    #     ax.imshow(m[0], vmin=np.min(m), vmax=np.max(m), cmap='gray')
    #     plt.title("Frame 0")

    #     plt.subplots_adjust(bottom=0.25)
    #     ax_slider = plt.axes([0.1,0.1, 0.8, 0.05], facecolor='teal')

    #     def update_line(indx):
    #         ax.clear()

    #         # zplot = m[indx].T
    #         # nem = 100
    #         # q = np.quantile(zplot[nem:-nem, nem:-nem], [0.01,0.99])
    #         # plt.imshow(m[indx], cmap='bone', vmin=q[0], vmax=q[1])
            
    #         ax.imshow(m[indx], vmin=np.min(m), vmax=np.max(m), cmap='gray')
    #         plt.title(f"Frame {indx}")
    #         plt.draw()

    #     slider = Slider(ax_slider, "Height (cross-section)", valmin=0, valmax=m.shape[0]-1, valinit=20, valstep = 1)
    #     slider.on_changed(update_line)

    #     plt.show()

    # -------------------------
    # CONFIGURATION FLAGS
    # -------------------------
    log = False           # Set to True to enable logging output to a file
    saveToFile = False     # Set to True to save aligned projection data to a TIFF file
    reconstruct = False     # Set to True to save the reconstruction to a TIFF file

    # -------------------------
    # SETUP: Timing & Logging
    # -------------------------
    start_time = time.time()  # Start execution timer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Generate timestamp for filenames
    print("timestamp:", timestamp)

    if log:
        # Redirect stdout/stderr to both console and log file
        sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')

    print("Running Image Registration Script")
    try:
        import cupy as cp
        if cp.is_available():
            print("CuPy GPU available — array operations will use GPU")
    except ImportError:
        pass

    # -------------------------
    # DATA IMPORT (EXAMPLE FOR REAL DATA)
    # -------------------------
    # Uncomment the following lines to use experimental projection data from a TIFF file:
    # tif_file = "alignedProjections/aligned_manually_3_3_25.tif"
    # obj, scale_info = convert_to_numpy(tif_file)
    # print(obj.shape)

    #Importing data from Taylor Buckway h5 file (APS data)
    import h5py
    import numpy as np
    # filename = r"/home/ljh79/TomoMono/data/noglow_tomo_128.hdf5"
    # filename = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final.hdf5'
    filename = "/Users/levihancock/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/IFE-STAR/IFE-Ptycho-Tomo/APS_2ID_GUP1013052_August_2025/levi_tomoReconstructions"


    with h5py.File(filename, "r") as f:
        data = np.array(f["data"][...])
        angles = degree_to_positiveRadians(list(f["angles"][...]))

    print("data shape is: ", data.shape)
    print("angles shape is: ", len(angles))

    runwidget(data)

    #Flip and crop data
    # data = np.flip(data, axis=1)
    # data = data[:, :-156, 402:-401]  # Crop to remove holder and focus on sample region
    # indicesToRemove = [0,1,2,3,4,5,53,60,67,69,82,90,178,183,186, 209, 218, 219, 221, 224, 254, 258, 276, 278, 304, 306] + list(range(308,325)) ##Look out for 200, 
    # noGlow_indicesToRemove = [54, 61, 76, 177, 180, 215, 248, 252, 301]

    # #removing images with holder in the way
    # # data = np.delete(data, noGlow_indicesToRemove, axis = 0)
    # # angles = np.delete(angles, noGlow_indicesToRemove, axis = 0)
    # # print("data shape after removing is: ", data.shape)
    # # print("angles shape after removing is: ", len(angles))

    tomo = tomoDataClass.tomoData(data, angles)
    scale_info = None

    # # # -------------------------
    # # # DATA IMPORT (EXAMPLE FOR SIMULATED DATA): Tomopy Simulated Projections (Shepp-Logan Phantom)
    # # # -------------------------
    # # numAngles = 800
    # # shepp3d = tomopy.shepp3d(size=128)  # Generate 3D Shepp-Logan phantom
    # # ang = tomopy.angles(nang=numAngles, ang1=0, ang2=360)  # Define projection angles
    # # obj = tomopy.project(shepp3d, ang, pad=False)  # Create projection data
    # # tomo = tomoDataClass.tomoData(obj)  # Wrap projections in tomoData class
    # # scale_info = None
    # # tomo.jitter(maxShift=5)  # Add random misalignment to simulate experimental shifts

    # # -------------------------
    # # ALIGNMENT INSTRUCTIONS
    # # -------------------------

    
    # """
    # Alignment Options (defined in alignment_methods.py):
    # - cross_correlate_align
    # - rotate_correlate_align
    # - vertical_mass_fluctuation_align
    # - tomopy_align            # TomoPy’s implementation of joint reprojection alignment (PMA)
    # - PMA                     # Custom projection matching algorithm
    # - optical_flow_align
    # - center_projections
    # """






    print("Starting alignment")

    savePath = f"/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/aligned_XC_summed_{timestamp}.tif"
    
    print("\n\nCreating aligned projections:", savePath)

    # # Ensure alignment begins from original, unmodified projections
    # tomo.reset_workingProjections(x_size = data.shape[2], y_size=data.shape[1])
    tomo.reset_workingProjections(x_size = 512+256, y_size=256, cropTopCenter=True) #if halfing input
    # tomo.reset_workingProjections(x_size = 1024+512, y_size=512, cropTopCenter=True)


    # -------------------------
    # ALIGNMENT STRATEGY
    # -------------------------
    # Choose and configure alignment algorithm below:
    tomo.shift_min_to_middle()
    tomo.cross_correlate_align_to_sum(tolerance=0.01, max_iterations = 2, yROI_Range=[0, tomo.workingProjections.shape[1]], xROI_Range=[0, tomo.workingProjections.shape[2]], maxShiftTolerance=2)
    # tomo.PMA(max_iterations = 5, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = tomo.workingProjections.shape[1]-80, crop_bottom_center_x = 1200, isPhaseData = True)
    tomo.center_projections()


    # # Choose and configure alignment algorithm below:
    # tomo.shift_min_to_middle()
    # tomo.cross_correlate_align_to_sum(tolerance=0.01, max_iterations = 10, yROI_Range=[0, tomo.workingProjections.shape[1]], xROI_Range=[0, tomo.workingProjections.shape[2]], maxShiftTolerance=2)
    # tomo.PMA(max_iterations = 5, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = tomo.workingProjections.shape[1]-80, crop_bottom_center_x = 1200, isPhaseData = True)
    # tomo.center_projections()

    #  # data = data[:, :-156, 202:-201]  # Crop to remove holder and focus on sample region
    # tomo.center_projections()
    # tomo.cross_correlate_align(tolerance=0.01, max_iterations = 2, yROI_Range=[0, tomo.workingProjections.shape[1]], xROI_Range=[0, tomo.workingProjections.shape[2]], maxShiftTolerance=2, isFull360=False)
    # tomo.center_projections()
    # tomo.cross_correlate_align(tolerance=0.01, max_iterations = 2, yROI_Range=[0, tomo.workingProjections.shape[1]], xROI_Range=[200, -200], maxShiftTolerance=2, isFull360=False)
    # # tomo.PMA(max_iterations = 5, tolerance=0.0001, algorithm="SIRT_CUDA", crop_bottom_center_y = tomo.workingProjections.shape[1]-80, crop_bottom_center_x = 1200, isPhaseData = True)
    # tomo.center_projections()

    # Apply the computed shifts to original data to finalize alignment
    tomo.make_updates_shift()

    runwidget(tomo.get_finalProjections())
    # -------------------------
    # SAVE RESULTS (Optional)
    # -------------------------
    if saveToFile:
        convert_to_tiff(tomo.get_finalProjections(), savePath, scale_info)
    if reconstruct:
        tomo.reconstruct(algorithm="SIRT_CUDA", snr_db=None)
        recon_path = f"reconstructions/APSbeamtime_Oct25/recon_XC_{timestamp}.tif"
        convert_to_tiff(tomo.get_recon(), recon_path, scale_info)




