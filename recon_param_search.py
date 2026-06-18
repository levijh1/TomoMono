def save_orthogonal_slices(tomo_obj, output_path):
    """
    Save orthogonal slices through the reconstruction to a PNG file.
    """
    import matplotlib.pyplot as plt
    recon = tomo_obj.recon
    nz, ny, nx = recon.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(recon[cz, :, :], cmap='gray', aspect='equal')
    axes[0].set_title(f'XY  (z={cz})')

    axes[1].imshow(recon[:, cy, :], cmap='gray', aspect='auto')
    axes[1].set_title(f'XZ  (y={cy})')

    axes[2].imshow(recon[:, :, cx], cmap='gray', aspect='auto')
    axes[2].set_title(f'YZ  (x={cx})')

    for ax in axes:
        ax.axis('off')

    plt.suptitle('Orthogonal slices through reconstruction')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main(args):
    # ===================== Imports =====================
    import os
    import sys
    import time
    import torch
    import h5py
    import numpy as np
    import tomopy
    import matplotlib.pyplot as plt
    from datetime import datetime

    import tomoDataClass
    from helperFunctions import DualLogger, convert_to_numpy, convert_to_tiff
    from metrics import reprojection_consistency_score

    # ================== Configuration ==================
    TIFF_FILE  = args.tiff_file
    BASE_OUTPUT_DIR = args.output_dir or '/home/ljh79/TomoMono/reconstructions/APSbeamtime_Oct25/tomopy'
    RAW_HDF5   = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5'
    DROP_ANGLES = [19, 26]
    LOG        = True
    SAVE       = args.save

    # Algorithm comparison.
    # Note: GRIDREC_CUDA and TV_CUDA have no GPU implementations in ASTRA or TomoPy;
    # they are routed to the CPU gridrec/tv algorithms below.
    ALGORITHMS = [
        {'label': 'SIRT_CUDA',  'algorithm': 'SIRT_CUDA'},
        {'label': 'ART_CUDA',   'algorithm': 'ART_CUDA'},
        {'label': 'FBP_CUDA',   'algorithm': 'FBP_CUDA'},
        {'label': 'gridrec',    'algorithm': 'gridrec'},   # GRIDREC_CUDA does not exist
        {'label': 'tv',         'algorithm': 'tv'},         # TV_CUDA does not exist
    ]

    # SIRT_CUDA hyperparameter variations.
    # Varying: number of iterations (convergence) and positivity constraint.
    SIRT_VARIANTS = [
        {'label': 'SIRT_CUDA_100iter',    'num_iter': 100,  'extra_options': {}},
        {'label': 'SIRT_CUDA_200iter',    'num_iter': 200,  'extra_options': {}},
        {'label': 'SIRT_CUDA_400iter',    'num_iter': 400,  'extra_options': {}},
        {'label': 'SIRT_CUDA_600iter',    'num_iter': 600,  'extra_options': {}},
        {'label': 'SIRT_CUDA_positivity', 'num_iter': 400,  'extra_options': {'MinConstraint': 0}},
    ]

    # =============== Logging Setup =====================
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'tomopyAlgorithmCheck_{timestamp}')
    SLICES_DIR = os.path.join(OUTPUT_DIR, 'orthogonal_slices')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SLICES_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    if LOG:
        sys.stdout = DualLogger(f'logs/main_tomopy_{timestamp}.txt', 'w')

    # ============== Runtime Info =======================
    print('=' * 72)
    print('TomoMono 3D Reconstruction — tomopy algorithm comparison')
    print(f'Started:  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'TIFF:     {TIFF_FILE}')
    print(f'Output:   {OUTPUT_DIR}')
    print('=' * 72)
    if torch.cuda.is_available():
        print('GPU available (CUDA/ASTRA)')
    try:
        import cupy as cp
        if cp.is_available():
            print('CuPy GPU available — array ops on GPU')
    except ImportError:
        pass

    # ============== Load Data ==========================
    print(f'\nLoading projections: {TIFF_FILE}')
    t0 = time.time()
    projections, scale_info = convert_to_numpy(TIFF_FILE)
    projections = projections.astype(np.float32)
    print(f'  shape (n_angles, h, w): {projections.shape}  [{time.time()-t0:.1f}s]')

    # ============== Crop: y-slice range + centered x width ==================
    h, w = projections.shape[1], projections.shape[2]
    y_start = args.y_start if args.y_start is not None else 0
    y_end   = args.y_end   if args.y_end   is not None else h
    y_start, y_end = max(0, y_start), min(h, y_end)

    if args.width is not None and args.width < w:
        cx   = w // 2
        half = args.width // 2
        projections = projections[:, y_start:y_end, cx - half : cx + half]
    else:
        projections = projections[:, y_start:y_end, :]

    print(f'\nCrop: y=[{y_start}, {y_end})  '
          f'x=[{w//2 - (args.width or w)//2}, {w//2 + (args.width or w)//2})'
          f'  → shape {projections.shape}')

    # ============== Load Angles ========================
    print(f'\nLoading angles from: {RAW_HDF5}')
    with h5py.File(RAW_HDF5, 'r') as hf:
        ang_deg = hf['angles'][...]
    ang_rad = ang_deg * np.pi / 180.0
    if DROP_ANGLES:
        ang_rad = np.delete(ang_rad, DROP_ANGLES, axis=0)
    angles = (ang_rad - np.mean(ang_rad)).astype(np.float32)
    assert len(angles) == projections.shape[0], (
        f'Angle count {len(angles)} != projection count {projections.shape[0]}. '
        f'Check DROP_ANGLES.'
    )
    print(f'  angles: {len(angles)}  '
          f'range [{np.degrees(angles.min()):.2f}, {np.degrees(angles.max()):.2f}] deg')

    # ============== Normalize ==========================
    # Normalize before building tomoData so finalProjections has clean data.
    # Phase data: invert sign then scale to [0, 1].
    print('\nNormalizing (phase data: invert + scale to [0, 1])...')
    projections = -projections
    projections -= projections.min()
    projections /= projections.max()

    # ============== Build tomoData =====================
    print('\nCreating tomoData object...')
    tomo_base = tomoDataClass.tomoData(projections, angles)
    rotation_center = float(tomopy.find_center_vo(tomo_base.finalProjections))
    tomo_base.center_offset = rotation_center - tomo_base.image_size[1] / 2
    print(f'Rotation center: {rotation_center:.2f}  center_offset: {tomo_base.center_offset:.2f}')
    del projections  # free memory; tomo_base holds the data

    name_stem = os.path.splitext(os.path.basename(TIFF_FILE))[0]
    total_start = time.time()

    # =========================================================
    # Helper: reconstruct, score, and save one configuration
    # =========================================================
    # reconstruct() only writes to tomo.recon — it never modifies finalProjections
    # or ang — so we safely reuse tomo_base across all configs (no deepcopy needed,
    # which matters for ~160 GB full-res projections).
    def run_config(label, algorithm, num_iter=400, extra_options=None):
        print(f'\n{"─"*60}')
        print(f'Config: {label}')
        extra_options = extra_options or {}

        try:
            t_start = time.time()
            tomo_base.reconstruct(algorithm=algorithm, num_iter=num_iter, extra_options=extra_options)
            elapsed = time.time() - t_start
            print(f'  Reconstruction completed in {elapsed:.1f}s')
        except Exception as e:
            print(f'  FAILED: {e}')
            return

        try:
            tomo_base.finalReprojections = None  # Clear cached reprojections for fresh RCS per config
            rcs, _, _ = reprojection_consistency_score(tomo_base, plot=False, normalize_method='affine')
            _, fsc_resolutions, _ = tomo_base.fourier_shell_correlation(algorithm=algorithm, plot=False)
            fsc_res = fsc_resolutions.get('half-bit')
            fsc_str = f'{fsc_res:.4f}' if fsc_res is not None else 'N/A'
            print(f'  RCS = {rcs:.4f}  |  FSC(half-bit) = {fsc_str} px')
        except Exception as e:
            print(f'  Metrics failed: {e}')

        if SAVE:
            out_path = os.path.join(OUTPUT_DIR, f'{name_stem}_{label}_{timestamp}.tif')
            convert_to_tiff(tomo_base.get_recon(), out_path, scale_info)
            print(f'  Saved: {out_path}')

            slices_path = os.path.join(SLICES_DIR, f'{name_stem}_{label}_{timestamp}_slices.png')
            save_orthogonal_slices(tomo_base, slices_path)
            print(f'  Saved orthogonal slices: {slices_path}')

    # =========================================================
    # Section 1: Algorithm Comparison
    # =========================================================
    print('\n' + '=' * 72)
    print('SECTION 1 — Algorithm Comparison')
    print('  SIRT_CUDA  : ASTRA GPU SIRT (400 iter)')
    print('  ART_CUDA   : ASTRA GPU ART  (400 iter)')
    print('  FBP_CUDA   : ASTRA GPU FBP')
    print('  gridrec    : TomoPy CPU gridrec  (GRIDREC_CUDA has no GPU version)')
    print('  tv         : TomoPy CPU TV       (TV_CUDA has no GPU version)')
    print('=' * 72)

    for cfg in ALGORITHMS:
        run_config(cfg['label'], cfg['algorithm'])

    # =========================================================
    # Section 2: SIRT_CUDA Hyperparameter Sweep
    # =========================================================
    print('\n' + '=' * 72)
    print('SECTION 2 — SIRT_CUDA Hyperparameter Sweep')
    print('  Varying: num_iter (100/200/400/600) and positivity constraint')
    print('=' * 72)

    for v in SIRT_VARIANTS:
        run_config(
            v['label'],
            'SIRT_CUDA',
            num_iter=v['num_iter'],
            extra_options=v['extra_options'],
        )

    # =============== End ===================================
    total_elapsed = time.time() - total_start
    print(f'\n{"=" * 72}')
    print(f'All configs completed in {total_elapsed/60:.1f} min.')

    if LOG:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='TomoMono 3D reconstruction with algorithm comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tiff-file',   type=str, required=True,
                        help='Path to aligned projections TIFF')
    parser.add_argument('--y-start',     type=int, default=None,
                        help='Y crop start index (inclusive)')
    parser.add_argument('--y-end',       type=int, default=None,
                        help='Y crop end index (exclusive)')
    parser.add_argument('--width',       type=int, default=None,
                        help='Centered detector width crop in pixels')
    parser.add_argument('--output-dir',  type=str, default=None,
                        help='Output directory for reconstructions')
    parser.add_argument('--no-save',     action='store_true', default=False,
                        help='Do not save reconstructions to disk')
    args = parser.parse_args()
    args.save = not args.no_save

    main(args)
