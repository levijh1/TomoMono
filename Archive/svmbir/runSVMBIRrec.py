#!/usr/bin/env python3
"""
SVMBIR reconstruction from pre-aligned projections (Oct25 APS beamtime data).

Usage:
    python runSVMBIRrec.py --tiff-file <path> --y-start N --y-end N --width N [options]

Cluster submission (three separate jobs):
    sbatch runSVMBIRrec_4xds.sh
    sbatch runSVMBIRrec_2xds.sh
    sbatch runSVMBIRrec_fullres.sh
"""

import sys
import os
import argparse
import time
from datetime import datetime

import numpy as np
import h5py
import tomopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from helperFunctions import DualLogger, convert_to_numpy, convert_to_tiff
import tomoDataClass
from alignment_methods import reprojection_consistency_score

RAW_HDF5    = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5'
DROP_ANGLES = [19, 26]


def _correct_svmbir_geometry(recon):
    """Align SVMBIR reconstruction to TomoPy coordinate system.

    SVMBIR uses different rotation direction and detector conventions than TomoPy.
    This function applies the necessary transformations to match TomoPy's geometry.
    """
    recon = np.flip(recon, axis=2)  # flip x-axis
    recon = np.rot90(recon, k=1, axes=(1, 2))  # rotate 90° in XY plane
    return recon


def load_angles(hdf5_path, drop_indices):
    with h5py.File(hdf5_path, 'r') as hf:
        ang_deg = hf['angles'][...]
    ang_rad = ang_deg * np.pi / 180.0
    if drop_indices:
        ang_rad = np.delete(ang_rad, drop_indices, axis=0)
    return ang_rad - np.mean(ang_rad)


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    hostname  = os.environ.get('HOSTNAME', os.uname().nodename)

    output_dir  = args.output_dir or os.path.join(_SCRIPT_DIR, 'reconstructions', 'APSbeamtime_Oct25')
    name_stem   = args.output_name or os.path.splitext(os.path.basename(args.tiff_file))[0]
    log_path    = os.path.join(_SCRIPT_DIR, 'logs', f'svmbir_{name_stem}_{timestamp}.txt')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'logs'), exist_ok=True)

    sys.stdout = DualLogger(log_path, 'w')

    print('=' * 72, flush=True)
    print(f'SVMBIR reconstruction  |  host: {hostname}  |  pid: {os.getpid()}', flush=True)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'TIFF:    {args.tiff_file}', flush=True)
    print('=' * 72, flush=True)

    t_total = time.time()

    # ── Load projections ──────────────────────────────────────────────────────
    print(f'\nLoading projections: {args.tiff_file}', flush=True)
    t0 = time.time()
    projections, scale_info = convert_to_numpy(args.tiff_file)
    projections = projections.astype(np.float32)
    print(f'  shape (n_angles, h, w): {projections.shape}  [{time.time()-t0:.1f}s]', flush=True)

    # ── Load angles ───────────────────────────────────────────────────────────
    print(f'\nLoading angles from: {RAW_HDF5}', flush=True)
    angles = load_angles(RAW_HDF5, DROP_ANGLES).astype(np.float32)
    assert len(angles) == projections.shape[0], (
        f'Angle count {len(angles)} != projection count {projections.shape[0]}. '
        f'Check DROP_ANGLES.'
    )
    print(f'  angles: {len(angles)}  '
          f'range [{np.degrees(angles.min()):.2f}, {np.degrees(angles.max()):.2f}] deg', flush=True)

    # ── Crop: y-slice range + centered x width ────────────────────────────────
    h, w = projections.shape[1], projections.shape[2]
    y_start = args.y_start if args.y_start is not None else 0
    y_end   = args.y_end   if args.y_end   is not None else h
    y_start, y_end = max(0, y_start), min(h, y_end)

    if args.width is not None and args.width < w:
        cx   = w // 2
        half = args.width // 2
        projs = projections[:, y_start:y_end, cx - half : cx + half]
    else:
        projs = projections[:, y_start:y_end, :]

    print(f'\nCrop: y=[{y_start}, {y_end})  '
          f'x=[{w//2 - (args.width or w)//2}, {w//2 + (args.width or w)//2})'
          f'  → shape {projs.shape}', flush=True)
    del projections

    # ── Normalize (phase data) before creating tomoData ───────────────────────
    # normalize() in tomoDataClass only updates workingProjections, not
    # finalProjections — doing it here ensures reconstruct() sees clean data.
    print('\nNormalizing (phase data: invert + scale to [0, 1])...', flush=True)
    projs = -projs
    projs -= projs.min()
    projs /= projs.max()

    # ── Build tomoData, find rotation center, reconstruct ─────────────────────
    print('\nCreating tomoData object...', flush=True)
    tomo = tomoDataClass.tomoData(projs, angles)

    center = float(tomopy.find_center_vo(tomo.finalProjections))
    tomo.center_offset = center - tomo.image_size[1] / 2
    print(f'Rotation center: {center:.2f}  center_offset: {tomo.center_offset:.2f}', flush=True)

    print('\nRunning SVMBIR reconstruction...', flush=True)
    t_recon = time.time()
    tomo.reconstruct(algorithm='svmbir')
    print(f'Reconstruction completed in {(time.time()-t_recon)/60:.1f} min', flush=True)

    # ── Quality metrics ───────────────────────────────────────────────────────
    print('\nComputing reconstruction quality metrics...', flush=True)
    rcs, _, _ = reprojection_consistency_score(tomo, plot=False, normalize_method='affine')
    _, fsc_resolutions, _ = tomo.fourier_shell_correlation(algorithm='svmbir', plot=False)
    fsc_res = fsc_resolutions.get('FSC=0.143')
    fsc_str = f"{fsc_res:.4f}" if fsc_res is not None else "N/A"
    print(f'RCS = {rcs:.4f}  |  FSC(0.143) = {fsc_str} px', flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, f'{name_stem}_svmbir_{timestamp}.tif')
    print(f'\nSaving: {out_path}', flush=True)
    convert_to_tiff(tomo.get_recon(), out_path, scale_info)

    print(f'\nTotal wall time: {(time.time()-t_total)/60:.1f} min', flush=True)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
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
                        help='Output directory (default: reconstructions/APSbeamtime_Oct25)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename stem (default: TIFF basename)')
    main(parser.parse_args())
