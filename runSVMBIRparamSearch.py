#!/usr/bin/env python3
"""
SVMBIR parameter search — focus on max_resolutions values (2-8) and sharpness tuning
with default snr_db=30.0 and positivity constraint.

Usage:
    python runSVMBIRparamSearch.py --tiff-file <path> [options]

Options:
    --tiff-file     Path to aligned projections TIFF (required)
    --y-start N     Y crop start (inclusive)
    --y-end   N     Y crop end (exclusive)
    --width   N     Centered detector width in pixels
    --output-dir    Base directory; script creates dated recons/ and slices/ subdirs
                    (default: reconstructions/APSbeamtime_Oct25/svmbir)
    --csv-file      CSV file path to save results (default: hyperparam_results/svmbir_search_<timestamp>.csv)
    --configs       Comma-separated config names to run (default: all)
    --fsc           Also compute FSC resolution for each config (very slow: 3x compute)
    --list-configs  Print all available config names and exit
    --num-workers N Run N configs in parallel (default: 1)

Current focus (15 configs): maxres=2-8 sweep + sharpness tuning on maxres=3,4 + p=2.0 variants
"""

import sys
import os
import argparse
import csv
import time
import concurrent.futures
from datetime import datetime

import numpy as np
import h5py
import tomopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from helperFunctions import DualLogger, convert_to_numpy, convert_to_tiff
import tomoDataClass
from alignment_methods import reprojection_consistency_score, fourier_shell_correlation

try:
    import svmbir
except ImportError:
    print("ERROR: svmbir not installed — activate tomoMono conda env")
    sys.exit(1)


def _correct_svmbir_geometry(recon):
    """Align SVMBIR reconstruction to TomoPy coordinate system.

    SVMBIR uses different rotation direction and detector conventions than TomoPy.
    This function applies the necessary transformations to match TomoPy's geometry.
    """
    recon = np.flip(recon, axis=2)  # flip x-axis
    recon = np.rot90(recon, k=1, axes=(1, 2))  # rotate 90° in XY plane
    return recon

RAW_HDF5    = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5'
DROP_ANGLES = [19, 26]

# ---------------------------------------------------------------------------
# Parameter configurations
# Each dict is one SVMBIR run. Keys map to svmbir.recon() kwargs.
# 'name' is used for filenames and the results table.
# Focus: higher maxres values (2-8) with default snr_db and sharpness sweep
# ---------------------------------------------------------------------------
ALL_CONFIGS = [
    # ── maxres sweep with default snr_db=30.0 ──────────────────────────────
    dict(name='maxres_2',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=2, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_3',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=3, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_4',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=4, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_5',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=5, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_6',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=6, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_7',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=7, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_8',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=8, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres_9',
         snr_db=30.0, sharpness=0.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=9, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    # ── sharpness sweep with maxres=3 ───────────────────────────────────────
    dict(name='maxres3_sharp_neg1',
         snr_db=30.0, sharpness=-1.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=3, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres3_sharp_pos1',
         snr_db=30.0, sharpness=1.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=3, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    # ── sharpness sweep with maxres=4 ───────────────────────────────────────
    dict(name='maxres4_sharp_neg1',
         snr_db=30.0, sharpness=-1.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=4, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres4_sharp_pos1',
         snr_db=30.0, sharpness=1.0, p=1.2, q=2.0, T=1.0,
         max_resolutions=4, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    # ── Cleaner prior (p=2.0) with higher maxres ──────────────────────────
    dict(name='maxres5_p2p0',
         snr_db=30.0, sharpness=0.0, p=2.0, q=2.0, T=2.0,
         max_resolutions=5, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres6_p2p0',
         snr_db=30.0, sharpness=0.0, p=2.0, q=2.0, T=2.0,
         max_resolutions=6, positivity=True, weight_type='unweighted',
         b_interslice=1.0),

    dict(name='maxres7_p2p0',
         snr_db=30.0, sharpness=0.0, p=2.0, q=2.0, T=2.0,
         max_resolutions=7, positivity=True, weight_type='unweighted',
         b_interslice=1.0),
]

CONFIG_MAP = {c['name']: c for c in ALL_CONFIGS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_angles(hdf5_path, drop_indices):
    with h5py.File(hdf5_path, 'r') as hf:
        ang_deg = hf['angles'][...]
    ang_rad = ang_deg * np.pi / 180.0
    if drop_indices:
        ang_rad = np.delete(ang_rad, drop_indices, axis=0)
    return ang_rad - np.mean(ang_rad)


def _fmt(val, width=8, decimals=4):
    if val is None:
        return 'N/A'.center(width)
    return f'{val:.{decimals}f}'.rjust(width)


def print_table_header(do_fsc):
    cols  = ['Name', 'snr_db', 'sharp', 'p', 'T', 'maxres', 'pos', 'wt', 'time(m)', 'RCS']
    widths = [14,      7,       7,      5,  6,   7,    4,    14,    8,     8]
    if do_fsc:
        cols.append('FSC0.143')
        widths.append(9)
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    hdr = '|' + '|'.join(f' {c:<{w}} ' for c, w in zip(cols, widths)) + '|'
    print(sep)
    print(hdr)
    print(sep)
    return sep, widths, do_fsc


def print_table_row(cfg, elapsed_min, rcs, fsc_res, sep, widths, do_fsc):
    maxres_str = str(cfg['max_resolutions']) if cfg['max_resolutions'] is not None else 'auto'
    pos_str = 'T' if cfg['positivity'] else 'F'
    wt_str  = cfg['weight_type']

    vals = [
        cfg['name'],
        str(cfg['snr_db']),
        str(cfg['sharpness']),
        str(cfg['p']),
        str(cfg['T']),
        maxres_str,
        pos_str,
        wt_str,
        f'{elapsed_min:.1f}',
        _fmt(rcs).strip(),
    ]
    if do_fsc:
        vals.append(_fmt(fsc_res).strip() if fsc_res else 'N/A')

    row = '|' + '|'.join(f' {v:<{w}} ' for v, w in zip(vals, widths)) + '|'
    print(row)
    print(sep)


def save_ortho_slices(recon, path, name):
    """Save a 3-panel orthogonal slice figure (mid-Z, mid-Y, mid-X)."""
    nz, ny, nx = recon.shape
    slices = [
        (recon[nz // 2],       'Z mid-slice (axial)'),
        (recon[:, ny // 2, :], 'Y mid-slice (sagittal)'),
        (recon[:, :, nx // 2], 'X mid-slice (coronal)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'SVMBIR: {name}', fontsize=13)
    for ax, (img, title) in zip(axes, slices):
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto',
                  interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_csv(results, csv_path, do_fsc):
    """Write parameter search results to a CSV file."""
    fieldnames = [
        'name', 'snr_db', 'sharpness', 'p', 'q', 'T',
        'max_resolutions', 'positivity', 'weight_type', 'b_interslice',
        'elapsed_min', 'rcs',
    ]
    if do_fsc:
        fieldnames.append('fsc_0143')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r['cfg']
            row = {
                'name':           cfg['name'],
                'snr_db':         cfg['snr_db'],
                'sharpness':      cfg['sharpness'],
                'p':              cfg['p'],
                'q':              cfg['q'],
                'T':              cfg['T'],
                'max_resolutions': cfg['max_resolutions'] if cfg['max_resolutions'] is not None else 'auto',
                'positivity':     cfg['positivity'],
                'weight_type':    cfg['weight_type'],
                'b_interslice':   cfg['b_interslice'],
                'elapsed_min':    f'{r["elapsed_min"]:.2f}' if r['elapsed_min'] is not None else '',
                'rcs':            f'{r["rcs"]:.6f}' if r['rcs'] is not None else '',
            }
            if do_fsc:
                row['fsc_0143'] = f'{r["fsc_res"]:.6f}' if r['fsc_res'] is not None else ''
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess for parallel execution)
# ---------------------------------------------------------------------------

def run_single_config(task):
    """Run one SVMBIR reconstruction.  Called in worker processes when --num-workers > 1."""
    import os, time
    import numpy as np
    import tomopy
    import svmbir

    sys.path.insert(0, _SCRIPT_DIR)
    import tomoDataClass
    from alignment_methods import reprojection_consistency_score, fourier_shell_correlation
    from helperFunctions import convert_to_tiff

    # Set OMP threads so workers share CPUs evenly
    threads_per_worker = max(1, task['total_cpus'] // task['num_workers'])
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)

    cfg          = task['cfg']
    projs        = task['projs']
    angles       = task['angles']
    center       = task['center']
    center_offset = task['center_offset']
    recons_dir   = task['recons_dir']
    slices_dir   = task['slices_dir']
    timestamp    = task['timestamp']
    scale_info   = task['scale_info']
    do_fsc       = task['do_fsc']
    name         = cfg['name']

    print(f'[pid={os.getpid()}] Starting: {name}  (OMP_NUM_THREADS={threads_per_worker})', flush=True)

    tomo = tomoDataClass.tomoData(projs.copy(), angles.copy())
    tomo.center_offset  = center_offset
    tomo.rotation_center = center

    svmbir_kwargs = dict(
        center_offset=center_offset,
        snr_db=cfg['snr_db'],
        sharpness=cfg['sharpness'],
        p=cfg['p'],
        q=cfg['q'],
        T=cfg['T'],
        positivity=cfg['positivity'],
        weight_type=cfg['weight_type'],
        b_interslice=cfg['b_interslice'],
        verbose=1,
    )
    if cfg['max_resolutions'] is not None:
        svmbir_kwargs['max_resolutions'] = cfg['max_resolutions']

    t_recon = time.time()
    tomo.recon = svmbir.recon(tomo.finalProjections, tomo.ang, **svmbir_kwargs)
    tomo.recon = _correct_svmbir_geometry(tomo.recon)
    tomo.recon = tomopy.circ_mask(tomo.recon, axis=0, ratio=0.99)
    elapsed_min = (time.time() - t_recon) / 60.0
    print(f'[pid={os.getpid()}] Done: {name}  ({elapsed_min:.1f} min)', flush=True)

    # Save TIFF reconstruction
    out_path = os.path.join(recons_dir, f'svmbir_{name}_{timestamp}.tif')
    convert_to_tiff(tomo.recon, out_path, scale_info)
    print(f'[pid={os.getpid()}] Saved TIFF: {out_path}', flush=True)

    # Save orthogonal slices figure
    slices_path = os.path.join(slices_dir, f'svmbir_{name}_{timestamp}.png')
    save_ortho_slices(tomo.recon, slices_path, name)
    print(f'[pid={os.getpid()}] Saved slices: {slices_path}', flush=True)

    # Metrics
    print(f'[pid={os.getpid()}] Computing RCS: {name}', flush=True)
    rcs, _, _ = reprojection_consistency_score(tomo, plot=False, normalize_method='affine')

    fsc_res = None
    if do_fsc:
        print(f'[pid={os.getpid()}] Computing FSC: {name}', flush=True)
        try:
            _, fsc_resolutions, _ = fourier_shell_correlation(tomo, algorithm='svmbir', plot=False)
            fsc_res = fsc_resolutions.get('FSC=0.143')
        except Exception as e:
            print(f'[pid={os.getpid()}] FSC failed ({name}): {e}', flush=True)

    print(f'[pid={os.getpid()}] RCS={rcs:.4f}' +
          (f'  FSC(0.143)={fsc_res:.4f} px' if fsc_res else '') +
          f'  [{name}]', flush=True)

    return dict(cfg=cfg, rcs=rcs, fsc_res=fsc_res, elapsed_min=elapsed_min)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.list_configs:
        print("Available configurations:")
        for c in ALL_CONFIGS:
            print(f"  {c['name']:<20} snr_db={c['snr_db']} sharpness={c['sharpness']} "
                  f"p={c['p']} T={c['T']} max_resolutions={c['max_resolutions']} "
                  f"positivity={c['positivity']} weight_type={c['weight_type']}")
        return

    # Select configs
    if args.configs:
        names = [n.strip() for n in args.configs.split(',')]
        bad = [n for n in names if n not in CONFIG_MAP]
        if bad:
            print(f"ERROR: unknown config names: {bad}")
            print(f"Available: {list(CONFIG_MAP)}")
            sys.exit(1)
        configs = [CONFIG_MAP[n] for n in names]
    else:
        configs = ALL_CONFIGS

    timestamp   = datetime.now().strftime('%Y%m%d-%H%M%S')
    hostname    = os.environ.get('HOSTNAME', os.uname().nodename)
    total_cpus  = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    num_workers = min(args.num_workers, len(configs))

    base_dir = args.output_dir or os.path.join(_SCRIPT_DIR, 'reconstructions', 'APSbeamtime_Oct25', 'svmbir')
    recons_dir = os.path.join(base_dir, f'svmbir_recons_{timestamp}')
    slices_dir = os.path.join(base_dir, f'svmbir_slices_{timestamp}')
    csv_dir    = os.path.join(_SCRIPT_DIR, 'hyperparam_results')

    # Use specified CSV file if provided, otherwise create a new timestamped one
    if args.csv_file:
        csv_path = args.csv_file
    else:
        csv_path = os.path.join(csv_dir, f'svmbir_search_{timestamp}.csv')

    log_path   = os.path.join(_SCRIPT_DIR, 'logs', f'svmbir_search_{timestamp}.txt')

    os.makedirs(recons_dir, exist_ok=True)
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(csv_dir,    exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'logs'), exist_ok=True)

    sys.stdout = DualLogger(log_path, 'w')

    print('=' * 80, flush=True)
    print(f'SVMBIR parameter search  |  host: {hostname}  |  pid: {os.getpid()}', flush=True)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'TIFF:    {args.tiff_file}', flush=True)
    print(f'Configs: {len(configs)}  |  FSC: {"yes" if args.fsc else "no"}  |  '
          f'workers: {num_workers}  |  total CPUs: {total_cpus}', flush=True)
    print(f'Recons:  {recons_dir}', flush=True)
    print(f'Slices:  {slices_dir}', flush=True)
    print(f'CSV:     {csv_path}', flush=True)
    print('=' * 80, flush=True)

    t_total = time.time()

    # ── Load projections (once) ───────────────────────────────────────────────
    print(f'\nLoading projections: {args.tiff_file}', flush=True)
    projections, scale_info = convert_to_numpy(args.tiff_file)
    projections = projections.astype(np.float32)
    print(f'  shape (n_angles, h, w): {projections.shape}', flush=True)

    # ── Load angles ───────────────────────────────────────────────────────────
    print(f'\nLoading angles from: {RAW_HDF5}', flush=True)
    angles = load_angles(RAW_HDF5, DROP_ANGLES).astype(np.float32)
    assert len(angles) == projections.shape[0], (
        f'Angle count {len(angles)} != projection count {projections.shape[0]}')
    print(f'  angles: {len(angles)}  '
          f'range [{np.degrees(angles.min()):.2f}, {np.degrees(angles.max()):.2f}] deg', flush=True)

    # ── Crop ─────────────────────────────────────────────────────────────────
    h, w = projections.shape[1], projections.shape[2]
    y_start = args.y_start if args.y_start is not None else 0
    y_end   = args.y_end   if args.y_end   is not None else h
    y_start, y_end = max(0, y_start), min(h, y_end)

    if args.width is not None and args.width < w:
        cx   = w // 2
        half = args.width // 2
        projs_raw = projections[:, y_start:y_end, cx - half : cx + half]
    else:
        projs_raw = projections[:, y_start:y_end, :]

    print(f'\nCrop → shape {projs_raw.shape}', flush=True)
    del projections

    # ── Normalize once; reuse for all configs ─────────────────────────────────
    projs = -projs_raw.copy()
    projs -= projs.min()
    projs /= projs.max()
    del projs_raw

    # ── Find rotation center once ─────────────────────────────────────────────
    print('\nCreating reference tomoData for rotation-center detection...', flush=True)
    _ref_tomo = tomoDataClass.tomoData(projs, angles)
    center = float(tomopy.find_center_vo(_ref_tomo.finalProjections))
    center_offset = center - _ref_tomo.image_size[1] / 2
    print(f'  Rotation center: {center:.2f}  center_offset: {center_offset:.2f}', flush=True)
    del _ref_tomo

    print('\n' + '=' * 80, flush=True)
    print(f'Starting {len(configs)} reconstructions  (parallel workers: {num_workers})', flush=True)
    print('=' * 80, flush=True)

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = [
        dict(
            projs=projs,
            angles=angles,
            center=center,
            center_offset=center_offset,
            cfg=cfg,
            recons_dir=recons_dir,
            slices_dir=slices_dir,
            timestamp=timestamp,
            scale_info=scale_info,
            do_fsc=args.fsc,
            num_workers=num_workers,
            total_cpus=total_cpus,
        )
        for cfg in configs
    ]

    # ── Run (parallel or serial) ──────────────────────────────────────────────
    if num_workers > 1:
        print(f'Parallelizing across {num_workers} workers '
              f'({max(1, total_cpus // num_workers)} OMP threads each)', flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(run_single_config, tasks))
    else:
        results = [run_single_config(t) for t in tasks]

    # ── Final summary table ───────────────────────────────────────────────────
    print('\n' + '=' * 80, flush=True)
    print('FINAL RESULTS (sorted by RCS, ascending = better)', flush=True)
    print('=' * 80, flush=True)

    sorted_results = sorted(results, key=lambda r: (r['rcs'] if r['rcs'] is not None else 1e9))

    sep2, widths2, _ = print_table_header(args.fsc)
    for r in sorted_results:
        print_table_row(r['cfg'], r['elapsed_min'], r['rcs'], r['fsc_res'], sep2, widths2, args.fsc)

    print('\nRanking (lower RCS = better fit to data):', flush=True)
    for rank, r in enumerate(sorted_results, 1):
        rcs_str = f'{r["rcs"]:.4f}' if r['rcs'] is not None else 'N/A'
        fsc_str = f'{r["fsc_res"]:.4f} px' if r['fsc_res'] is not None else ''
        extra = f'  FSC(0.143)={fsc_str}' if fsc_str else ''
        print(f'  {rank:2d}. {r["cfg"]["name"]:<20}  RCS={rcs_str}{extra}', flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(sorted_results, csv_path, args.fsc)
    print(f'\nCSV saved to: {csv_path}', flush=True)

    print(f'\nTotal wall time: {(time.time()-t_total)/60:.1f} min', flush=True)
    print(f'Log saved to: {log_path}', flush=True)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tiff-file',    type=str, required=False, default=None,
                        help='Path to aligned projections TIFF')
    parser.add_argument('--y-start',      type=int, default=None)
    parser.add_argument('--y-end',        type=int, default=None)
    parser.add_argument('--width',        type=int, default=None)
    parser.add_argument('--output-dir',   type=str, default=None,
                        help='Base dir; recons and slices subdirs are created inside it '
                             '(default: reconstructions/APSbeamtime_Oct25/svmbir)')
    parser.add_argument('--csv-file',     type=str, default=None,
                        help='CSV file to save results to (default: hyperparam_results/svmbir_search_<timestamp>.csv)')
    parser.add_argument('--configs',      type=str, default=None,
                        help='Comma-separated config names to run (default: all)')
    parser.add_argument('--fsc',          action='store_true',
                        help='Compute FSC resolution for each config (3x slower)')
    parser.add_argument('--list-configs', action='store_true',
                        help='Print all config names and exit')
    parser.add_argument('--num-workers',  type=int, default=1,
                        help='Number of configs to run in parallel (default: 1). '
                             'OMP threads per worker = SLURM_CPUS_PER_TASK / num_workers')
    args = parser.parse_args()

    if not args.list_configs and args.tiff_file is None:
        parser.error('--tiff-file is required unless --list-configs is used')

    main(args)
