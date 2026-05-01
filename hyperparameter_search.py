#!/usr/bin/env python3
"""
Hyperparameter search for tomographic alignment pipeline.

Pipeline: reset → normalize → XCA (varied) → PMA (varied)
          → make_updates_shift → SIRT_CUDA reconstruction → RCS score.

Varied axes:
  XCA: multi-pass strategy, use_grad, stepRatio
  PMA: stepRatio, of_sigma, levels, iterations_per_level

All (XCA config × PMA config) combinations are evaluated.  Results are
continuously appended to a CSV log so partial runs are safe to inspect
or resume.  Run on a GPU node.

Usage:
    python hyperparameter_search.py
    python hyperparameter_search.py --resume --logfile <path>   # resume a timed-out run
    python hyperparameter_search.py --recon art                 # CPU fallback for testing
"""

import os
import csv
import time
import h5py
import traceback
import argparse
import numpy as np
from datetime import datetime

from tomoDataClass import tomoData
from alignment_methods import reprojection_consistency_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILENAME = "/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5"

DEFAULT_RECON_ALG  = 'SIRT_CUDA'
LOG_FILE           = f"hyperparam_results/xca_pma_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DOWNSAMPLE_SPATIAL = 4   # matches notebook: zoom(..., 1/4, 1/4)

# ══════════════════════════════════════════════════════════════════════════════
# XCA CONFIGS  (each entry is a list of per-pass kwargs)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed for all passes: tolerance=0, maxShiftTolerance=0,
#                       yROI_Range=None, xROI_Range=None, isFull360=False.

XCA_CONFIGS = {
    # 3-pass, gradient mode — best strategy from previous sweep
    'xca_3p_grad': [
        dict(downsample=4, max_iterations=10, stepRatio=0.9, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.9, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8, use_grad=True),
    ],
    # 3-pass, intensity mode (no gradient preprocessing)
    'xca_3p_nograd': [
        dict(downsample=4, max_iterations=10, stepRatio=0.9, use_grad=False),
        dict(downsample=2, max_iterations=10, stepRatio=0.9, use_grad=False),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8, use_grad=False),
    ],
    # 3-pass, gradient, conservative stepRatio throughout
    'xca_3p_grad_sr08': [
        dict(downsample=4, max_iterations=10, stepRatio=0.8, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.8, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8, use_grad=True),
    ],
    # 2-pass, gradient (skip ds=2 middle pass — faster, less coarse-to-fine)
    'xca_2p_grad': [
        dict(downsample=4, max_iterations=10, stepRatio=0.9, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8, use_grad=True),
    ],
    # 3-pass, gradient, stepRatio=0.6
    'xca_3p_grad_sr06': [
        dict(downsample=4, max_iterations=10, stepRatio=0.6, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.6, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.6, use_grad=True),
    ],
    # 3-pass, gradient, stepRatio=0.7
    'xca_3p_grad_sr07': [
        dict(downsample=4, max_iterations=10, stepRatio=0.7, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.7, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.7, use_grad=True),
    ],
    # 4-pass, gradient, stepRatio=0.8 (adds ds=8 coarse pass)
    'xca_4p_grad_sr08': [
        dict(downsample=8, max_iterations=10, stepRatio=0.8, use_grad=True),
        dict(downsample=4, max_iterations=10, stepRatio=0.8, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.8, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8, use_grad=True),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# PMA PARAMETER GRID
# ══════════════════════════════════════════════════════════════════════════════

PMA_CONFIGS = {
    'pma_skip': dict(run=False),

    # ── 3-level ───────────────────────────────────────────────────────────────
    'pma_3lev_sr09_sig3': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 2],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.9,
    ),
    'pma_3lev_sr08_sig3': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 2],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.8,
    ),
    'pma_3lev_sr09_sig2': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 2],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.9,
    ),
    'pma_3lev_sr09_sig3_heavy': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 5],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.9,
    ),

    # ── 2-level ───────────────────────────────────────────────────────────────
    'pma_2lev_sr09_sig1': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=1.0, stepRatio=0.9,
    ),
    'pma_2lev_sr08_sig1': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=1.0, stepRatio=0.8,
    ),
    'pma_2lev_sr09_sig2': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.9,
    ),
    'pma_2lev_sr08_sig2': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    'pma_2lev_sr09_sig3': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.9,
    ),
    'pma_2lev_sr08_sig3': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.8,
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'config_id',
    'xca_name', 'xca_n_passes',
    'pma_name',
    'pma_run', 'pma_levels', 'pma_scale', 'pma_iters_per_level',
    'pma_of_sigma', 'pma_step_ratio',
    'rcs', 'time_s', 'status',
]


def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(path, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'ok':
                done.add(int(row['config_id']))
    return done


def build_row(config_id, xca_name, xca_passes, pma_name, pma_cfg, rcs, elapsed, status):
    return dict(
        config_id=config_id,
        xca_name=xca_name,
        xca_n_passes=len(xca_passes),
        pma_name=pma_name,
        pma_run=pma_cfg.get('run', False),
        pma_levels=pma_cfg.get('levels', ''),
        pma_scale=pma_cfg.get('scale', ''),
        pma_iters_per_level=str(pma_cfg.get('iterations_per_level', '')),
        pma_of_sigma=pma_cfg.get('of_sigma', ''),
        pma_step_ratio=pma_cfg.get('stepRatio', ''),
        rcs=f'{rcs:.6f}' if not np.isnan(rcs) else 'nan',
        time_s=f'{elapsed:.1f}',
        status=status,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filename):
    with h5py.File(filename) as hf:
        projs  = hf['data'][...]
        angles = hf['angles'][...] * np.pi / 180
    return projs, angles


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(projections, angles, xca_passes, pma_cfg, recon_alg):
    """
    Run one complete alignment + reconstruction pipeline and return the RCS score.
    Creates a fresh tomoData object each time so configs don't bleed into each other.

    Crop timing matches the notebook: run the first (coarsest) XCA pass on the
    full-width data, commit those shifts, crop, then run the remaining passes.
    """
    t = tomoData(projections, angles)

    # No crop yet — first pass sees the full-width projections
    t.reset_workingProjections(x_size=None, y_size=None)
    t.normalize(isPhaseData=True)

    # First XCA pass (coarsest downsample) on full-width data
    t.cross_correlate_align(
        tolerance=0, maxShiftTolerance=0,
        yROI_Range=None, xROI_Range=None,
        isFull360=False,
        **xca_passes[0],
    )

    # Commit shifts then crop — matching notebook timing
    t.make_updates_shift()
    x_size = projections.shape[2] - (500 // DOWNSAMPLE_SPATIAL)
    t.crop_center(new_x=x_size, new_y=None)

    # Remaining XCA passes on the cropped data
    for pass_cfg in xca_passes[1:]:
        t.cross_correlate_align(
            tolerance=0, maxShiftTolerance=0,
            yROI_Range=None, xROI_Range=None,
            isFull360=False,
            **pass_cfg,
        )

    # PMA (optional)
    if pma_cfg.get('run'):
        t.PMA(
            tolerance=0, algorithm=recon_alg, plot=False,
            levels=pma_cfg['levels'],
            scale=pma_cfg['scale'],
            iterations_per_level=pma_cfg['iterations_per_level'],
            shift_method=pma_cfg['shift_method'],
            of_sigma=pma_cfg['of_sigma'],
            stepRatio=pma_cfg['stepRatio'],
        )

    t.make_updates_shift()
    t.reconstruct(algorithm=recon_alg)
    rcs, _, _ = reprojection_consistency_score(t, plot=False)
    return rcs, t.recon


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',  action='store_true',
                        help='Skip configs already marked ok in the log file')
    parser.add_argument('--recon',   default=DEFAULT_RECON_ALG,
                        help='Reconstruction algorithm (default: SIRT_CUDA)')
    parser.add_argument('--logfile', default=LOG_FILE,
                        help='CSV output path')
    args = parser.parse_args()

    recon_alg = args.recon
    log_path  = args.logfile

    print("=" * 70)
    print("TOMOGRAPHIC ALIGNMENT HYPERPARAMETER SEARCH")
    print("=" * 70)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA GPU : {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: No CUDA GPU detected. "
                  "SIRT_CUDA will fail — pass --recon art for CPU testing.")
    except ImportError:
        print("  WARNING: torch not importable.")
    try:
        import cupy as cp
        if cp.is_available():
            print("  CuPy GPU : available")
    except ImportError:
        pass
    print(f"  Recon alg: {recon_alg}")
    print(f"  Log file : {log_path}")
    print()

    # Data loading + downsampling
    print(f"Loading {FILENAME} ...")
    projections_og, angles_og = load_data(FILENAME)
    print(f"  Original shape : {projections_og.shape}")

    from scipy.ndimage import zoom
    projections = zoom(projections_og, (1, 1 / DOWNSAMPLE_SPATIAL, 1 / DOWNSAMPLE_SPATIAL), order=1)
    angles      = angles_og
    print(f"  Downsampled to : {projections.shape}  ({len(angles)} angles)\n")

    # Explicit experiment list — (config_id, xca_name, pma_name)
    EXPERIMENTS = [
        # === xca_3p_grad_sr06 ===
        (43, 'xca_3p_grad_sr06', 'pma_skip'),
        (44, 'xca_3p_grad_sr06', 'pma_2lev_sr09_sig1'),
        (45, 'xca_3p_grad_sr06', 'pma_2lev_sr08_sig1'),
        (46, 'xca_3p_grad_sr06', 'pma_2lev_sr09_sig2'),
        (47, 'xca_3p_grad_sr06', 'pma_2lev_sr08_sig2'),
        (48, 'xca_3p_grad_sr06', 'pma_2lev_sr09_sig3'),
        (49, 'xca_3p_grad_sr06', 'pma_2lev_sr08_sig3'),
        # === xca_3p_grad_sr07 ===
        (50, 'xca_3p_grad_sr07', 'pma_skip'),
        (51, 'xca_3p_grad_sr07', 'pma_2lev_sr09_sig1'),
        (52, 'xca_3p_grad_sr07', 'pma_2lev_sr08_sig1'),
        (53, 'xca_3p_grad_sr07', 'pma_2lev_sr09_sig2'),
        (54, 'xca_3p_grad_sr07', 'pma_2lev_sr08_sig2'),
        (55, 'xca_3p_grad_sr07', 'pma_2lev_sr09_sig3'),
        (56, 'xca_3p_grad_sr07', 'pma_2lev_sr08_sig3'),
        # === xca_3p_grad_sr08 (3 new configs — 4 already done) ===
        (57, 'xca_3p_grad_sr08', 'pma_2lev_sr09_sig1'),
        (58, 'xca_3p_grad_sr08', 'pma_2lev_sr08_sig1'),
        (59, 'xca_3p_grad_sr08', 'pma_2lev_sr08_sig2'),
        # === xca_4p_grad_sr08 ===
        (60, 'xca_4p_grad_sr08', 'pma_skip'),
        (61, 'xca_4p_grad_sr08', 'pma_2lev_sr09_sig1'),
        (62, 'xca_4p_grad_sr08', 'pma_2lev_sr08_sig1'),
        (63, 'xca_4p_grad_sr08', 'pma_2lev_sr09_sig2'),
        (64, 'xca_4p_grad_sr08', 'pma_2lev_sr08_sig2'),
        (65, 'xca_4p_grad_sr08', 'pma_2lev_sr09_sig3'),
        (66, 'xca_4p_grad_sr08', 'pma_2lev_sr08_sig3'),
    ]

    all_configs = [
        (cfg_id, xca_name, XCA_CONFIGS[xca_name], pma_name, PMA_CONFIGS[pma_name])
        for cfg_id, xca_name, pma_name in EXPERIMENTS
    ]
    n_total = len(all_configs)

    print(f"Running {n_total} explicit configs (IDs {all_configs[0][0]}–{all_configs[-1][0]})")
    print()

    # Init or resume CSV
    if args.resume and os.path.exists(log_path):
        done_ids = load_done_ids(log_path)
        print(f"Resuming: {len(done_ids)}/{n_total} already done, "
              f"{n_total - len(done_ids)} remaining.\n")
    else:
        init_csv(log_path)
        done_ids = set()
        print(f"Starting fresh search over {n_total} configs.\n")

    best_rcs   = float('inf')
    best_recon = None
    best_cfg_id = None

    # Main loop
    for i, (cfg_id, xca_name, xca_passes, pma_name, pma_cfg) in enumerate(all_configs):

        if cfg_id in done_ids:
            print(f"  [{i+1}/{n_total}] skipping {xca_name} + {pma_name} (done)")
            continue

        print("━" * 70)
        print(f"  [{i+1}/{n_total}]  cfg_id={cfg_id}  {xca_name}  +  {pma_name}")
        print("━" * 70)

        t0 = time.time()
        try:
            rcs, recon = run_pipeline(projections, angles, xca_passes, pma_cfg, recon_alg)
            if rcs < best_rcs:
                best_rcs    = rcs
                best_recon  = recon
                best_cfg_id = cfg_id
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs = float('nan')
            status = f'error: {str(e)[:100]}'

        elapsed = time.time() - t0
        append_row(log_path, build_row(cfg_id, xca_name, xca_passes, pma_name, pma_cfg,
                                       rcs, elapsed, status))
        print(f"\n  → RCS = {rcs:.6f}  |  time = {elapsed:.0f}s  |  {status}\n")

    # Save best reconstruction
    if best_recon is not None:
        tiff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paramSearchBestRecon.tiff')
        tifffile.imwrite(tiff_path, best_recon.astype(np.float32))
        print(f"Best recon (cfg_id={best_cfg_id}, RCS={best_rcs:.6f}) saved to: {tiff_path}\n")
    else:
        print("No successful runs — best recon not saved.\n")

    # Final summary
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    try:
        import pandas as pd
        df = pd.read_csv(log_path)
        df_ok = df[df['status'] == 'ok'].copy()
        df_ok['rcs'] = pd.to_numeric(df_ok['rcs'], errors='coerce')
        df_sorted = df_ok.sort_values('rcs').reset_index(drop=True)
        df_sorted.index.name = 'rank'
        print(df_sorted[
            ['config_id', 'xca_name', 'pma_name',
             'pma_levels', 'pma_iters_per_level',
             'pma_of_sigma', 'pma_step_ratio', 'rcs', 'time_s']
        ].to_string())
    except ImportError:
        print("(pandas not available — open the CSV directly for the full table.)")

    print(f"\nFull results saved to: {log_path}")


if __name__ == '__main__':
    main()
