#!/usr/bin/env python3
"""
Hyperparameter search for tomographic alignment pipeline.

Pipeline (fixed): reset → normalize → XCA (3-pass best params) → PMA (varied)
                  → make_updates_shift → SIRT_CUDA reconstruction → RCS score.

Varied: PMA stepRatio, of_sigma, levels, iterations_per_level.

Results are continuously appended to a CSV log so partial runs are safe to
inspect or resume.  Run on a GPU node.

Usage:
    python hyperparameter_search.py
    python hyperparameter_search.py --resume          # skip already-done configs
    python hyperparameter_search.py --recon art       # CPU fallback for testing
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
LOG_FILE           = f"hyperparam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DOWNSAMPLE_SPATIAL = 4   # matches notebook: zoom(..., 1/4, 1/4)

# ══════════════════════════════════════════════════════════════════════════════
# FIXED XCA PASSES  (best params from previous sweep)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed across all passes: tolerance=0, maxShiftTolerance=0,
#                          yROI_Range=None, xROI_Range=None,
#                          isFull360=False, use_grad=True.

XCA_PASSES = [
    dict(downsample=4, max_iterations=10, stepRatio=0.9),
    dict(downsample=2, max_iterations=10, stepRatio=0.9),
    dict(downsample=1, max_iterations=5,  stepRatio=0.8),
]

# ══════════════════════════════════════════════════════════════════════════════
# PMA PARAMETER GRID
# ══════════════════════════════════════════════════════════════════════════════
# Varied axes: stepRatio, of_sigma, levels, iterations_per_level.
# scale=4 matches the notebook example.
# Kept short per config to avoid cluster wall-time issues.

PMA_CONFIGS = {
    'pma_skip': dict(run=False),

    # ── 3-level (matches notebook format) ─────────────────────────────────────
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

    # ── 2-level (faster; useful as an upper bound check) ──────────────────────
    'pma_2lev_sr09_sig3': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.9,
    ),
    'pma_2lev_sr08_sig3': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.8,
    ),
    'pma_2lev_sr09_sig2': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.9,
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'config_id', 'pma_name',
    'pma_run', 'pma_levels', 'pma_scale', 'pma_iters_per_level',
    'pma_of_sigma', 'pma_step_ratio',
    'rcs', 'time_s', 'status',
]


def init_csv(path):
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


def build_row(config_id, pma_name, pma_cfg, rcs, elapsed, status):
    return dict(
        config_id=config_id,
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

def run_pipeline(projections, angles, pma_cfg, recon_alg):
    """
    Run one complete alignment + reconstruction pipeline and return the RCS score.
    Creates a fresh tomoData object each time so configs don't bleed into each other.
    """
    t = tomoData(projections, angles)

    x_crop = 500 // DOWNSAMPLE_SPATIAL   # matches notebook: shape[2] - (500//downsample)
    t.reset_workingProjections(
        x_size=projections.shape[2] - x_crop,
        y_size=projections.shape[1],
    )
    t.normalize(isPhaseData=True)

    # ── Fixed XCA: best params from sweep ────────────────────────────────────
    for pass_cfg in XCA_PASSES:
        t.cross_correlate_align(
            tolerance=0, maxShiftTolerance=0,
            yROI_Range=None, xROI_Range=None,
            isFull360=False, use_grad=True,
            **pass_cfg,
        )

    # ── PMA (optional) ────────────────────────────────────────────────────────
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

    # ── Apply accumulated shifts, reconstruct, score ──────────────────────────
    t.make_updates_shift()
    t.reconstruct(algorithm=recon_alg)
    rcs, _, _ = reprojection_consistency_score(t, plot=False)
    return rcs


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

    # ── GPU / environment check ───────────────────────────────────────────────
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
        print("  WARNING: torch not importable. "
              "Uncomment 'import torch' in tomoDataClass.py, or use --recon art.")
    try:
        import cupy as cp
        if cp.is_available():
            print("  CuPy GPU : available")
    except ImportError:
        pass
    print(f"  Recon alg: {recon_alg}")
    print(f"  Log file : {log_path}")
    print()

    # ── Data loading + downsampling ───────────────────────────────────────────
    print(f"Loading {FILENAME} ...")
    projections_og, angles_og = load_data(FILENAME)
    print(f"  Original shape : {projections_og.shape}")

    from scipy.ndimage import zoom
    projections = zoom(projections_og, (1, 1 / DOWNSAMPLE_SPATIAL, 1 / DOWNSAMPLE_SPATIAL), order=1)
    angles      = angles_og
    print(f"  Downsampled to : {projections.shape}  ({len(angles)} angles)\n")

    pma_items  = list(PMA_CONFIGS.items())
    n_configs  = len(pma_items)
    n_total    = n_configs + 1   # +1 for no-alignment baseline

    # ── Init or resume CSV ────────────────────────────────────────────────────
    if args.resume and os.path.exists(log_path):
        done_ids = load_done_ids(log_path)
        print(f"Resuming: {len(done_ids)}/{n_total} already done, "
              f"{n_total - len(done_ids)} remaining.\n")
    else:
        init_csv(log_path)
        done_ids = set()
        print(f"Starting fresh search over {n_total} configs "
              f"({n_configs} PMA variants + 1 baseline).\n")

    # ── Baseline: XCA only, no PMA ────────────────────────────────────────────
    BASELINE_ID = -1
    NO_CFG      = dict(run=False)

    if BASELINE_ID not in done_ids:
        print("━" * 70)
        print("  BASELINE  (XCA only, no PMA)")
        print("━" * 70)
        t0 = time.time()
        try:
            rcs_base = run_pipeline(projections, angles, NO_CFG, recon_alg)
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs_base = float('nan')
            status = f'error: {str(e)[:100]}'

        elapsed = time.time() - t0
        append_row(log_path, build_row(BASELINE_ID, 'none', NO_CFG, rcs_base, elapsed, status))
        print(f"  → RCS = {rcs_base:.6f}  |  time = {elapsed:.0f}s  |  {status}\n")
    else:
        print("  Baseline already done — skipping.\n")

    # ── PMA parameter search ──────────────────────────────────────────────────
    for cfg_id, (pma_name, pma_cfg) in enumerate(pma_items):

        if cfg_id in done_ids:
            print(f"  [{cfg_id+1}/{n_configs}] skipping {pma_name} (done)")
            continue

        print("━" * 70)
        print(f"  [{cfg_id+1}/{n_configs}]  {pma_name}")
        print("━" * 70)

        t0 = time.time()
        try:
            rcs = run_pipeline(projections, angles, pma_cfg, recon_alg)
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs = float('nan')
            status = f'error: {str(e)[:100]}'

        elapsed = time.time() - t0
        append_row(log_path, build_row(cfg_id, pma_name, pma_cfg, rcs, elapsed, status))
        print(f"\n  → RCS = {rcs:.6f}  |  time = {elapsed:.0f}s  |  {status}\n")

    # ── Final summary ─────────────────────────────────────────────────────────
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
            ['config_id', 'pma_name', 'pma_levels', 'pma_iters_per_level',
             'pma_of_sigma', 'pma_step_ratio', 'rcs', 'time_s']
        ].to_string())
    except ImportError:
        print("(pandas not available — open the CSV directly for the full table.)")

    print(f"\nFull results saved to: {log_path}")


if __name__ == '__main__':
    main()
