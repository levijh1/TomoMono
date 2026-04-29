#!/usr/bin/env python3
"""
Hyperparameter search for tomographic alignment pipeline.

Searches over XCA, VMF, and PMA parameters using the real APS dataset.
Each config runs: XCA (multi-pass) → VMF (optional) → PMA (optional)
                  → make_updates_shift → SIRT_CUDA reconstruction → RCS score.

Results are continuously appended to a CSV log so partial runs are safe to
inspect or resume.  Run on a GPU node; make sure 'import torch' is uncommented
in tomoDataClass.py.

Usage:
    python hyperparameter_search.py
    python hyperparameter_search.py --resume          # skip already-done configs
    python hyperparameter_search.py --recon art       # CPU fallback for testing
"""

import os
import sys
import csv
import time
import h5py
import itertools
import traceback
import argparse
import numpy as np
from datetime import datetime

from tomoDataClass import tomoData
from alignment_methods import reprojection_consistency_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these before submitting to the cluster
# ══════════════════════════════════════════════════════════════════════════════

FILENAME = (
    "/Users/levihancock/Library/CloudStorage/Box-Box/"
    "BYU_CXI_Research_Team/ProjectFolders/IFE-STAR/IFE-Ptycho-Tomo/"
    "APS_2ID_GUP1013052_August_2025/levi_tomoReconstructions/"
    "tomo_data_run_final_2.hdf5"
)

DEFAULT_RECON_ALG = 'SIRT_CUDA'   # change to 'art' or 'gridrec' for CPU testing
LOG_FILE = f"hyperparam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRIDS
# ══════════════════════════════════════════════════════════════════════════════
# Each XCA config is a list of per-pass kwargs (coarse → fine).
# Fixed across all passes: tolerance=0, maxShiftTolerance=0, use_grad=True.

XCA_CONFIGS = {
    'xca_baseline': [
        dict(downsample=4, max_iterations=10, stepRatio=0.9),
        dict(downsample=2, max_iterations=10, stepRatio=0.9),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8),
    ],
    'xca_more_iters': [
        dict(downsample=4, max_iterations=15, stepRatio=0.9),
        dict(downsample=2, max_iterations=15, stepRatio=0.9),
        dict(downsample=1, max_iterations=20, stepRatio=0.8),
    ],
    'xca_fine_heavy': [
        dict(downsample=4, max_iterations=10, stepRatio=0.9),
        dict(downsample=2, max_iterations=10, stepRatio=0.9),
        dict(downsample=1, max_iterations=30, stepRatio=0.75),
    ],
    'xca_conservative': [
        dict(downsample=4, max_iterations=10, stepRatio=0.85),
        dict(downsample=2, max_iterations=10, stepRatio=0.85),
        dict(downsample=1, max_iterations=15, stepRatio=0.75),
    ],
}

# run=False → skip this stage entirely
VMF_CONFIGS = {
    'vmf_skip': dict(run=False),
    'vmf_soft_light': dict(
        run=True, max_iterations=3, smooth_sigma=1.0,
        window='soft_roi', roi_sigma=0.3,
    ),
    'vmf_soft_heavy': dict(
        run=True, max_iterations=5, smooth_sigma=1.0,
        window='soft_roi', roi_sigma=0.3,
    ),
    'vmf_soft_wide_sigma': dict(
        run=True, max_iterations=5, smooth_sigma=2.0,
        window='soft_roi', roi_sigma=0.4,
    ),
    'vmf_hanning': dict(
        run=True, max_iterations=5, smooth_sigma=2.0,
        window='hanning', roi_sigma=0.3,
    ),
}

PMA_CONFIGS = {
    'pma_skip': dict(run=False),
    'pma_of_2lev': dict(
        run=True, levels=2, scale=2,
        iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=3.0,
    ),
    'pma_of_2lev_sigma2': dict(
        run=True, levels=2, scale=2,
        iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=2.0,
    ),
    'pma_of_3lev': dict(
        run=True, levels=3, scale=2,
        iterations_per_level=[5, 5, 3],
        shift_method='optical_flow', of_sigma=3.0,
    ),
    'pma_of_3lev_more': dict(
        run=True, levels=3, scale=2,
        iterations_per_level=[10, 5, 3],
        shift_method='optical_flow', of_sigma=3.0,
    ),
    'pma_cc_2lev': dict(
        run=True, levels=2, scale=2,
        iterations_per_level=[5, 5],
        shift_method='cross_correlation', of_sigma=3.0,
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'config_id', 'xca_name', 'vmf_name', 'pma_name',
    'xca_passes',
    'vmf_run', 'vmf_max_iters', 'vmf_smooth_sigma', 'vmf_window', 'vmf_roi_sigma',
    'pma_run', 'pma_levels', 'pma_scale', 'pma_iters_per_level',
    'pma_shift_method', 'pma_of_sigma',
    'rcs', 'time_s', 'status',
]


def init_csv(path):
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(path, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_done_ids(path):
    """Return set of config_ids already successfully completed in an existing log."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'ok':
                done.add(int(row['config_id']))
    return done


def build_row(config_id, xca_name, vmf_name, pma_name,
              xca_passes, vmf_cfg, pma_cfg, rcs, elapsed, status):
    return dict(
        config_id=config_id,
        xca_name=xca_name,
        vmf_name=vmf_name,
        pma_name=pma_name,
        xca_passes=str(xca_passes),
        vmf_run=vmf_cfg.get('run', False),
        vmf_max_iters=vmf_cfg.get('max_iterations', ''),
        vmf_smooth_sigma=vmf_cfg.get('smooth_sigma', ''),
        vmf_window=vmf_cfg.get('window', ''),
        vmf_roi_sigma=vmf_cfg.get('roi_sigma', ''),
        pma_run=pma_cfg.get('run', False),
        pma_levels=pma_cfg.get('levels', ''),
        pma_scale=pma_cfg.get('scale', ''),
        pma_iters_per_level=str(pma_cfg.get('iterations_per_level', '')),
        pma_shift_method=pma_cfg.get('shift_method', ''),
        pma_of_sigma=pma_cfg.get('of_sigma', ''),
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

def run_pipeline(projections, angles, xca_passes, vmf_cfg, pma_cfg, recon_alg):
    """
    Run one complete alignment + reconstruction pipeline and return the RCS score.
    Creates a fresh tomoData object each time so configs don't bleed into each other.
    """
    t = tomoData(projections, angles)
    t.reset_workingProjections(x_size=None, y_size=None)
    t.normalize(isPhaseData=True)

    # ── Cross-correlation alignment (coarse → fine multi-pass) ────────────────
    for pass_cfg in xca_passes:
        t.cross_correlate_align(
            tolerance=0, maxShiftTolerance=0,
            yROI_Range=None, xROI_Range=None,
            isFull360=False, use_grad=True,
            **pass_cfg,
        )

    # ── Vertical mass fluctuation alignment (optional) ────────────────────────
    if vmf_cfg.get('run'):
        t.vertical_mass_fluctuation_align(
            tolerance=0, y_range=None, sigma=None,
            use_gradient=True, plot=False,
            max_iterations=vmf_cfg['max_iterations'],
            smooth_sigma=vmf_cfg['smooth_sigma'],
            window=vmf_cfg['window'],
            roi_sigma=vmf_cfg['roi_sigma'],
        )

    # ── Projection matching alignment (optional) ──────────────────────────────
    if pma_cfg.get('run'):
        t.PMA(
            tolerance=0, algorithm=recon_alg, plot=False,
            levels=pma_cfg['levels'],
            scale=pma_cfg['scale'],
            iterations_per_level=pma_cfg['iterations_per_level'],
            shift_method=pma_cfg['shift_method'],
            of_sigma=pma_cfg['of_sigma'],
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
            print("  CuPy GPU : available — array ops will run on GPU")
    except ImportError:
        pass
    print(f"  Recon alg: {recon_alg}")
    print(f"  Log file : {log_path}")
    print()

    # ── Data loading + downsampling ───────────────────────────────────────────
    print(f"Loading {FILENAME} ...")
    projections_og, angles_og = load_data(FILENAME)
    print(f"  Original shape : {projections_og.shape}")

    projections = projections_og[::2, ::4, ::4]   # same as notebook
    angles      = angles_og[::2]
    print(f"  Downsampled to : {projections.shape}  ({len(angles)} angles)\n")

    # ── Build full config list ─────────────────────────────────────────────────
    all_combos = list(itertools.product(
        XCA_CONFIGS.items(),
        VMF_CONFIGS.items(),
        PMA_CONFIGS.items(),
    ))
    n_search = len(all_combos)
    n_total  = n_search + 1   # +1 for the no-alignment baseline

    # ── Init or resume CSV ────────────────────────────────────────────────────
    if args.resume and os.path.exists(log_path):
        done_ids = load_done_ids(log_path)
        print(f"Resuming: {len(done_ids)}/{n_total} already done, "
              f"{n_total - len(done_ids)} remaining.\n")
    else:
        init_csv(log_path)
        done_ids = set()
        print(f"Starting fresh search over {n_total} configs "
              f"({len(XCA_CONFIGS)} XCA × {len(VMF_CONFIGS)} VMF × {len(PMA_CONFIGS)} PMA + 1 baseline).\n")

    # ── Baseline: no alignment ────────────────────────────────────────────────
    BASELINE_ID = -1
    NO_CFG      = dict(run=False)

    if BASELINE_ID not in done_ids:
        print("━" * 70)
        print("  BASELINE  (no alignment)")
        print("━" * 70)
        t0 = time.time()
        try:
            tb = tomoData(projections, angles)
            tb.reset_workingProjections(x_size=None, y_size=None)
            tb.normalize(isPhaseData=True)
            tb.make_updates_shift()
            tb.reconstruct(algorithm=recon_alg)
            rcs_base, _, _ = reprojection_consistency_score(tb, plot=False)
            del tb
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs_base = float('nan')
            status = f'error: {str(e)[:100]}'

        elapsed = time.time() - t0
        append_row(log_path, build_row(
            BASELINE_ID, 'none', 'none', 'none',
            [], NO_CFG, NO_CFG, rcs_base, elapsed, status,
        ))
        print(f"  → RCS = {rcs_base:.6f}  |  time = {elapsed:.0f}s  |  {status}\n")
    else:
        print("  Baseline already done — skipping.\n")

    # ── Parameter search ──────────────────────────────────────────────────────
    for cfg_id, ((xca_name, xca_passes), (vmf_name, vmf_cfg), (pma_name, pma_cfg)) \
            in enumerate(all_combos):

        if cfg_id in done_ids:
            print(f"  [{cfg_id+1}/{n_search}] skipping {xca_name}+{vmf_name}+{pma_name} (done)")
            continue

        print("━" * 70)
        print(f"  [{cfg_id+1}/{n_search}]  {xca_name}  +  {vmf_name}  +  {pma_name}")
        print("━" * 70)

        t0 = time.time()
        try:
            rcs = run_pipeline(
                projections, angles, xca_passes, vmf_cfg, pma_cfg, recon_alg,
            )
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs = float('nan')
            status = f'error: {str(e)[:100]}'

        elapsed = time.time() - t0
        append_row(log_path, build_row(
            cfg_id, xca_name, vmf_name, pma_name,
            xca_passes, vmf_cfg, pma_cfg, rcs, elapsed, status,
        ))
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
            ['config_id', 'xca_name', 'vmf_name', 'pma_name', 'rcs', 'time_s']
        ].to_string())
    except ImportError:
        print("(pandas not available — open the CSV directly for the full table.)")

    print(f"\nFull results saved to: {log_path}")


if __name__ == '__main__':
    main()
