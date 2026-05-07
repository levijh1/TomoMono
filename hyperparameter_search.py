#!/usr/bin/env python3
"""
Hyperparameter search for tomographic alignment pipeline.

Pipeline: reset → normalize → XCA (varied) → PMA (varied)
          → make_updates_shift → SIRT_CUDA reconstruction → RCS score.

Varied axes:
  XCA: multi-pass strategy, use_grad, stepRatio, ROI (lowerboundCropPercentage)
  PMA: stepRatio, of_sigma, levels, iterations_per_level,
       use_matching_preprocess, use_grad, ROI

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
import shutil
import h5py
import traceback
import argparse
import numpy as np
from datetime import datetime
import tifffile

from tomoDataClass import tomoData
from alignment_methods import reprojection_consistency_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILENAME = "/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5"

DEFAULT_RECON_ALG       = 'SIRT_CUDA'
DEFAULT_DOWNSAMPLE      = 4   # matches notebook: zoom(..., 1/4, 1/4)

# ══════════════════════════════════════════════════════════════════════════════
# XCA CONFIGS  (each entry is a list of per-pass kwargs)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed for all passes: tolerance=0, maxShiftTolerance=0, isFull360=False.
# ROI for passes 2+ is controlled by XCA_ROI_PARAMS (below), NOT here.
# Pass 1 always uses full frame (no ROI) — matching the notebook pattern.

def _xca_passes_4p_sr08(use_grad=True):
    return [
        dict(downsample=8, max_iterations=10, stepRatio=0.8,  use_grad=use_grad),
        dict(downsample=4, max_iterations=10, stepRatio=0.8,  use_grad=use_grad),
        dict(downsample=2, max_iterations=10, stepRatio=0.8,  use_grad=use_grad),
        dict(downsample=1, max_iterations=5,  stepRatio=0.8,  use_grad=use_grad),
    ]

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
    'xca_4p_grad_sr08': _xca_passes_4p_sr08(use_grad=True),
    # 4-pass, no gradient, stepRatio=0.8  ← new (133-135)
    'xca_4p_nograd_sr08': _xca_passes_4p_sr08(use_grad=False),
    # 4-pass, gradient, stepRatio=0.75
    'xca_4p_grad_sr075': [
        dict(downsample=8, max_iterations=10, stepRatio=0.75, use_grad=True),
        dict(downsample=4, max_iterations=10, stepRatio=0.75, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.75, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.75, use_grad=True),
    ],
    # 4-pass, gradient, stepRatio=0.82  ← new (118-120)
    'xca_4p_grad_sr082': [
        dict(downsample=8, max_iterations=10, stepRatio=0.82, use_grad=True),
        dict(downsample=4, max_iterations=10, stepRatio=0.82, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.82, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.82, use_grad=True),
    ],
    # 4-pass, gradient, stepRatio=0.83  ← new (121-123)
    'xca_4p_grad_sr083': [
        dict(downsample=8, max_iterations=10, stepRatio=0.83, use_grad=True),
        dict(downsample=4, max_iterations=10, stepRatio=0.83, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.83, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.83, use_grad=True),
    ],
    # 4-pass, gradient, stepRatio=0.85
    'xca_4p_grad_sr085': [
        dict(downsample=8, max_iterations=10, stepRatio=0.85, use_grad=True),
        dict(downsample=4, max_iterations=10, stepRatio=0.85, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.85, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.85, use_grad=True),
    ],
    # 3-pass, gradient, stepRatio=0.75
    'xca_3p_grad_sr075': [
        dict(downsample=4, max_iterations=10, stepRatio=0.75, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.75, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.75, use_grad=True),
    ],
    # 3-pass, gradient, stepRatio=0.85
    'xca_3p_grad_sr085': [
        dict(downsample=4, max_iterations=10, stepRatio=0.85, use_grad=True),
        dict(downsample=2, max_iterations=10, stepRatio=0.85, use_grad=True),
        dict(downsample=1, max_iterations=5,  stepRatio=0.85, use_grad=True),
    ],
    # ── ROI variants of xca_4p_grad_sr08 ─────────────────────────────────────
    # Same passes as xca_4p_grad_sr08.  ROI for passes 2+ is applied by
    # run_pipeline using XCA_ROI_PARAMS[xca_name] after the initial crop.
    'xca_4p_grad_sr08_roi_lb07':  _xca_passes_4p_sr08(use_grad=True),
    'xca_4p_grad_sr08_roi_lb08':  _xca_passes_4p_sr08(use_grad=True),
    'xca_4p_grad_sr08_roi_lb085': _xca_passes_4p_sr08(use_grad=True),
    'xca_4p_grad_sr08_roi_lb09':  _xca_passes_4p_sr08(use_grad=True),
    'xca_4p_grad_sr08_roi_lb095': _xca_passes_4p_sr08(use_grad=True),
}

# ══════════════════════════════════════════════════════════════════════════════
# XCA ROI PARAMS
# Maps XCA config names that use ROI to their lowerboundCropPercentage.
# Configs NOT listed here use no ROI (yROI_Range=None, xROI_Range=None).
# ROI formula matches the notebook:
#   edgeCropping = 200 // downsample
#   yROI = [int(0.05*H), int(lb*H)]
#   xROI = [edgeCropping + int(0.15*(W-2*edgeCropping)),
#           edgeCropping + int(0.85*(W-2*edgeCropping))]
# ══════════════════════════════════════════════════════════════════════════════

XCA_ROI_PARAMS = {
    'xca_4p_grad_sr08_roi_lb07':  {'lb': 0.70},
    'xca_4p_grad_sr08_roi_lb08':  {'lb': 0.80},
    'xca_4p_grad_sr08_roi_lb085': {'lb': 0.85},
    'xca_4p_grad_sr08_roi_lb09':  {'lb': 0.90},
    'xca_4p_grad_sr08_roi_lb095': {'lb': 0.95},
}

# ══════════════════════════════════════════════════════════════════════════════
# PMA PARAMETER GRID
# New fields vs. original schema:
#   use_matching_preprocess (bool, default True)  — highpass+normalize preprocessing
#   use_grad                (bool, default False) — gradient magnitude preprocessing
#   use_roi                 (bool, default False) — use ROI for shift estimation
#   lb                      (float)               — lowerboundCropPercentage for ROI
# ══════════════════════════════════════════════════════════════════════════════

PMA_CONFIGS = {
    'pma_skip': dict(run=False),

    # ── 3-level, scale=4 ─────────────────────────────────────────────────────
    'pma_3lev_sr09_sig3': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 2],
        shift_method='optical_flow', of_sigma=3.0, stepRatio=0.9,
    ),
    'pma_3lev_sr08_sig4': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.8,
    ),
    'pma_3lev_sr09_sig4': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
    ),
    'pma_3lev_sr08_sig3': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
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
    # sigma fine-tuning around 4.0, scale=4 ← new (124-125)
    'pma_3lev_sr09_sig35': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=3.5, stepRatio=0.9,
    ),
    'pma_3lev_sr09_sig45': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.5, stepRatio=0.9,
    ),
    # 4-level, scale=4 ← new (128)
    'pma_4lev_sr09_sig4': dict(
        run=True, levels=4, scale=4, iterations_per_level=[8, 5, 3, 2],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
    ),

    # ── 2-level, scale=4 ─────────────────────────────────────────────────────
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
    'pma_2lev_sr09_sig15': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=1.5, stepRatio=0.9,
    ),
    'pma_2lev_sr08_sig15': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=1.5, stepRatio=0.8,
    ),
    'pma_2lev_sr08_sig2_heavy': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 10],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    'pma_2lev_sr08_sig2_xheavy': dict(
        run=True, levels=2, scale=4, iterations_per_level=[5, 20],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    'pma_3lev_sr08_sig2': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    'pma_3lev_sr08_sig1': dict(
        run=True, levels=3, scale=4, iterations_per_level=[5, 5, 2],
        shift_method='optical_flow', of_sigma=1.0, stepRatio=0.8,
    ),

    # ── scale=2 pyramid ───────────────────────────────────────────────────────
    'pma_2lev_sc2_sr08_sig2': dict(
        run=True, levels=2, scale=2, iterations_per_level=[5, 5],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    'pma_3lev_sc2_sr08_sig2': dict(
        run=True, levels=3, scale=2, iterations_per_level=[5, 5, 5],
        shift_method='optical_flow', of_sigma=2.0, stepRatio=0.8,
    ),
    # scale=2, sigma=4 variants ← new (115-117, 120, 123, 129, 131-132)
    'pma_3lev_sc2_sr09_sig4': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
    ),
    'pma_4lev_sc2_sr09_sig4': dict(
        run=True, levels=4, scale=2, iterations_per_level=[8, 5, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
    ),
    'pma_2lev_sc2_sr09_sig4': dict(
        run=True, levels=2, scale=2, iterations_per_level=[8, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
    ),
    # sigma fine-tuning around 4.0, scale=2 ← new (126-127)
    'pma_3lev_sc2_sr09_sig35': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=3.5, stepRatio=0.9,
    ),
    'pma_3lev_sc2_sr09_sig45': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.5, stepRatio=0.9,
    ),
    # step_ratio=0.8 on scale=2 ← new (130)
    'pma_3lev_sc2_sr08_sig4': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.8,
    ),

    # ── PMA preprocessing ablation (use_matching_preprocess / use_grad) ───────
    # Baseline (preprocess=True, grad=False) is the default used by 100-132.
    # Testing the other 3 combinations on scale=4 and scale=2.

    # scale=4
    'pma_3lev_sr09_sig4_nopreprocess': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=False, use_grad=False,
    ),
    'pma_3lev_sr09_sig4_pmagrad': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=False, use_grad=True,
    ),
    'pma_3lev_sr09_sig4_preprocess_pmagrad': dict(
        run=True, levels=3, scale=4, iterations_per_level=[8, 5, 3],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=True, use_grad=True,
    ),
    # scale=2
    'pma_3lev_sc2_sr09_sig4_nopreprocess': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=False, use_grad=False,
    ),
    'pma_3lev_sc2_sr09_sig4_pmagrad': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=False, use_grad=True,
    ),
    'pma_3lev_sc2_sr09_sig4_preprocess_pmagrad': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_matching_preprocess=True, use_grad=True,
    ),

    # ── PMA ROI ablation ──────────────────────────────────────────────────────
    # scale=2 with use_roi=True and varying lb.
    # Baseline (no ROI) = pma_3lev_sc2_sr09_sig4.
    'pma_3lev_sc2_sr09_sig4_roi_lb07': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_roi=True, lb=0.70,
    ),
    'pma_3lev_sc2_sr09_sig4_roi_lb08': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_roi=True, lb=0.80,
    ),
    'pma_3lev_sc2_sr09_sig4_roi_lb085': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_roi=True, lb=0.85,
    ),
    'pma_3lev_sc2_sr09_sig4_roi_lb09': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_roi=True, lb=0.90,
    ),
    'pma_3lev_sc2_sr09_sig4_roi_lb095': dict(
        run=True, levels=3, scale=2, iterations_per_level=[8, 5, 5],
        shift_method='optical_flow', of_sigma=4.0, stepRatio=0.9,
        use_roi=True, lb=0.95,
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# New columns appended after the original 14 so old rows stay valid.
# Defaults for configs 100-114 (backfilled on resume):
#   xca_use_roi = False, xca_lb_crop_pct = ''
#   pma_use_matching_preprocess = True (when pma_run=True), else ''
#   pma_use_grad = False (when pma_run=True), else ''
#   pma_use_roi = False (when pma_run=True), else ''
#   pma_lb_crop_pct = ''
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'config_id',
    'xca_name', 'xca_n_passes',
    'xca_use_roi', 'xca_lb_crop_pct',
    'pma_name',
    'pma_run', 'pma_levels', 'pma_scale', 'pma_iters_per_level',
    'pma_of_sigma', 'pma_step_ratio',
    'pma_use_matching_preprocess', 'pma_use_grad',
    'pma_use_roi', 'pma_lb_crop_pct',
    'rcs', 'align_time_s', 'recon_time_s', 'status',
]


def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(path, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS,
                       extrasaction='ignore', restval='').writerow(row)


def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'ok':
                done.add(int(row['config_id']))
    return done


def build_row(config_id, xca_name, xca_passes, xca_roi_params,
              pma_name, pma_cfg, rcs, align_time, recon_time, status):
    pma_run = pma_cfg.get('run', False)
    return dict(
        config_id=config_id,
        xca_name=xca_name,
        xca_n_passes=len(xca_passes),
        xca_use_roi=xca_roi_params is not None,
        xca_lb_crop_pct=xca_roi_params['lb'] if xca_roi_params else '',
        pma_name=pma_name,
        pma_run=pma_run,
        pma_levels=pma_cfg.get('levels', ''),
        pma_scale=pma_cfg.get('scale', ''),
        pma_iters_per_level=str(pma_cfg.get('iterations_per_level', '')),
        pma_of_sigma=pma_cfg.get('of_sigma', ''),
        pma_step_ratio=pma_cfg.get('stepRatio', ''),
        pma_use_matching_preprocess=pma_cfg.get('use_matching_preprocess', True) if pma_run else '',
        pma_use_grad=pma_cfg.get('use_grad', False) if pma_run else '',
        pma_use_roi=pma_cfg.get('use_roi', False) if pma_run else '',
        pma_lb_crop_pct=pma_cfg.get('lb', '') if pma_run else '',
        rcs=f'{rcs:.6f}' if not np.isnan(rcs) else 'nan',
        align_time_s=f'{align_time:.1f}',
        recon_time_s=f'{recon_time:.1f}',
        status=status,
    )


def migrate_csv_schema(path):
    """
    Backfill new columns into an existing CSV that uses the old 14-column schema.
    Creates a .bak backup before modifying.  Safe to call on an already-migrated file.
    """
    NEW_FIELDS = [
        'xca_use_roi', 'xca_lb_crop_pct',
        'pma_use_matching_preprocess', 'pma_use_grad',
        'pma_use_roi', 'pma_lb_crop_pct',
    ]

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    if all(field in existing_fields for field in NEW_FIELDS):
        return  # already migrated

    print(f"  Migrating CSV schema: adding {NEW_FIELDS} with defaults for existing rows…")
    shutil.copy(path, path + '.bak')

    for row in rows:
        pma_run = str(row.get('pma_run', '')).strip().lower() in ('true', '1')
        row.setdefault('xca_use_roi',                'False')
        row.setdefault('xca_lb_crop_pct',            '')
        row.setdefault('pma_use_matching_preprocess', 'True'  if pma_run else '')
        row.setdefault('pma_use_grad',               'False' if pma_run else '')
        row.setdefault('pma_use_roi',                'False' if pma_run else '')
        row.setdefault('pma_lb_crop_pct',            '')

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Migration complete. Backup saved to {path}.bak")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filename):
    with h5py.File(filename) as hf:
        projs  = hf['data'][...]
        angles = hf['angles'][...] * np.pi / 180
    #Shift angles to be centered around 0
    angles = angles - np.mean(angles)
    return projs, angles


# ══════════════════════════════════════════════════════════════════════════════
# ROI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def compute_roi(tomo_obj, lb, downsample, edge_margin=200, x_margin=0.15):
    """
    Compute yROI and xROI from current working-projection dimensions.
    Matches the notebook formula exactly:
        edgeCropping = edge_margin // downsample
        yROI = [int(0.05*H), int(lb*H)]
        xROI = [edgeCropping + int(x_margin*(W-2*edgeCropping)),
                edgeCropping + int((1-x_margin)*(W-2*edgeCropping))]
    Returns two numpy arrays (yROI, xROI).
    """
    H = tomo_obj.workingProjections.shape[1]
    W = tomo_obj.workingProjections.shape[2]
    edge = edge_margin // downsample
    inner = W - 2 * edge
    y_roi = np.array([int(0.05 * H), int(lb * H)])
    x_roi = np.array([edge + int(x_margin * inner),
                      edge + int((1 - x_margin) * inner)])
    return y_roi, x_roi


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(projections, angles, xca_name, xca_passes, xca_roi_params,
                 pma_cfg, recon_alg, downsample):
    """
    Run one complete alignment + reconstruction pipeline and return the RCS score.
    Creates a fresh tomoData object each time so configs don't bleed into each other.

    XCA structure:
      Pass 1  — full frame, no ROI  (coarsest downsample)
      Passes 2+  — full frame; ROI applied if xca_roi_params is not None
      make_updates_shift  (commits all XCA shifts at the end)

    PMA:
      ROI applied if pma_cfg['use_roi'] is True (lb from pma_cfg['lb'])
      use_matching_preprocess and use_grad taken from pma_cfg (defaults: True, False)
    """
    t = tomoData(projections, angles)

    t0_align = time.time()

    t.reset_workingProjections(x_size=None, y_size=None)
    t.normalize(isPhaseData=True)

    # Pass 1 — full frame, no ROI
    t.cross_correlate_align(
        tolerance=0, maxShiftTolerance=0,
        yROI_Range=None, xROI_Range=None,
        isFull360=False,
        **xca_passes[0],
    )

    # Compute ROI for passes 2+
    xca_yROI = xca_xROI = None
    if xca_roi_params:
        xca_yROI, xca_xROI = compute_roi(t, xca_roi_params['lb'], downsample)
        print(f"  XCA ROI (lb={xca_roi_params['lb']}): "
              f"y={xca_yROI.tolist()}  x={xca_xROI.tolist()}")

    # Passes 2+
    for pass_cfg in xca_passes[1:]:
        t.cross_correlate_align(
            tolerance=0, maxShiftTolerance=0,
            yROI_Range=xca_yROI, xROI_Range=xca_xROI,
            isFull360=False,
            **pass_cfg,
        )

    # PMA (optional)
    if pma_cfg.get('run'):
        pma_yROI = pma_xROI = None
        if pma_cfg.get('use_roi'):
            pma_yROI, pma_xROI = compute_roi(t, pma_cfg['lb'], downsample)
            print(f"  PMA ROI (lb={pma_cfg['lb']}): "
                  f"y={pma_yROI.tolist()}  x={pma_xROI.tolist()}")

        t.PMA(
            tolerance=0, algorithm=recon_alg, plot=False,
            levels=pma_cfg['levels'],
            scale=pma_cfg['scale'],
            iterations_per_level=pma_cfg['iterations_per_level'],
            shift_method=pma_cfg['shift_method'],
            of_sigma=pma_cfg['of_sigma'],
            stepRatio=pma_cfg['stepRatio'],
            use_matching_preprocess=pma_cfg.get('use_matching_preprocess', True),
            use_grad=pma_cfg.get('use_grad', False),
            yROI_Range=pma_yROI,
            xROI_Range=pma_xROI,
        )

    t.make_updates_shift()
    align_time = time.time() - t0_align

    t0_recon = time.time()
    t.reconstruct(algorithm=recon_alg)
    rcs, _, _ = reprojection_consistency_score(t, plot=False)
    recon_time = time.time() - t0_recon

    return rcs, t.recon, align_time, recon_time


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',     action='store_true',
                        help='Skip configs already marked ok in the log file')
    parser.add_argument('--recon',      default=DEFAULT_RECON_ALG,
                        help='Reconstruction algorithm (default: SIRT_CUDA)')
    parser.add_argument('--downsample', type=int, default=DEFAULT_DOWNSAMPLE,
                        help='Spatial downsample factor applied to projections (default: 4)')
    parser.add_argument('--logfile',    default=None,
                        help='CSV output path (default: auto-generated with downsample factor in name)')
    args = parser.parse_args()

    recon_alg  = args.recon
    downsample = args.downsample
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = args.logfile or (
        f"hyperparam_results/xca_pma_search_ds{downsample}_"
        f"{timestamp}.csv"
    )

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
    projections = zoom(projections_og, (1, 1 / downsample, 1 / downsample), order=1) if downsample > 1 else projections_og
    angles      = angles_og
    print(f"  Downsample     : {downsample}x")
    print(f"  Working shape  : {projections.shape}  ({len(angles)} angles)\n")

    # Explicit experiment list — (config_id, xca_name, pma_name)
    EXPERIMENTS = [
        # ── original 2x search (100-114) ──────────────────────────────────────
        # xca_use_grad=True (all), pma_use_matching_preprocess=True (all), no ROI
        (100, 'xca_4p_grad_sr08',   'pma_skip'),
        (101, 'xca_4p_grad_sr08',   'pma_3lev_sr08_sig4'),
        (102, 'xca_4p_grad_sr08',   'pma_3lev_sr08_sig3'),
        (103, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig4'),
        (104, 'xca_4p_grad_sr08',   'pma_3lev_sr08_sig2'),
        (105, 'xca_4p_grad_sr075',  'pma_skip'),
        (106, 'xca_4p_grad_sr075',  'pma_3lev_sr08_sig4'),
        (107, 'xca_4p_grad_sr075',  'pma_3lev_sr08_sig3'),
        (108, 'xca_4p_grad_sr075',  'pma_3lev_sr09_sig4'),
        (109, 'xca_4p_grad_sr075',  'pma_3lev_sr08_sig2'),
        (110, 'xca_4p_grad_sr085',  'pma_skip'),
        (111, 'xca_4p_grad_sr085',  'pma_3lev_sr08_sig4'),
        (112, 'xca_4p_grad_sr085',  'pma_3lev_sr08_sig3'),
        (113, 'xca_4p_grad_sr085',  'pma_3lev_sr09_sig4'),
        (114, 'xca_4p_grad_sr085',  'pma_3lev_sr08_sig2'),

        # ── scale=2 pyramid (missing from 2x search) ──────────────────────────
        (115, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4'),
        (116, 'xca_4p_grad_sr08',   'pma_4lev_sc2_sr09_sig4'),
        (117, 'xca_4p_grad_sr08',   'pma_2lev_sc2_sr09_sig4'),

        # ── fine XCA stepRatio around 0.8 ─────────────────────────────────────
        (118, 'xca_4p_grad_sr082',  'pma_skip'),
        (119, 'xca_4p_grad_sr082',  'pma_3lev_sr09_sig4'),
        (120, 'xca_4p_grad_sr082',  'pma_3lev_sc2_sr09_sig4'),
        (121, 'xca_4p_grad_sr083',  'pma_skip'),
        (122, 'xca_4p_grad_sr083',  'pma_3lev_sr09_sig4'),
        (123, 'xca_4p_grad_sr083',  'pma_3lev_sc2_sr09_sig4'),

        # ── fine-tune PMA sigma around 4.0 ────────────────────────────────────
        (124, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig35'),
        (125, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig45'),
        (126, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig35'),
        (127, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig45'),

        # ── 4-level PMA ───────────────────────────────────────────────────────
        (128, 'xca_4p_grad_sr08',   'pma_4lev_sr09_sig4'),
        (129, 'xca_4p_grad_sr08',   'pma_4lev_sc2_sr09_sig4'),  # note: same cfg as 116

        # ── step_ratio=0.8 on scale=2 ─────────────────────────────────────────
        (130, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr08_sig4'),

        # ── sr075 with scale=2 ────────────────────────────────────────────────
        (131, 'xca_4p_grad_sr075',  'pma_3lev_sc2_sr09_sig4'),
        (132, 'xca_4p_grad_sr075',  'pma_4lev_sc2_sr09_sig4'),

        # ── XCA use_grad ablation (xca_4p_nograd_sr08 vs xca_4p_grad_sr08) ────
        (133, 'xca_4p_nograd_sr08', 'pma_skip'),
        (134, 'xca_4p_nograd_sr08', 'pma_3lev_sr09_sig4'),
        (135, 'xca_4p_nograd_sr08', 'pma_3lev_sc2_sr09_sig4'),

        # ── PMA preprocessing ablation ────────────────────────────────────────
        # Baseline (preprocess=True, grad=False): configs 103 and 115
        # scale=4
        (136, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig4_nopreprocess'),          # F/F
        (137, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig4_pmagrad'),               # F/T
        (138, 'xca_4p_grad_sr08',   'pma_3lev_sr09_sig4_preprocess_pmagrad'),    # T/T
        # scale=2
        (139, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_nopreprocess'),      # F/F
        (140, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_pmagrad'),           # F/T
        (141, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_preprocess_pmagrad'),# T/T

        # ── ROI ablation: XCA-ROI + PMA-ROI (same lb) ─────────────────────────
        (142, 'xca_4p_grad_sr08_roi_lb07',  'pma_3lev_sc2_sr09_sig4_roi_lb07'),
        (143, 'xca_4p_grad_sr08_roi_lb08',  'pma_3lev_sc2_sr09_sig4_roi_lb08'),
        (144, 'xca_4p_grad_sr08_roi_lb085', 'pma_3lev_sc2_sr09_sig4_roi_lb085'),
        (145, 'xca_4p_grad_sr08_roi_lb09',  'pma_3lev_sc2_sr09_sig4_roi_lb09'),
        (146, 'xca_4p_grad_sr08_roi_lb095', 'pma_3lev_sc2_sr09_sig4_roi_lb095'),

        # ── ROI ablation: XCA-ROI + PMA-no-ROI ────────────────────────────────
        (147, 'xca_4p_grad_sr08_roi_lb07',  'pma_3lev_sc2_sr09_sig4'),
        (148, 'xca_4p_grad_sr08_roi_lb08',  'pma_3lev_sc2_sr09_sig4'),
        (149, 'xca_4p_grad_sr08_roi_lb085', 'pma_3lev_sc2_sr09_sig4'),
        (150, 'xca_4p_grad_sr08_roi_lb09',  'pma_3lev_sc2_sr09_sig4'),
        (151, 'xca_4p_grad_sr08_roi_lb095', 'pma_3lev_sc2_sr09_sig4'),

        # ── ROI ablation: XCA-no-ROI + PMA-ROI ────────────────────────────────
        (152, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_roi_lb07'),
        (153, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_roi_lb08'),
        (154, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_roi_lb085'),
        (155, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_roi_lb09'),
        (156, 'xca_4p_grad_sr08',   'pma_3lev_sc2_sr09_sig4_roi_lb095'),
    ]

    all_configs = [
        (cfg_id,
         xca_name, XCA_CONFIGS[xca_name], XCA_ROI_PARAMS.get(xca_name),
         pma_name, PMA_CONFIGS[pma_name])
        for cfg_id, xca_name, pma_name in EXPERIMENTS
    ]
    n_total = len(all_configs)

    print(f"Running {n_total} explicit configs (IDs {all_configs[0][0]}–{all_configs[-1][0]})")
    print()

    # Init or resume CSV
    if args.resume and os.path.exists(log_path):
        migrate_csv_schema(log_path)
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
    for i, (cfg_id, xca_name, xca_passes, xca_roi_params, pma_name, pma_cfg) in enumerate(all_configs):

        if cfg_id in done_ids:
            print(f"  [{i+1}/{n_total}] skipping {xca_name} + {pma_name} (done)")
            continue

        print("━" * 70)
        print(f"  [{i+1}/{n_total}]  cfg_id={cfg_id}  {xca_name}  +  {pma_name}")
        if xca_roi_params:
            print(f"  XCA ROI: lb={xca_roi_params['lb']}")
        if pma_cfg.get('run'):
            print(f"  PMA: use_matching_preprocess={pma_cfg.get('use_matching_preprocess', True)}"
                  f"  use_grad={pma_cfg.get('use_grad', False)}"
                  f"  use_roi={pma_cfg.get('use_roi', False)}")
        print("━" * 70)

        try:
            rcs, recon, align_time, recon_time = run_pipeline(
                projections, angles,
                xca_name, xca_passes, xca_roi_params,
                pma_cfg, recon_alg, downsample)
            if rcs < best_rcs:
                best_rcs    = rcs
                best_recon  = recon
                best_cfg_id = cfg_id
            status = 'ok'
        except Exception as e:
            traceback.print_exc()
            rcs = float('nan')
            align_time = recon_time = float('nan')
            status = f'error: {str(e)[:100]}'

        append_row(log_path, build_row(cfg_id, xca_name, xca_passes, xca_roi_params,
                                       pma_name, pma_cfg,
                                       rcs, align_time, recon_time, status))
        total_s = align_time + recon_time if not np.isnan(align_time) else float('nan')
        print(f"\n  → RCS = {rcs:.6f}  |  align = {align_time:.0f}s  |  recon = {recon_time:.0f}s  |  total = {total_s:.0f}s  |  {status}\n")

    # Save best reconstruction
    if best_recon is not None:
        tiff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'paramSearchBestRecon_{timestamp}.tiff')
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
            ['config_id', 'xca_name', 'xca_use_roi', 'xca_lb_crop_pct',
             'pma_name', 'pma_levels', 'pma_iters_per_level',
             'pma_of_sigma', 'pma_step_ratio',
             'pma_use_matching_preprocess', 'pma_use_grad',
             'pma_use_roi', 'pma_lb_crop_pct',
             'rcs', 'align_time_s', 'recon_time_s']
        ].to_string())
    except ImportError:
        print("(pandas not available — open the CSV directly for the full table.)")

    print(f"\nFull results saved to: {log_path}")


if __name__ == '__main__':
    main()
