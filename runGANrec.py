#!/usr/bin/env python3
"""
GANrec 3D reconstruction from pre-aligned projections.

Run modes:
  Reconstruction (default):
      python runGANrec.py
      Supports SLURM array jobs — set SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
      to split the slice range across tasks automatically.

  Merge partial results into final TIFF:
      python runGANrec.py --merge --session-dir reconstructions/ganrec/session_<id>

  Resume incomplete reconstruction (re-run with same session-id; existing slices are skipped):
      python runGANrec.py --aligned-tiff <tiff> --session-id <id> [--y-start N --y-end N ...]
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Paths (can be overridden via --aligned-tiff argument)
ALIGNED_TIFF    = None

RAW_HDF5        = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5'
OUTPUT_DIR      = 'reconstructions/ganrec'
OUTPUT_NAME     = 'ganrec_recon_4xds'

# Angle handling
# Indices to drop from the raw angle array (dataset-specific bad projections).
DROP_ANGLES     = [19, 26]

# Slice range: y-indices into the projection height dimension to reconstruct.
# None → use the full height. Y_END is exclusive (Python slice semantics).
# At full resolution: Y_START=40, Y_END=444
Y_START         = None      # e.g. 100
Y_END           = None      # e.g. 300

# Detector width crop: keep this many pixels centered on the detector mid-point.
# Reduces both GPU memory and per-slice compute time.
# None → use full detector width.
# At full resolution: 1200 pixels
PROJ_WIDTH_CROP = None

# GANrec hyperparameters (match values used in ganrec_realData_walkthrough.ipynb)
ITER_NUM        = 1000
L1_RATIO        = 300
G_LEARNING_RATE = 5e-4
RADIUS_RATIO    = 0.9
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import argparse
import glob
import time
import json
from datetime import datetime

import numpy as np
import h5py
import tifffile
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, OUTPUT_DIR)   # make absolute

sys.path.insert(0, _SCRIPT_DIR)
from helperFunctions import convert_to_numpy, convert_to_tiff
from ganrectorch.ganrec import GANtomo

# Generate or use provided session ID (ensures unique output dirs for parallel runs)
_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
OUTPUT_NAME = f'{OUTPUT_NAME}_{_timestamp}'


def get_session_dir(output_dir, session_id=None):
    """Get or create a unique session directory to isolate concurrent runs."""
    if session_id is None:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    session_dir = os.path.join(output_dir, f'session_{session_id}')
    return session_dir, session_id


def save_session_state(session_dir, state):
    """Save reconstruction state for resume capability."""
    state_file = os.path.join(session_dir, 'state.json')
    os.makedirs(session_dir, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_session_state(session_dir):
    """Load reconstruction state, returns None if not found."""
    state_file = os.path.join(session_dir, 'state.json')
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


def list_incomplete_sessions(output_dir):
    """Find incomplete session directories (no final TIFF yet)."""
    incomplete = []
    if not os.path.exists(output_dir):
        return incomplete

    for item in os.listdir(output_dir):
        if not item.startswith('session_'):
            continue
        session_path = os.path.join(output_dir, item)
        if not os.path.isdir(session_path):
            continue
        state = load_session_state(session_path)
        if state and not state.get('merge_complete', False):
            incomplete.append((item, state))

    return incomplete


def load_angles(hdf5_path, drop_indices):
    """Load angles from HDF5, drop bad projections, and centre around 0 (radians)."""
    with h5py.File(hdf5_path, 'r') as hf:
        ang_deg = hf['angles'][...]
    ang_rad = ang_deg * np.pi / 180.0
    if drop_indices:
        ang_rad = np.delete(ang_rad, drop_indices, axis=0)
    return ang_rad - np.mean(ang_rad)


def _gpu_info():
    """Return a one-line string describing the assigned GPU(s)."""
    if not torch.cuda.is_available():
        return 'no CUDA device'
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        lines.append(
            f'GPU {i}: {props.name}  '
            f'{props.total_memory / 2**30:.1f} GB total  '
            f'{free / 2**30:.1f} GB free'
        )
    return ' | '.join(lines)


def reconstruct(args):
    if ALIGNED_TIFF is None:
        print('Error: ALIGNED_TIFF must be provided via --aligned-tiff argument or set in config', flush=True)
        sys.exit(1)

    task_id    = int(os.environ.get('SLURM_ARRAY_TASK_ID',    0))
    task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    hostname   = os.environ.get('HOSTNAME', os.uname().nodename)

    # ── Task / GPU header ─────────────────────────────────────────────────────
    print('=' * 72, flush=True)
    print(f'GANrec  task {task_id}/{task_count-1}  |  host: {hostname}  |  pid: {os.getpid()}', flush=True)
    print(f'{_gpu_info()}', flush=True)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print('=' * 72, flush=True)

    session_dir, session_id = get_session_dir(OUTPUT_DIR, args.session_id)

    # ── Load projections ─────────────────────────────────────────────────────
    print(f'Loading projections: {ALIGNED_TIFF}', flush=True)
    t0 = time.time()
    projections, scale_info = convert_to_numpy(ALIGNED_TIFF)
    projections = projections.astype(np.float32)
    print(f'  shape (n_angles, h, w): {projections.shape}  [{time.time()-t0:.1f}s]', flush=True)

    # ── Load angles ──────────────────────────────────────────────────────────
    print(f'Loading angles from: {RAW_HDF5}', flush=True)
    angles = load_angles(RAW_HDF5, DROP_ANGLES).astype(np.float32)
    assert len(angles) == projections.shape[0], (
        f'Angle count {len(angles)} does not match projection count '
        f'{projections.shape[0]}. Adjust DROP_ANGLES.'
    )
    print(f'  angles: {len(angles)}  '
          f'range [{np.degrees(angles.min()):.2f}, {np.degrees(angles.max()):.2f}] deg', flush=True)

    # ── Y-slice range ─────────────────────────────────────────────────────────
    h = projections.shape[1]
    y_start = Y_START if Y_START is not None else 0
    y_end   = Y_END   if Y_END   is not None else h
    y_start = max(0, min(y_start, h))
    y_end   = max(y_start, min(y_end, h))
    n_slices = y_end - y_start
    print(f'Slice range: y=[{y_start}, {y_end})  ({n_slices} slices)', flush=True)

    # ── Detector width crop ───────────────────────────────────────────────────
    w = projections.shape[2]
    if PROJ_WIDTH_CROP is not None and PROJ_WIDTH_CROP < w:
        cx   = w // 2
        half = PROJ_WIDTH_CROP // 2
        projs = projections[:, y_start:y_end, cx - half : cx + half]
        print(f'Width crop: {w} → {projs.shape[2]} px (center {PROJ_WIDTH_CROP})', flush=True)
    else:
        projs = projections[:, y_start:y_end, :]

    del projections  # free the unneeded portion

    # ── SLURM array task splitting ────────────────────────────────────────────
    all_local_indices = list(range(n_slices))
    my_local_indices  = all_local_indices[task_id::task_count]

    print(f'Task {task_id}/{task_count-1}: assigned {len(my_local_indices)} of {n_slices} slices '
          f'(indices {my_local_indices[0]}–{my_local_indices[-1]} interleaved)', flush=True)

    # ── Output directory (per-session) ────────────────────────────────────────
    slices_dir = os.path.join(session_dir, 'slices')
    os.makedirs(slices_dir, exist_ok=True)

    # Save session metadata for resume capability
    session_state = {
        'aligned_tiff': ALIGNED_TIFF,
        'y_start': y_start,
        'y_end': y_end,
        'n_slices': n_slices,
        'output_name': OUTPUT_NAME,
        'merge_complete': False,
    }
    save_session_state(session_dir, session_state)

    print(f'Session ID:       {session_id}', flush=True)
    print(f'Slices directory: {slices_dir}', flush=True)
    print('-' * 72, flush=True)

    # ── Per-slice reconstruction ──────────────────────────────────────────────
    task_start = time.time()
    for idx, local_y in enumerate(my_local_indices):
        global_y   = y_start + local_y
        slice_path = os.path.join(slices_dir, f'slice_{global_y:04d}.npy')

        if os.path.exists(slice_path):
            print(f'[task {task_id} | {idx+1}/{len(my_local_indices)}] '
                  f'slice {global_y:4d} — skipping (already exists)', flush=True)
            continue

        t_slice = time.time()
        sino    = projs[:, local_y, :].astype(np.float32)

        print(f'[task {task_id} | {idx+1}/{len(my_local_indices)}] '
              f'slice {global_y:4d} — starting ({ITER_NUM} iters) ...', flush=True)

        out = GANtomo(
            -sino,
            angles,
            iter_num        = ITER_NUM,
            l1_ratio        = L1_RATIO,
            g_learning_rate = G_LEARNING_RATE,
            radius_ratio    = RADIUS_RATIO,
            recon_monitor   = False,
        ).recon()

        out = np.asarray(out, dtype=np.float32)
        while out.ndim > 2:
            out = out[0]

        np.save(slice_path, out)

        elapsed_slice = time.time() - t_slice
        elapsed_total = time.time() - task_start
        remaining     = (len(my_local_indices) - idx - 1) * elapsed_slice
        print(
            f'[task {task_id} | {idx+1}/{len(my_local_indices)}] '
            f'slice {global_y:4d} DONE  '
            f'slice={elapsed_slice:.1f}s  '
            f'total={elapsed_total:.0f}s  '
            f'ETA≈{remaining/60:.1f}min',
            flush=True,
        )

    print(f'\nTask {task_id} finished — {len(my_local_indices)} slices in '
          f'{(time.time()-task_start)/60:.1f} min', flush=True)


def merge(args):
    """Merge slices from a session directory into final TIFF and optionally cleanup."""
    session_dir = args.session_dir
    if session_dir is None:
        print('Merge mode requires --session-dir argument')
        sys.exit(1)

    if not os.path.exists(session_dir):
        print(f'Error: session directory not found: {session_dir}')
        sys.exit(1)

    state = load_session_state(session_dir)
    if state is None:
        print(f'Error: no session state found in {session_dir}', flush=True)
        sys.exit(1)

    slices_dir = os.path.join(session_dir, 'slices')
    pattern    = os.path.join(slices_dir, 'slice_*.npy')
    files      = sorted(glob.glob(pattern))

    if not files:
        print(f'No slice files found in {slices_dir}. Run reconstruction first.', flush=True)
        sys.exit(1)

    expected = state.get('n_slices')
    if expected and len(files) != expected:
        print(f'Error: expected {expected} slices but found {len(files)}. '
              f'Some reconstruction tasks may have failed. '
              f'Re-run with the same --session-id to fill missing slices, '
              f'or delete the session directory to start fresh.', flush=True)
        sys.exit(1)

    print(f'Merging {len(files)} slices from {slices_dir} ...', flush=True)
    slices = [np.load(f) for f in files]
    volume = np.stack(slices, axis=0).astype(np.float32)
    print(f'  Volume shape: {volume.shape}', flush=True)

    # Read only scale metadata from the aligned TIFF (avoid loading the full array)
    aligned_tiff = state['aligned_tiff']
    output_name  = state['output_name']
    try:
        with tifffile.TiffFile(aligned_tiff) as tif:
            page = tif.pages[0]
            scale_info = {
                'XResolution': page.tags['XResolution'].value,
                'YResolution': page.tags['YResolution'].value,
                'Unit':        page.tags['ResolutionUnit'].value,
            }
    except (KeyError, Exception):
        scale_info = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{output_name}.tiff')
    convert_to_tiff(volume, out_path, scale_info)
    print(f'Saved: {out_path}', flush=True)

    # Update state to mark merge as complete
    state['merge_complete'] = True
    save_session_state(session_dir, state)

    # Cleanup intermediate slice files if requested
    if args.cleanup_slices:
        print('Cleaning up intermediate slice files...', flush=True)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f'  Warning: could not delete {f}: {e}', flush=True)
        try:
            os.rmdir(slices_dir)
            print(f'  Removed slices directory: {slices_dir}', flush=True)
        except OSError:
            pass

    print('Merge complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--merge', action='store_true',
                        help='Merge saved per-slice .npy files into the final 3D TIFF')
    parser.add_argument('--aligned-tiff', type=str, default=None,
                        help='Path to aligned projections TIFF file')
    parser.add_argument('--y-start', type=int, default=None,
                        help='Y slice start index (overrides Y_START config)')
    parser.add_argument('--y-end', type=int, default=None,
                        help='Y slice end index (overrides Y_END config)')
    parser.add_argument('--width-crop', type=int, default=None,
                        help='Detector width crop in pixels (overrides PROJ_WIDTH_CROP config)')
    parser.add_argument('--session-id', type=str, default=None,
                        help='Session ID for reconstruction. If not provided, a unique one is generated.')
    parser.add_argument('--session-dir', type=str, default=None,
                        help='Full path to session directory (required for --merge)')
    parser.add_argument('--cleanup-slices', action='store_true',
                        help='Delete intermediate .npy slice files after successful merge')
    parser.add_argument('--list-incomplete', action='store_true',
                        help='List incomplete sessions and exit')
    parsed = parser.parse_args()

    if parsed.aligned_tiff is not None:
        ALIGNED_TIFF = parsed.aligned_tiff
    if parsed.y_start is not None:
        Y_START = parsed.y_start
    if parsed.y_end is not None:
        Y_END = parsed.y_end
    if parsed.width_crop is not None:
        PROJ_WIDTH_CROP = parsed.width_crop

    if parsed.list_incomplete:
        incomplete = list_incomplete_sessions(OUTPUT_DIR)
        if incomplete:
            print(f'Found {len(incomplete)} incomplete session(s):')
            for session_name, state in incomplete:
                session_path = os.path.join(OUTPUT_DIR, session_name)
                slices_dir = os.path.join(session_path, 'slices')
                n_slices = len(glob.glob(os.path.join(slices_dir, 'slice_*.npy')))
                print(f'  {session_name}')
                print(f'    Path: {session_path}')
                print(f'    Slices completed: {n_slices}/{state.get("n_slices", "?")}')
        else:
            print('No incomplete sessions found.')
        sys.exit(0)

    if parsed.merge:
        merge(parsed)
    else:
        reconstruct(parsed)
