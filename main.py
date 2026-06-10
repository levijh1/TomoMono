import os
import sys
import time
import h5py
import numpy as np
from datetime import datetime

import tomoDataClass
from helperFunctions import DualLogger, convert_to_numpy, convert_to_tiff

# =============================================================================
# Configuration — edit these before running
# =============================================================================

# Path to aligned projections TIFF (output of align.py)
TIFF_FILE = '/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_fullres_aligned_20260514-115952.tif'

# Where to save the reconstruction
OUTPUT_DIR = '/home/ljh79/TomoMono/reconstructions/APSbeamtime_Oct25/tomopy'

# Reconstruction algorithm and iteration count
# GPU options: 'SIRT_CUDA', 'ART_CUDA', 'FBP_CUDA'
# CPU options: 'gridrec', 'tv', 'sirt', 'art'
ALGORITHM = 'SIRT_CUDA'
NUM_ITER  = 400

# Set to False to skip saving (useful for quick tests)
SAVE = True

# Raw HDF5 file — used only to read the acquisition angles
RAW_HDF5 = '/home/ljh79/groups/grp_ptychi/nobackup/autodelete/Oct2025APSdata/tomo_data_run_final_2.hdf5'

# Projection indices to drop (bad angles). [] to keep all.
DROP_ANGLES = [19, 26]

# =============================================================================
# Setup
# =============================================================================

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)
sys.stdout = DualLogger(f'logs/main_{timestamp}.txt', 'w')

total_start = time.time()

print('=' * 72)
print(f'TomoMono 3D Reconstruction — {ALGORITHM}')
print(f'Started:  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'TIFF:     {TIFF_FILE}')
print(f'Output:   {OUTPUT_DIR}')
print('=' * 72)

# =============================================================================
# Load projections
# =============================================================================

print(f'\nLoading projections: {TIFF_FILE}')
t0 = time.time()
projections, scale_info = convert_to_numpy(TIFF_FILE)
projections = projections.astype(np.float32)
print(f'  shape (n_angles, h, w): {projections.shape}  [{time.time()-t0:.1f}s]')

# =============================================================================
# Load angles from raw HDF5
# =============================================================================

print(f'\nLoading angles from: {RAW_HDF5}')
with h5py.File(RAW_HDF5, 'r') as hf:
    ang_deg = hf['angles'][...]
ang_rad = ang_deg * np.pi / 180.0
if DROP_ANGLES:
    ang_rad = np.delete(ang_rad, DROP_ANGLES, axis=0)
# Center angles around zero so the rotation axis sits at the middle of the volume
angles = (ang_rad - np.mean(ang_rad)).astype(np.float32)
assert len(angles) == projections.shape[0], (
    f'Angle count {len(angles)} != projection count {projections.shape[0]}. '
    f'Check DROP_ANGLES.'
)
print(f'  angles: {len(angles)}  '
      f'range [{np.degrees(angles.min()):.2f}, {np.degrees(angles.max()):.2f}] deg')

# =============================================================================
# Normalize
# =============================================================================

# Phase contrast data: flip sign so high-phase regions are bright, then scale to [0, 1]
print('\nNormalizing (phase data: invert + scale to [0, 1])...')
projections = -projections
projections -= projections.min()
projections /= projections.max()

# =============================================================================
# Build tomoData object and center projections
# =============================================================================

print('\nCreating tomoData object...')
tomo = tomoDataClass.tomoData(projections, angles)
del projections  # free memory; all data now lives in tomo

# Iteratively shift projections until the rotation axis is centered
tomo.center_projections()
# Commit the centering shifts from workingProjections into finalProjections
tomo.make_updates_shift()

# =============================================================================
# Reconstruct
# =============================================================================

print(f'\nRunning {ALGORITHM} reconstruction ({NUM_ITER} iterations)...')
t_start = time.time()
tomo.reconstruct(algorithm=ALGORITHM, num_iter=NUM_ITER)
print(f'Reconstruction completed in {time.time()-t_start:.1f}s')

# =============================================================================
# Save
# =============================================================================

if SAVE:
    name_stem = os.path.splitext(os.path.basename(TIFF_FILE))[0]
    out_path = os.path.join(OUTPUT_DIR, f'{name_stem}_{ALGORITHM}_{timestamp}.tif')
    convert_to_tiff(tomo.get_recon(), out_path, scale_info)
    print(f'\nSaved: {out_path}')

print(f'\n{"="*72}')
print(f'Total time: {(time.time()-total_start)/60:.1f} min')
print(f'Done.  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
