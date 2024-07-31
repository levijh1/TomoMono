
import time
import sys
import tomoDataClass
from tiffConverter import convert_to_numpy, convert_to_tiff
from datetime import datetime
import torch
import argparse
from helperFunctions import DualLogger, subpixel_shift
from tqdm import tqdm
from scipy.ndimage import shift
import numpy as np


# Configuration flags
log = False  # Enable logging to file
saveToFile = True  # Enable saving data to file

# Start the timer for execution duration tracking
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Timestamp for file naming


# Setup logging if enabled
if log:
    sys.stdout = DualLogger(f'logs/output_tomoMono_align{timestamp}.txt', 'w')


print("Running Image Registration Script")

# Check for GPU availability
if torch.cuda.is_available():
    print("GPU is available")


# Import foam data
numAngles = 800
tif_file = "data/fullTomoReconstructions2.tif"
obj, scale_info = convert_to_numpy(tif_file)
obj = obj[0:numAngles]
print(obj.shape)
tomo = tomoDataClass.tomoData(obj)
tomo.crop_center(900,550)


#Actually align data
print("Starting alignment")
tomo.track_shifts()
tomo.cross_correlate_align()
tomo.center_projections()
# tomo.tomopy_align(iterations = 10)
tomo.optical_flow_align()
tomo.center_projections()
print(tomo.tracked_shifts)

#Apply changes on unchanged projections
for m in tqdm(range(tomo.num_angles), desc='Center projections'):
    tomo.originalProjections[m] = subpixel_shift(tomo.originalProjections[m], tomo.tracked_shifts[m,0], tomo.tracked_shifts[m,1])
tomo.projections = tomo.originalProjections

tomo.makeScriptProjMovie()


# #Save the aligned data
if saveToFile:
    convert_to_tiff(tomo.get_projections(), f"alignedProjections/aligned_foamTomo{timestamp}.tif", scale_info)
    np.save(f'shiftValues_{timestamp}.npy', tomo.tracked_shifts)


# End the timer
end_time = time.time()

# Calculate and print the duration
print(f"Script completed in {end_time - start_time} seconds.")

if log:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

