#!/bin/bash
#SBATCH --partition=ampere
##SBATCH --nodelist=sdfampere005
#SBATCH --account=lcls:xppl1026722
#SBATCH --job-name=reconstruct_dark
#SBATCH --output=/sdf/home/l/levih/output/output-%j.txt
#SBATCH --error=/sdf/home/l/levih/output/output-error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=80g
#SBATCH --time=0-03:0:00
#SBATCH --gpus 3

PTY_OPTIONS1=/sdf/home/l/levih/xppl026722/results/specs/xppl1026722/recipe_run490.sh
PTY_OPTIONS2=/sdf/home/l/levih/xppl026722/results/specs/xppl1026722/max_output.sh
NUM_INPUT_FRAMES=100
LOAD_FRAMES=true
MASK_OUTLIERS=false
INPUT_FILE="${PTY_INPUT_DIR}/hdf5/smalldata/xppl1026722_Run0394.h5"


/sdf/home/l/levih/xppl026722/results/bin/PtyRelease --pty_options=$PTY_OPTIONS1 --pty_options=$PTY_OPTIONS2 --export_input_frames_to_tiff_period=$NUM_INPUT_FRAMES --export_load_frames_to_tiff=$LOAD_FRAMES --mask_beam_energy_outliers=$MASK_OUTLIERS --input_file=$INPUT_FILE