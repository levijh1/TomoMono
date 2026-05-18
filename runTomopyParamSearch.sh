#!/bin/bash

#SBATCH --time=48:00:00  # walltime (48 hours for full-res parameter search)
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/tomopy_param_search/output-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/tomopy_param_search/output-error-%j.txt
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4   # OpenMP threads for tomopy reconstruction
#SBATCH --mem-per-cpu=125000M  # 4 CPUs x 125 GB = 500 GB total; full-res data ~160 GB + 2 copies + overhead
#SBATCH -J "tomopyParamSearch"   # job name
#SBATCH --mail-user=ljh79@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
ulimit -c 0

# Create output directories if they don't exist
mkdir -p /home/ljh79/TomoMono/sbatch_output/tomopy_param_search

TIFF_FILE=/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_fullres_aligned_20260514-115952.tif

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/home/ljh79/.conda/envs/tomoMono/bin/python /home/ljh79/TomoMono/main.py \
    --tiff-file "$TIFF_FILE" \
    --y-start 40 \
    --y-end 440 \
    --width 1200 \
    --output-dir /home/ljh79/TomoMono/reconstructions/APSbeamtime_Oct25/tomopy
