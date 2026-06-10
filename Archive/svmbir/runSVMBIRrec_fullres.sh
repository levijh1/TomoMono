#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/general_recons/output-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/general_recons/output-error-%j.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6500M   # 32 CPUs x 6.5 GB = 208 GB total; full-res SVMBIR
#SBATCH -J "svmbir_fullres"
#SBATCH --mail-user=ljh79@byu.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -c 0

/home/ljh79/.conda/envs/tomoMono/bin/python /home/ljh79/TomoMono/runSVMBIRrec.py \
    --tiff-file /home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_fullres_aligned_20260514-115952.tif \
    --y-start 40 \
    --y-end 440 \
    --width 1200 \
    --output-name cfg_fullres
