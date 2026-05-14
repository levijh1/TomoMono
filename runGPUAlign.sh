#!/bin/bash

#SBATCH --time=6:00:00  # walltime
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/align/output-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/align/output-error-%j.txt
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4   # OpenMP threads for tomopy reconstruction
#SBATCH --mem-per-cpu=100000M  # 4 CPUs x 50 GB = 200 GB total; 1x data needs ~160 GB (16x the 10 GB used at 4x)
#SBATCH -J "alignAPSdata"   # job name
#SBATCH --mail-user=ljh79@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
ulimit -c 0

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/home/ljh79/.conda/envs/tomoMono/bin/python /home/ljh79/TomoMono/align.py
