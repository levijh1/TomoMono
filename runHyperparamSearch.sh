#!/bin/bash

#SBATCH --time=12:00:00   # walltime — 0.5 days; ~120 configs x ~20-30 min each
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/hyperparam-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/hyperparam-error-%j.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4   # OpenMP threads for tomopy reconstruction
#SBATCH --mem-per-cpu=8192M   # 8 GB x 4 CPUs = 32 GB total
#SBATCH -J "hyperparamSearch"
#SBATCH --mail-user=ljh79@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set OpenMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Results directory — created here so the path is guaranteed to exist
mkdir -p /home/ljh79/TomoMono/hyperparam_results

# Fixed log file name so --resume works if the job is resubmitted after a timeout
LOG=/home/ljh79/TomoMono/hyperparam_results/hyperparam_results.csv

/home/ljh79/.conda/envs/tomoMono/bin/python \
    /home/ljh79/TomoMono/hyperparameter_search.py \
    --resume \
    --recon SIRT_CUDA \
    --logfile "$LOG"
