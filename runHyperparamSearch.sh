#!/bin/bash

#SBATCH --time=36:00:00   # walltime — 2x data ~16h est; 36h gives buffer for convergence variance
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/hyperparam-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/hyperparam-error-%j.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4   # OpenMP threads for tomopy reconstruction
#SBATCH --mem-per-cpu=12288M   # 12 GB x 4 CPUs = 48 GB total; 2x data is 4x larger (~41 GB peak)
#SBATCH -J "hyperparamSearch"
#SBATCH --mail-user=ljh79@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set OpenMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -c 0

# Results directory — created here so the path is guaranteed to exist
mkdir -p /home/ljh79/TomoMono/hyperparam_results

/home/ljh79/.conda/envs/tomoMono/bin/python \
    /home/ljh79/TomoMono/hyperparameter_search.py \
    --recon SIRT_CUDA \
    --downsample 2 \
    --resume \
    --logfile /home/ljh79/TomoMono/hyperparam_results/xca_pma_search_ds2_20260505_124846.csv
