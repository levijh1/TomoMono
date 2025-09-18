#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20480M   # memory per CPU core
#SBATCH -J "alignAPSdata"   # job name
#SBATCH --mail-user=ljh79@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/home/ljh79/.conda/envs/tomoMono/bin/python /home/ljh79/TomoMono/align.py
