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

python /Users/levih/Desktop/TomoMono/main.py --algorithms "$@"