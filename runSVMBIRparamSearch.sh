#!/bin/bash
# SVMBIR is CPU-only (no GPU required).
#SBATCH --time=36:00:00
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/general_recons/svmbir_search-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/general_recons/svmbir_search-error-%j.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6500M   # 32 CPUs x 6.5 GB = 208 GB; same as runSVMBIRrec_fullres.sh
#SBATCH -J "svmbir_search"
#SBATCH --mail-user=ljh79@byu.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -c 0

TIFF_FILE=/home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_fullres_aligned_20260514-115952.tif

# With 32 CPUs: 4 workers × 8 OMP threads each = 32 CPUs fully utilized.
# Adjust --num-workers to trade per-recon speed vs. total throughput.
# Results CSV → hyperparam_results/svmbir_search_<timestamp>.csv
# TIFFs       → reconstructions/APSbeamtime_Oct25/svmbir_recons_<timestamp>/
# Slice PNGs  → reconstructions/APSbeamtime_Oct25/svmbir_slices_<timestamp>/

/home/ljh79/.conda/envs/tomoMono/bin/python /home/ljh79/TomoMono/runSVMBIRparamSearch.py \
    --tiff-file "$TIFF_FILE" \
    --y-start 40 \
    --y-end 440 \
    --width 1200 \
    --output-dir /home/ljh79/TomoMono/reconstructions/APSbeamtime_Oct25 \
    --num-workers 4
    # Add --fsc to also compute FSC resolution (triples runtime)
    # Add --configs name1,name2 to run only specific configs
