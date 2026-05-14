#!/bin/bash
#SBATCH --output=/home/ljh79/TomoMono/sbatch_output/ganrec/ganrec-submit-%j.txt
#SBATCH --error=/home/ljh79/TomoMono/sbatch_output/ganrec/ganrec-submit-err-%j.txt
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --job-name="ganrec_submit"
#SBATCH --mail-user=ljh79@byu.edu
#SBATCH --mail-type=FAIL
# GANrec SLURM submission script with resume capability and auto-cleanup.
# Comment out the sections you don't need and uncomment the one you want to use.

# # ═════════════════════════════════════════════════════════════════════════════
# # FULL RESOLUTION (1x)
# # ═════════════════════════════════════════════════════════════════════════════
# ALIGNED_TIFF=$(ls -t /home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_fullres_aligned_*.tif 2>/dev/null | head -1)
# Y_START=40
# Y_END=440
# WIDTH_CROP=1200
# NUM_TASKS=8
# WALLTIME="20:00:00"
# MEM="120G"

# ═════════════════════════════════════════════════════════════════════════════
# 2x DOWNSAMPLED
# ═════════════════════════════════════════════════════════════════════════════
#ALIGNED_TIFF=$(ls -t /home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_2xds_aligned_*.tif 2>/dev/null | head -1)
#Y_START=20
#Y_END=220
#WIDTH_CROP=600
#NUM_TASKS=4
#WALLTIME="05:00:00"
#MEM="60G"

# ═════════════════════════════════════════════════════════════════════════════
# 4x DOWNSAMPLED
# ═════════════════════════════════════════════════════════════════════════════
ALIGNED_TIFF=$(ls -t /home/ljh79/TomoMono/alignedProjections/APSbeamtime_Oct25/cfg_4xds_aligned_*.tif 2>/dev/null | head -1)
Y_START=10
Y_END=110
WIDTH_CROP=300
NUM_TASKS=2
WALLTIME="02:00:00"
MEM="32G"

# ═════════════════════════════════════════════════════════════════════════════
# OPTIONS
# ═════════════════════════════════════════════════════════════════════════════
CLEANUP_SLICES=true  # Set to true to delete .npy files after successful merge
RESUME_SESSION=""     # If resuming, set to the session directory path (empty for new run)

# ═════════════════════════════════════════════════════════════════════════════

if [ -z "$ALIGNED_TIFF" ]; then
  echo "Error: No aligned TIFF found. Check that the correct config section is uncommented."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ljh79/.conda/envs/tomoMono/bin/python"
SBATCH_OUT="${SCRIPT_DIR}/sbatch_output/ganrec"
mkdir -p "${SBATCH_OUT}"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "GANrec Reconstruction Pipeline"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "TIFF: ${ALIGNED_TIFF}"
echo "Y range: [${Y_START}, ${Y_END})"
echo "Width crop: ${WIDTH_CROP} px"
echo "Tasks: ${NUM_TASKS}  |  Walltime: ${WALLTIME}  |  Memory: ${MEM}"
if [ "$CLEANUP_SLICES" = true ]; then
  echo "Cleanup: enabled (slice files will be deleted after merge)"
else
  echo "Cleanup: disabled"
fi
echo ""

# Check for incomplete sessions if not explicitly resuming
if [ -z "$RESUME_SESSION" ]; then
  echo "Checking for incomplete sessions..."
  INCOMPLETE_COUNT=$("${PYTHON}" "${SCRIPT_DIR}/runGANrec.py" --list-incomplete 2>&1 | grep -c "session_")
  if [ "$INCOMPLETE_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  Found incomplete session(s). To resume, use:"
    echo "     RESUME_SESSION='reconstructions/ganrec/session_<id>' bash runGPUGANrec.sh"
    echo ""
    "${PYTHON}" "${SCRIPT_DIR}/runGANrec.py" --list-incomplete
    echo ""
    if [ -t 0 ]; then
      read -p "Start a new run anyway? (y/n) " -n 1 -r
      echo
    else
      echo "Non-interactive mode: aborting to avoid overwriting incomplete sessions."
      exit 1
    fi
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborted."
      exit 1
    fi
  fi
fi

echo ""

# Generate or reuse session ID
if [ -n "$RESUME_SESSION" ]; then
  # Make the resume path absolute if it isn't already
  if [[ "$RESUME_SESSION" != /* ]]; then
    RESUME_SESSION="${SCRIPT_DIR}/${RESUME_SESSION}"
  fi
  SESSION_ID=$(basename "$RESUME_SESSION" | sed 's/^session_//')
  SESSION_DIR="$RESUME_SESSION"
  echo "Resuming session: ${SESSION_ID}"
else
  SESSION_ID=$(date '+%Y%m%d_%H%M%S_%N' | cut -c1-21)
  SESSION_DIR="${SCRIPT_DIR}/reconstructions/ganrec/session_${SESSION_ID}"
fi

echo "Session ID: ${SESSION_ID}"
echo "Session directory: ${SESSION_DIR}"
echo ""

# ── Step 1: reconstruction array job ─────────────────────────────────────────
RECON_JOB=$(sbatch --parsable \
    --time="${WALLTIME}" \
    --array="0-$((NUM_TASKS - 1))" \
    --output="${SBATCH_OUT}/ganrec-%A_%a.txt" \
    --error="${SBATCH_OUT}/ganrec-err-%A_%a.txt" \
    --ntasks=1 \
    --nodes=1 \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem="${MEM}" \
    --job-name="ganrec_recon" \
    --mail-user=ljh79@byu.edu \
    --mail-type=FAIL \
    --wrap="
        export OMP_NUM_THREADS=\$SLURM_CPUS_ON_NODE
        export PYTHONUNBUFFERED=1
        ulimit -c 0
        ${PYTHON} -u ${SCRIPT_DIR}/runGANrec.py \
          --aligned-tiff '${ALIGNED_TIFF}' \
          --y-start ${Y_START} --y-end ${Y_END} --width-crop ${WIDTH_CROP} \
          --session-id '${SESSION_ID}'
    "
)

echo "Reconstruction array job submitted: ${RECON_JOB}"
echo "  ${NUM_TASKS} tasks  |  walltime ${WALLTIME}  |  ${MEM} RAM / task"
echo ""

# ── Step 2: merge job (runs only if every array task succeeds) ────────────────
CLEANUP_FLAG=""
if [ "$CLEANUP_SLICES" = true ]; then
  CLEANUP_FLAG="--cleanup-slices"
fi

MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${RECON_JOB}" \
    --time="00:30:00" \
    --ntasks=1 \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=32G \
    --job-name="ganrec_merge" \
    --output="${SBATCH_OUT}/ganrec-merge-%j.txt" \
    --error="${SBATCH_OUT}/ganrec-merge-err-%j.txt" \
    --mail-user=ljh79@byu.edu \
    --mail-type=END,FAIL \
    --wrap="
        export PYTHONUNBUFFERED=1
        ulimit -c 0
        ${PYTHON} -u ${SCRIPT_DIR}/runGANrec.py --merge \
          --session-dir '${SESSION_DIR}' ${CLEANUP_FLAG}
    "
)

echo "Merge job submitted: ${MERGE_JOB}  (depends on ${RECON_JOB})"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          ${SBATCH_OUT}/"
echo ""
echo "To resume if this run is interrupted:"
echo "  RESUME_SESSION='${SESSION_DIR}' bash ${SCRIPT_DIR}/runGPUGANrec.sh"
echo ""
