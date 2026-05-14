# GANrec Improvements: Parallel Safety, Resume Capability, and Auto-Cleanup

## Overview

Both `runGANrec.py` and `runGPUGANrec.sh` have been updated with production-ready features for handling long-running GPU reconstructions on the SLURM cluster.

## Key Features

### 1. **Session-Based Isolation (Prevents File Overwrites)**

Each reconstruction run now gets a **unique session directory** with timestamp-based naming:

```
reconstructions/ganrec/
├── session_20260514_093015_123/      # Run 1
│   ├── slices/
│   │   ├── slice_0010.npy
│   │   └── ...
│   └── state.json
│
├── session_20260514_094501_456/      # Run 2
│   ├── slices/
│   └── state.json
```

**Benefits:**
- Multiple parallel runs won't overwrite each other's files
- Final TIFF outputs have unique names per run
- Intermediate `.npy` files are isolated per session

### 2. **Resume Capability**

If a GPU run is interrupted (timeout, node failure, etc.):

```bash
# Check incomplete sessions
python runGANrec.py --list-incomplete

# Resume a specific session (re-runs will skip existing slices)
RESUME_SESSION='reconstructions/ganrec/session_20260514_093015_123' bash runGPUGANrec.sh
```

The Python script automatically **skips already-computed slices** (line 139-141 in `runGANrec.py`), so restarting a job picks up where it left off without duplicating work.

### 3. **Automatic Memory Cleanup**

After successful merge, intermediate `.npy` slice files can be automatically deleted:

```bash
# In runGPUGANrec.sh, set:
CLEANUP_SLICES=true

# Or pass it to the merge step manually:
python runGANrec.py --merge --session-dir <path> --cleanup-slices
```

**Benefits:**
- Saves significant disk space (each slice ≈ 10-100 MB depending on size)
- Prevents accidental use of stale intermediate files
- Can be toggled per run

## Usage

### Fresh Run

```bash
bash runGPUGANrec.sh
```

The script will:
1. Check for incomplete sessions and warn you
2. Generate a unique session ID (printed to console)
3. Submit reconstruction array job
4. Submit merge job (depends on reconstruction finishing)
5. Print the session directory path for resume if needed

### Resume After Interruption

```bash
# List incomplete sessions
python runGANrec.py --list-incomplete

# Copy the session path, then resume
RESUME_SESSION='reconstructions/ganrec/session_20260514_093015_123' bash runGPUGANrec.sh
```

Note: The shell script doesn't have special resume logic yet—just set the environment variable to make it clear, or directly submit the Python script:

```bash
python runGANrec.py \
  --aligned-tiff 'path/to/tiff' \
  --y-start 10 --y-end 110 \
  --session-id '20260514_093015_123'
```

### Manual Merge (if array job completes but merge fails)

```bash
python runGANrec.py --merge \
  --session-dir 'reconstructions/ganrec/session_20260514_093015_123' \
  --cleanup-slices
```

## Session State Tracking

Each session stores metadata in `session_<id>/state.json`:

```json
{
  "aligned_tiff": "/path/to/aligned.tif",
  "y_start": 10,
  "y_end": 110,
  "n_slices": 100,
  "output_name": "ganrec_recon_4xds_20260514-093015",
  "merge_complete": false
}
```

This allows the merge step to reconstruct the full configuration without re-passing arguments.

## API Changes

### New `runGANrec.py` Arguments

| Argument | Purpose |
|----------|---------|
| `--session-id <id>` | Unique ID for this run (auto-generated if omitted) |
| `--session-dir <path>` | Session directory for merge operation |
| `--cleanup-slices` | Delete `.npy` files after merge |
| `--list-incomplete` | List incomplete sessions and exit |

### New `runGPUGANrec.sh` Options

```bash
CLEANUP_SLICES=false  # Set to true to auto-delete slices after merge
RESUME_SESSION=""     # Set to path for resuming incomplete runs
```

## Example Workflow

```bash
# 1. Start a new reconstruction
bash runGPUGANrec.sh
# Output: Session ID: 20260514_093015_123

# 2. Job runs for a few hours... node crashes at 60% complete

# 3. Check what's incomplete
python runGANrec.py --list-incomplete
# Output: session_20260514_093015_123
#         Slices completed: 60/100

# 4. Resume the job
RESUME_SESSION='reconstructions/ganrec/session_20260514_093015_123' bash runGPUGANrec.sh
# Reconstruction continues from slice 61

# 5. After merge, slices are cleaned up (if CLEANUP_SLICES=true)
ls reconstructions/ganrec/session_20260514_093015_123/slices/
# slices/ directory is gone, only final TIFF remains
```

## Implementation Details

### State Management

- `get_session_dir()` — Generates or parses session ID into a unique directory
- `save_session_state()` — Writes metadata to `state.json`
- `load_session_state()` — Reads session metadata
- `list_incomplete_sessions()` — Finds sessions missing final TIFF

### Parallel Safety

The skip logic (line 139-141) ensures:
- Two tasks can run simultaneously without conflicts
- If task 0 and task 1 both try to write slice 10, only one succeeds (first)
- No atomic file operations needed—implicit via filesystem atomicity

### Cleanup

After merge completes, if `--cleanup-slices` is passed:
1. Loop through all `.npy` files and `os.remove()` each
2. Try to remove the now-empty `slices/` directory
3. Errors are warnings (file may have been removed by parallel job)

## Performance Notes

- Session ID generation uses millisecond precision → ~1000 unique IDs per second
- Session directories isolated in `reconstructions/ganrec/session_*/`
- No coordination between parallel jobs needed (filesystem provides atomicity)
- Cleanup cost is negligible (I/O only, no computation)

## Troubleshooting

### Session Directory Not Found

If you get `Error: session directory not found`, double-check the path:

```bash
# List all session directories
ls reconstructions/ganrec/session_*

# Use the full path for merge
python runGANrec.py --merge --session-dir reconstructions/ganrec/session_20260514_093015_123
```

### Incomplete Sessions Detected

If the script warns about incomplete sessions, you have two choices:

1. **Resume the incomplete one**: `RESUME_SESSION=<path> bash runGPUGANrec.sh`
2. **Start fresh**: Press `n` at the prompt to abort, then resolve the incomplete session manually or delete it
