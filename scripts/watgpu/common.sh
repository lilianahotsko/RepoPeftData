#!/bin/bash
# Common setup for watgpu cluster.
# Sourced by all training scripts — do not submit directly.

set -euo pipefail

# ── Adjust these for your watgpu environment ──
module purge
module load python/3.12 cuda/12.6 2>/dev/null || true

# Virtual environment
VENV_DIR="${SCRATCH:-$HOME/scratch}/venvs/repopeft"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run: scripts/watgpu/setup_env.sh first"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Data paths (populated by setup_env.sh from HuggingFace) ──
export DATA_ROOT="${SCRATCH:-$HOME/scratch}/repopeft_data"
export SPLITS_DIR="$DATA_ROOT/splits/main"
export ORACLE_CACHE_DIR="$DATA_ROOT/oracle_context_cache"
export CKPT_DIR="${SCRATCH:-$HOME/scratch}/repopeft_checkpoints"
export BASELINES_DIR="${SCRATCH:-$HOME/scratch}/repopeft_baselines"

cd "$(dirname "$0")/../.."   # repo root

mkdir -p "$CKPT_DIR" "$BASELINES_DIR" slurm_logs
