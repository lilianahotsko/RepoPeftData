#!/bin/bash
# Common setup for watgpu cluster.
# Sourced by all training scripts — do not submit directly.

set -euo pipefail

# ── Adjust these for your watgpu environment ──
module purge 2>/dev/null || true
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

# Commit-level Parquet dataset (HF layout: commits/ qna/ splits/ [shards/]).
# Populated by scripts/watgpu/setup_commit_dataset.sh.
export COMMIT_DATA_DIR="$DATA_ROOT/commit_parquet_hf"

# Cache dirs so heavyweight downloads live on scratch (not $HOME quota).
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

# cd to repo root. SLURM copies sbatch scripts into its spool dir so
# $0 is unreliable inside jobs; prefer $SLURM_SUBMIT_DIR, then BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
fi

mkdir -p "$CKPT_DIR" "$BASELINES_DIR" slurm_logs
