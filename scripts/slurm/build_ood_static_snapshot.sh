#!/bin/bash
#SBATCH --job-name=build_ood_static_snapshot
#SBATCH --output=slurm_logs/build_ood_static_snapshot_%A_%a.out
#SBATCH --error=slurm_logs/build_ood_static_snapshot_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

# Build the OOD-test static snapshot parquet for Code2LoRA-direct evaluation.
#
# Phase 1 (per-shard): compute per-commit repo_state_embedding for all 1,950
#   OOD commits using the canonical Qwen3-Embedding-0.6B recipe + the per-repo
#   blob cache. Submit as a 4-shard array.
# Phase 2 (after array): merge shards into a snapshot-format parquet at
#   $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/ood_test.parquet
#   keeping ONE anchor commit per OOD repo (latest commit per repo).
#
# Usage:
#   sbatch --array=0-3 scripts/slurm/build_ood_static_snapshot.sh
#   # then once all 4 finish, run scripts/build_ood_snapshot_parquet.py
#   # (the post-process step is launched automatically if SLURM_ARRAY_TASK_ID==N-1
#   # is the last to complete -- but to be safe call it manually after the array.)

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"
INPUT_DIR="${INPUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood_for_embed}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_ood/repo_state}"

echo "===== build OOD repo-state embeddings  shard ${SHARD_I}/${NUM_SHARDS} ====="
echo "Input dir : $INPUT_DIR"
echo "Out dir   : $OUT_DIR"
echo "Start     : $(date)"
nvidia-smi -L || true

python create_dataset/build_repo_state_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --splits ood_test \
    --shard-index "$SHARD_I" \
    --shard-total "$NUM_SHARDS"

echo "Done: $(date)"
