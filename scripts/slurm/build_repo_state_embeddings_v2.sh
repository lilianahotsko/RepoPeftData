#!/bin/bash
# 8-shard array job: compute 2048-d repo-state embeddings for the full
# commits/{train,cr_val,cr_test}.parquet of the GRU dataset.
#
# Submit::
#
#     sbatch --array=0-7 scripts/slurm/build_repo_state_embeddings_v2.sh
#
# Input:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf/commits/*.parquet
#   $SCRATCH/REPO_DATASET/static_commit/cache/<repo>/{blob_*.,_snapshot_*.}
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/repo_state/<split>/shard_<I>_of_8.parquet
#
# Most blobs are already cached (~90% hit rate from the previous embedding
# job), so wall is dominated by ``git ls-tree`` per snapshot:
# ~10 min/shard for cr_val/cr_test (~1 k commits/shard) and ~20-40 min/shard
# for train (where ~6 k commits/shard need new pooling). Hits new GPU only
# for the rare cache-miss blob.

#SBATCH --job-name=build_repo_state_v2
#SBATCH --output=slurm_logs/build_repo_state_v2_%A_%a.out
#SBATCH --error=slurm_logs/build_repo_state_v2_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

SHARD_TOTAL="${SHARD_TOTAL:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"

INPUT_DIR="${INPUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf/commits}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/repo_state}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/REPO_DATASET/static_commit/cache}"

echo "Shard ${SHARD_INDEX}/${SHARD_TOTAL}"
echo "Input    : $INPUT_DIR"
echo "Output   : $OUT_DIR"
echo "Cache    : $CACHE_ROOT"

python create_dataset/build_repo_state_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --cache-root "$CACHE_ROOT" \
    --splits train cr_val cr_test \
    --shard-index "$SHARD_INDEX" \
    --shard-total "$SHARD_TOTAL" \
    --device cuda \
    "$@"

echo "Done: $(date)"
