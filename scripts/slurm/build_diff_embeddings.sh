#!/bin/bash
# 8-shard array job: compute 2048-d diff embeddings for the full
# commits/{train,cr_val,cr_test}.parquet of the GRU dataset.
#
# Submit::
#
#     sbatch --array=0-7 scripts/slurm/build_diff_embeddings.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/diff/<split>/shard_<I>_of_8.parquet
#
# Wall: ~30-60 min/shard on H100 with ~9 k diffs/shard.

#SBATCH --job-name=build_diff_emb
#SBATCH --output=slurm_logs/build_diff_emb_%A_%a.out
#SBATCH --error=slurm_logs/build_diff_emb_%A_%a.err
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
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/diff}"

echo "Shard ${SHARD_INDEX}/${SHARD_TOTAL}"
echo "Input  : $INPUT_DIR"
echo "Output : $OUT_DIR"

python create_dataset/build_diff_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --splits train cr_val cr_test \
    --shard-index "$SHARD_INDEX" \
    --shard-total "$SHARD_TOTAL" \
    --device cuda \
    --batch-size 16 \
    --diff-batch 64 \
    "$@"

echo "Done: $(date)"
