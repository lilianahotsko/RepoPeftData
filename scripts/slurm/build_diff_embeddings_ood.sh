#!/bin/bash
# 4-shard array job: compute 2048-d diff embeddings for the OOD commits
# parquet so that the v2 GRU evaluator can stream OOD diffs end-to-end on
# the same protocol as Tables 1/2.
#
# Submit::
#
#     sbatch --account=rrg-yuntian --array=0-3 \
#         scripts/slurm/build_diff_embeddings_ood.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/diff/ood_test/shard_<I>_of_4.parquet
#
# Wall: ~25-50 min/shard on H100 (~1950 commits total / 4 shards).

#SBATCH --job-name=build_diff_emb_ood
#SBATCH --output=slurm_logs/build_diff_emb_ood_%A_%a.out
#SBATCH --error=slurm_logs/build_diff_emb_ood_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=rrg-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

SHARD_TOTAL="${SHARD_TOTAL:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"

INPUT_DIR="${INPUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_ood_staging}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/diff}"

echo "Shard ${SHARD_INDEX}/${SHARD_TOTAL}"
echo "Input  : $INPUT_DIR"
echo "Output : $OUT_DIR"

python create_dataset/build_diff_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --splits ood_test \
    --shard-index "$SHARD_INDEX" \
    --shard-total "$SHARD_TOTAL" \
    --device cuda \
    --batch-size 16 \
    --diff-batch 64 \
    "$@"

echo "Done: $(date)"
