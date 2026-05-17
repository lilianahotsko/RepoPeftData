#!/bin/bash
# Merge the diff + repo-state embedding shards into commit_parquet_hf_v2/.
# CPU-only; depends on both shard array jobs finishing successfully.
#
# Submit::
#
#     sbatch --dependency=afterok:<DIFF_JOB_ID>:<REPO_JOB_ID> \
#         scripts/slurm/merge_gru_v2_embeddings.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits/{train,cr_val,cr_test}.parquet
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/EMBEDDINGS_README.json

#SBATCH --job-name=merge_gru_v2
#SBATCH --output=slurm_logs/merge_gru_v2_%j.out
#SBATCH --error=slurm_logs/merge_gru_v2_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf/commits}"
SHARDS_BASE="${SHARDS_BASE:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards}"
OUT_DIR_V2="${OUT_DIR_V2:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"

echo "Commits  : $COMMITS_DIR"
echo "Shards   : $SHARDS_BASE"
echo "Output   : $OUT_DIR_V2"

python create_dataset/merge_gru_v2_embeddings.py \
    --commits-dir "$COMMITS_DIR" \
    --shards-base "$SHARDS_BASE" \
    --out-dir "$OUT_DIR_V2" \
    --splits train cr_val cr_test \
    "$@"

echo "Done: $(date)"
