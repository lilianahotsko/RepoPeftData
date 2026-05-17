#!/bin/bash
# Assemble the Code2LoRA-snapshots dataset (Dataset B) into a self-contained
# parquet tree under $OUT_DIR. CPU-only; depends on the v2 merge step.
#
# Submit::
#
#     sbatch --dependency=afterok:<MERGE_JOB_ID> \
#         scripts/slurm/build_code2lora_snapshots.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/{commits,qna}/*.parquet
#   $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/SNAPSHOTS_README.json

#SBATCH --job-name=build_c2l_snap
#SBATCH --output=slurm_logs/build_c2l_snap_%j.out
#SBATCH --error=slurm_logs/build_c2l_snap_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

V2_COMMITS_DIR="${V2_COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits}"
BASE_QNA_DIR="${BASE_QNA_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf/qna}"
TRAIN_QNAS_JSON="${TRAIN_QNAS_JSON:-$SCRATCH/REPO_DATASET/static_commit/splits_at_anchor/train.json}"
SNAP_OUT_DIR="${SNAP_OUT_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"

echo "V2 commits : $V2_COMMITS_DIR"
echo "Base qna   : $BASE_QNA_DIR"
echo "Train json : $TRAIN_QNAS_JSON"
echo "Output     : $SNAP_OUT_DIR"

python create_dataset/build_code2lora_snapshots_parquet.py \
    --v2-commits-dir "$V2_COMMITS_DIR" \
    --base-qna-dir "$BASE_QNA_DIR" \
    --train-qnas-json "$TRAIN_QNAS_JSON" \
    --out-dir "$SNAP_OUT_DIR" \
    "$@"

echo "Done: $(date)"
