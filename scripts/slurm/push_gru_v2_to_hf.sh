#!/bin/bash
# Push the v2 GRU dataset to the HuggingFace Hub. CPU-only; depends on the
# merge step having produced commit_parquet_hf_v2/commits/*.parquet.
#
# Submit::
#
#     sbatch --dependency=afterok:<MERGE_JOB_ID> \
#         --export=ALL,HF_TOKEN \
#         scripts/slurm/push_gru_v2_to_hf.sh
#
# (HF_TOKEN must already be exported in the submitting shell, or stored at
# ~/.cache/huggingface/token from `huggingface-cli login`.)

#SBATCH --job-name=push_gru_v2
#SBATCH --output=slurm_logs/push_gru_v2_%j.out
#SBATCH --error=slurm_logs/push_gru_v2_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

REPO_ID="${REPO_ID:-nanigock/repopeft-gru-commits-v2}"
V2_DIR="${V2_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
V1_DIR="${V1_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
SMARTCAP_DIR="${SMARTCAP_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap}"
PRIVATE_ARG="${PRIVATE_ARG:---private}"
COMMIT_MSG="${COMMIT_MSG:-Initial upload: v2 (with diff/repo-state embeddings).}"

echo "Repo     : $REPO_ID"
echo "V2 dir   : $V2_DIR"
echo "V1 dir   : $V1_DIR"
echo "Smart    : $SMARTCAP_DIR"
echo "Private  : $PRIVATE_ARG"

python scripts/push_gru_v2_to_hf.py \
    --repo-id "$REPO_ID" \
    --v2-dir "$V2_DIR" \
    --v1-dir "$V1_DIR" \
    --smartcap-dir "$SMARTCAP_DIR" \
    $PRIVATE_ARG \
    --commit-message "$COMMIT_MSG" \
    "$@"

echo "Done: $(date)"
