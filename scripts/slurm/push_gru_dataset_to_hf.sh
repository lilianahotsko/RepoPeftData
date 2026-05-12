#!/bin/bash
# Push the smart-cap GRU train/val/test parquet to the HuggingFace Hub.
# This is CPU + network; runs on the login or a CPU node. ~13-14 GB upload.

#SBATCH --job-name=push_gru_to_hf
#SBATCH --output=slurm_logs/push_gru_to_hf_%j.out
#SBATCH --error=slurm_logs/push_gru_to_hf_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

: "${HF_REPO_ID:?must set HF_REPO_ID=<owner>/<repo>}"
: "${HF_TOKEN:?must set HF_TOKEN}"

STAGING_DIR="${STAGING_DIR:-$SCRATCH/REPO_DATASET/hf_push_staging}"
SMARTCAP_DIR="${SMARTCAP_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap}"
BASE_DIR="${BASE_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"

echo "HF repo     : $HF_REPO_ID"
echo "Smart-cap   : $SMARTCAP_DIR"
echo "Base        : $BASE_DIR"
echo "Staging     : $STAGING_DIR"

python scripts/push_gru_dataset_to_hf.py \
    --repo-id     "$HF_REPO_ID" \
    --smartcap-dir "$SMARTCAP_DIR" \
    --base-dir    "$BASE_DIR" \
    --staging-dir "$STAGING_DIR" \
    --token       "$HF_TOKEN" \
    "$@"

echo "Done: $(date)"
