#!/bin/bash
# Push the Code2LoRA-snapshots (Dataset B) to HuggingFace Hub.
#
# Submit::
#
#     sbatch --dependency=afterok:<BUILD_JOB_ID> \
#         --export=ALL,HF_TOKEN \
#         scripts/slurm/push_code2lora_snapshots_to_hf.sh

#SBATCH --job-name=push_c2l_snap
#SBATCH --output=slurm_logs/push_c2l_snap_%j.out
#SBATCH --error=slurm_logs/push_c2l_snap_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

REPO_ID="${REPO_ID:-nanigock/repopeft-code2lora-snapshots}"
BUILD_DIR="${BUILD_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
PRIVATE_ARG="${PRIVATE_ARG:---private}"
COMMIT_MSG="${COMMIT_MSG:-Initial upload: Code2LoRA snapshots dataset.}"

echo "Repo  : $REPO_ID"
echo "Build : $BUILD_DIR"
echo "Private: $PRIVATE_ARG"

python scripts/push_code2lora_snapshots_to_hf.py \
    --repo-id "$REPO_ID" \
    --build-dir "$BUILD_DIR" \
    $PRIVATE_ARG \
    --commit-message "$COMMIT_MSG" \
    "$@"

echo "Done: $(date)"
