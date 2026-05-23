#!/bin/bash
#SBATCH --job-name=extract_code_emb_v2
#SBATCH --output=slurm_logs/extract_code_emb_v2_%j.out
#SBATCH --error=slurm_logs/extract_code_emb_v2_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-yuntian

# Extract per-repo 2048-d code embeddings (Qwen3-Embedding) from the v2
# commit-derived snapshot parquets and write them to a single .pt that
# the Text2LoRA Code-SFT trainer (and the v2 baseline evaluator) can
# consume directly.
#
# The script also writes a sibling ``<output>.train_repos.txt`` with the
# v2 train repo list so downstream config generation never drifts.
#
# Usage::
#   sbatch scripts/slurm/extract_code_embeddings_v2.sh
#   # or, for ad-hoc runs on a login node:
#   bash   scripts/slurm/extract_code_embeddings_v2.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits}"
OUTPUT="${OUTPUT:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt}"

echo "===== extract_code_embeddings_v2 ====="
echo "commits_dir: $COMMITS_DIR"
echo "output     : $OUTPUT"
echo "Start      : $(date)"

python baselines/text2lora/extract_code_embeddings_v2.py \
    --commits-dir "$COMMITS_DIR" \
    --output      "$OUTPUT"

echo "Done       : $(date)"
