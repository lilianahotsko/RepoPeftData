#!/bin/bash
#SBATCH --job-name=positive_analysis
#SBATCH --output=slurm_logs/positive_analysis_%j.out
#SBATCH --error=slurm_logs/positive_analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --account=rrg-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/collect_positive_analysis_examples.py \
  --suite cr_test \
  --num-all 5 \
  --num-c2l 5 \
  --output RepoPeft_Paper/positive_analysis.md
