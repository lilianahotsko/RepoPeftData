#!/bin/bash
#SBATCH --account=rrg-yuntian
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=c2l_excl_pred
#SBATCH --output=/scratch/lhotsko/slurm_logs/c2l_exclusive_pred_%j.out

set -euo pipefail
REPO=/home/lhotsko/RepoPeftData
SCRATCH=${SCRATCH:-/scratch/lhotsko}
source "$SCRATCH/venvs/qwen-cu126-py312/bin/activate"
export HF_HOME="$SCRATCH/REPO_DATASET/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$REPO"
python scripts/collect_c2l_exclusive_examples.py \
  --suite cr_test \
  --limit 5 \
  --run-predictions \
  --output "$REPO/RepoPeft_Paper/qualitative_c2l_exclusive_examples.md"
