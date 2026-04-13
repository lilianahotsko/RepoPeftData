#!/bin/bash
# ============================================================
# Text2LoRA CODE-CONDITIONED **SFT** training
# Strongest variant: end-to-end SFT (not reconstruction).
# Uses pre-computed Qwen3-Embedding code vectors (2048-dim).
#
# Prerequisites (same as recon variant):
#   python baselines/text2lora/extract_code_embeddings.py \
#       --splits-dir $SCRATCH/REPO_DATASET \
#       --output $SCRATCH/TEXT2LORA_DATA/code_embeddings.pt
#
# Submit: sbatch scripts/slurm/train_text2lora_code_sft.sh
# ============================================================
#SBATCH --job-name=t2l-code-sft
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/text2lora_code_sft_%j.out
#SBATCH --error=logs/text2lora_code_sft_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code-Conditioned SFT Training ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

CONFIG="text2lora/configs/repopeft_code_sft.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

export PYTHONPATH="$(pwd)/text2lora/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora_code_sft
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python baselines/text2lora/train_code_sft.py "$CONFIG"

echo "==== Done: $(date) ===="
