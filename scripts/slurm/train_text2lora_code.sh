#!/bin/bash
# ============================================================
# Text2LoRA CODE-CONDITIONED reconstruction training
# Uses pre-computed Qwen3-Embedding code vectors (2048-dim)
# instead of text descriptions.
#
# Prerequisites:
#   python baselines/text2lora/extract_code_embeddings.py \
#       --splits-dir $SCRATCH/REPO_DATASET \
#       --output $SCRATCH/TEXT2LORA_DATA/code_embeddings.pt
#
#   python baselines/text2lora/prepare_oracle_loras.py \
#       --lora-root $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA \
#       --text2lora-dir text2lora \
#       --splits-dir $SCRATCH/REPO_DATASET
#
# Submit: sbatch scripts/slurm/train_text2lora_code.sh
# ============================================================
#SBATCH --job-name=t2l-code-recon
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/text2lora_code_recon_%j.out
#SBATCH --error=logs/text2lora_code_recon_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code-Conditioned Reconstruction Training ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

CONFIG="text2lora/configs/repopeft_code.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

export PYTHONPATH="$(pwd)/text2lora/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora_code
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python baselines/text2lora/train_code_conditioned.py "$CONFIG"

echo "==== Done: $(date) ===="
