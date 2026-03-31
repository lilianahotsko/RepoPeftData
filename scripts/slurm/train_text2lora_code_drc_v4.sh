#!/bin/bash
# ============================================================
# Text2LoRA CODE-CONDITIONED + DRC v4 reconstruction training
# Oracle LoRAs trained with DRC v4 (adaptive) context at 8K
#
# Prerequisites:
#   1. Per-repo LoRA + DRC v4 training completed
#   2. bash scripts/slurm/setup_oracle_lora_drc_v4.sh
#
# Submit: sbatch scripts/slurm/train_text2lora_code_drc_v4.sh
# ============================================================
#SBATCH --job-name=t2l-code-drc4
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/text2lora_code_drc_v4_%j.out
#SBATCH --error=logs/text2lora_code_drc_v4_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code-Conditioned + DRC v4 Reconstruction Training ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

CONFIG="text2lora/configs/repopeft_code_drc_v4.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

export PYTHONPATH="$(pwd)/text2lora/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora_code_drc_v4
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python baselines/text2lora/train_code_conditioned.py "$CONFIG"

echo "==== Done: $(date) ===="
