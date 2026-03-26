#!/bin/bash
# ============================================================
# Text2LoRA reconstruction training — watgpu cluster
# (non-SLURM, run interactively or via screen/tmux)
#
# Prerequisites (run once from project root):
#   python baselines/text2lora/build_repo_descriptions.py \
#       --splits-dir $DATA_ROOT \
#       --repos-dir  $DATA_ROOT/repositories \
#       --text2lora-dir text2lora
#
#   python baselines/text2lora/prepare_oracle_loras.py \
#       --lora-root  $CKPT_DIR/PER_REPO_LORA \
#       --text2lora-dir text2lora \
#       --splits-dir $DATA_ROOT
#
# Usage: bash scripts/watgpu/train_text2lora.sh
# ============================================================
set -euo pipefail

source "$(dirname "$0")/common.sh"

TEXT2LORA_DIR="$(pwd)/text2lora"
CONFIG="$TEXT2LORA_DIR/configs/repopeft_text.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    echo "Run prepare_oracle_loras.py first (see script header)."
    exit 1
fi

echo "==== Text2LoRA Reconstruction Training (watgpu) ===="
echo "Start: $(date)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$TEXT2LORA_DIR"

export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python scripts/train_hyper_recon.py configs/repopeft_text.yaml

echo "==== Done: $(date) ===="
