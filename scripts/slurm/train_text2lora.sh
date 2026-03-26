#!/bin/bash
# ============================================================
# Text2LoRA reconstruction training on Qwen2.5-Coder-1.5B
# Trains a hypernetwork to generate repo-specific LoRA weights
# from short natural-language repository descriptions.
#
# Prerequisites (run once from project root):
#   python baselines/text2lora/build_repo_descriptions.py \
#       --splits-dir $SCRATCH/REPO_DATASET \
#       --repos-dir  $SCRATCH/REPO_DATASET/repositories \
#       --text2lora-dir text2lora
#
#   python baselines/text2lora/prepare_oracle_loras.py \
#       --lora-root  $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA \
#       --text2lora-dir text2lora \
#       --splits-dir $SCRATCH/REPO_DATASET
#
# Submit: sbatch scripts/slurm/train_text2lora.sh
# ============================================================
#SBATCH --job-name=text2lora-recon
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/text2lora_recon_%j.out
#SBATCH --error=logs/text2lora_recon_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Reconstruction Training ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

# ── Prerequisites check ───────────────────────────────────────────────────────
TEXT2LORA_DIR="$(pwd)/text2lora"
CONFIG="$TEXT2LORA_DIR/configs/repopeft_text.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    echo "Run prepare_oracle_loras.py first (see script header)."
    exit 1
fi

N_TASKS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(len(cfg.get('train_ds_names', [])))
")
echo "Training repos: $N_TASKS"

# ── Training ──────────────────────────────────────────────────────────────────
cd "$TEXT2LORA_DIR"
export PYTHONPATH="$TEXT2LORA_DIR/src:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Disable wandb if no key is set
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python scripts/train_hyper_recon.py configs/repopeft_text.yaml

echo "==== Done: $(date) ===="
