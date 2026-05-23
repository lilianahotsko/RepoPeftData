#!/bin/bash
# ============================================================
# Text2LoRA CODE-CONDITIONED **SFT** training on the v2
# commit-derived RepoPeft dataset (code2lora_snapshots_hf).
#
# Mirrors scripts/slurm/train_text2lora_code_sft.sh but points at the
# v2 config (repopeft_code_sft_v2.yaml) and the v2 code-embedding
# artifact ($SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt).
#
# Prerequisites:
#   bash scripts/slurm/extract_code_embeddings_v2.sh   # ~1 min, CPU
#   # produces $SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt
#
# Submit: sbatch scripts/slurm/train_text2lora_code_sft_v2.sh
# ============================================================
#SBATCH --job-name=t2l-code-sft-v2
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/text2lora_code_sft_v2_%j.out
#SBATCH --error=logs/text2lora_code_sft_v2_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code-Conditioned SFT (v2 dataset) ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

CONFIG="text2lora/configs/repopeft_code_sft_v2.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

CODE_EMB="${CODE_EMB:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt}"
if [ ! -f "$CODE_EMB" ]; then
    echo "ERROR: v2 code embeddings missing: $CODE_EMB"
    echo "  Run: bash scripts/slurm/extract_code_embeddings_v2.sh"
    exit 1
fi

export PYTHONPATH="$(pwd)/text2lora/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=repopeft_text2lora_code_sft_v2
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

# Pin the trainer's run-name from the SLURM job id (override via the
# T2L_RUN_NAME env var) so a queued eval array can target the artifact
# path before training even starts.
export T2L_RUN_NAME="${T2L_RUN_NAME:-code_sft_v2_${SLURM_JOB_ID}}"
echo "Run name      : $T2L_RUN_NAME"
echo "Artifact dir  : text2lora/train_outputs/sft/hyper_lora/$T2L_RUN_NAME"

python baselines/text2lora/train_code_sft.py "$CONFIG"

echo "==== Done: $(date) ===="
