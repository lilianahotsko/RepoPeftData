#!/bin/bash
# ============================================================
# Text2LoRA CODE-CONDITIONED **SFT** training on the v2
# commit-derived RepoPeft dataset (code2lora_snapshots_hf) with
# ALL 7 attention + MLP target projections (q/k/v/o/gate/up/down).
#
# Sibling of train_text2lora_code_sft_v2.sh; only the config path
# and the WANDB project name differ. The companion config
# (repopeft_code_sft_v2_full7.yaml) halves inp_max_len 2048 -> 1024
# to fit the FFN-module LoRA bmm hook on a single H100 80GB.
#
# Prerequisites:
#   bash scripts/slurm/extract_code_embeddings_v2.sh
#
# Submit: sbatch scripts/slurm/train_text2lora_code_sft_v2_full7.sh
# ============================================================
#SBATCH --job-name=t2l-code-sft-v2-full7
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/text2lora_code_sft_v2_full7_%j.out
#SBATCH --error=logs/text2lora_code_sft_v2_full7_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code-Conditioned SFT (v2 dataset, 7 target modules) ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

CONFIG="text2lora/configs/repopeft_code_sft_v2_full7.yaml"

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
export WANDB_PROJECT=repopeft_text2lora_code_sft_v2_full7
export WANDB_WATCH=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

# Pin run-name from SLURM job id so a downstream eval array can target
# the artifact path before training even starts.
export T2L_RUN_NAME="${T2L_RUN_NAME:-code_sft_v2_full7_${SLURM_JOB_ID}}"
echo "Run name      : $T2L_RUN_NAME"
echo "Artifact dir  : text2lora/train_outputs/sft/hyper_lora/$T2L_RUN_NAME"

# Redirect this run's outputs to scratch via a per-run symlink so the ~12 GB
# hypermod.pt does not exhaust the 50 GiB /home quota (cause of prior
# job 14775662 failure: "Disk quota exceeded" while writing hypermod.pt).
SCRATCH_RUN_DIR="$SCRATCH/text2lora_train_outputs/sft/hyper_lora/$T2L_RUN_NAME"
HOME_RUN_DIR="text2lora/train_outputs/sft/hyper_lora/$T2L_RUN_NAME"
mkdir -p "$SCRATCH_RUN_DIR"
mkdir -p "$(dirname "$HOME_RUN_DIR")"
if [ ! -e "$HOME_RUN_DIR" ]; then
    ln -s "$SCRATCH_RUN_DIR" "$HOME_RUN_DIR"
fi
echo "Symlinked $HOME_RUN_DIR -> $SCRATCH_RUN_DIR"

python baselines/text2lora/train_code_sft.py "$CONFIG"

echo "==== Done: $(date) ===="
