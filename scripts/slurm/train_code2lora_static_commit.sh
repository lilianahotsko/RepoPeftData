#!/bin/bash
# Train Code2LoRA-direct (static hypernetwork) on the commit-aware splits
# produced by ``create_dataset/build_static_commit_train_jsonl.py``.
#
# Inputs:
#   $SCRATCH/REPO_DATASET/static_commit/splits/{train,cr_val,cr_test}.json
#       (one repo per train entry; one (repo, commit) per val/test entry)
# Output:
#   $CKPT_DIR/CODE2LORA_DIRECT/static_commit_v1/

#SBATCH --job-name=train_code2lora_static_commit
#SBATCH --output=slurm_logs/train_static_commit_%j.out
#SBATCH --error=slurm_logs/train_static_commit_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET/static_commit/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/CODE2LORA_DIRECT/static_commit_v1}"
EPOCHS="${EPOCHS:-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

echo "Splits dir : $SPLITS_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Epochs     : $EPOCHS  | max_seq_len=$MAX_SEQ_LEN  | grad_accum=$GRAD_ACCUM"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs "$EPOCHS" \
    --grad-accum "$GRAD_ACCUM" \
    "$@"

echo "Done: $(date)"
