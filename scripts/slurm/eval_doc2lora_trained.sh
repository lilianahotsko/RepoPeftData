#!/bin/bash
# ============================================================
# Evaluate trained Doc2LoRA (Qwen2.5-Coder-1.5B) on RepoPeftBench
# Uses DRC oracle context for internalization
#
# Prerequisites:
#   sbatch scripts/slurm/train_doc2lora_qwen.sh  (must complete first)
#
# Submit: sbatch scripts/slurm/eval_doc2lora_trained.sh [checkpoint_step]
# Example: sbatch scripts/slurm/eval_doc2lora_trained.sh 20000
# ============================================================
#SBATCH --job-name=d2l-eval-qwen
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/d2l_eval_trained_%j.out
#SBATCH --error=slurm_logs/d2l_eval_trained_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

export PYTHONPATH="$PWD/doc2lora/src:$PYTHONPATH"

# Checkpoint step (default: latest)
STEP="${1:-20000}"
CKPT_BASE="$CKPT_DIR/DOC2LORA_QWEN"

# Find checkpoint
CKPT="$CKPT_BASE/checkpoint-${STEP}/pytorch_model.bin"
if [ ! -f "$CKPT" ]; then
    # Try to find latest checkpoint
    LATEST=$(ls -d "$CKPT_BASE"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No checkpoint found in $CKPT_BASE"
        exit 1
    fi
    CKPT="$LATEST/pytorch_model.bin"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: pytorch_model.bin not found in $LATEST"
        exit 1
    fi
    echo "Using latest checkpoint: $CKPT"
fi

echo "==== Doc2LoRA (Qwen) Evaluation ===="
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- Evaluating on $SPLIT (with DRC) ---"
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path "$CKPT" \
        --base-model Qwen/Qwen2.5-Coder-1.5B \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --use-oracle \
        --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
        --output "$BASELINES_DIR/doc2lora_trained_drc_${SPLIT}.json" \
        --batch-size 4 \
        --max-oracle-tokens 4096

    echo -e "\n--- Evaluating on $SPLIT (no DRC) ---"
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path "$CKPT" \
        --base-model Qwen/Qwen2.5-Coder-1.5B \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --output "$BASELINES_DIR/doc2lora_trained_${SPLIT}.json" \
        --batch-size 4
done

echo "==== Done: $(date) ===="
