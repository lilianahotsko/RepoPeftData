#!/bin/bash
# ============================================================
# Doc-to-LoRA evaluation (pretrained, no oracle context)
# Base model: Gemma-2-2B-IT (D2L's native architecture)
#
# Prerequisites:
#   bash scripts/slurm/download_doc2lora.sh
#
# Submit: sbatch scripts/slurm/eval_doc2lora.sh
# ============================================================
#SBATCH --job-name=d2l-eval
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/doc2lora_eval_%j.out
#SBATCH --error=logs/doc2lora_eval_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

export PYTHONPATH="$PWD/doc2lora/src:$PYTHONPATH"

CKPT="doc2lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    echo "Run: bash scripts/slurm/download_doc2lora.sh"
    exit 1
fi

echo "==== Doc-to-LoRA Evaluation (no oracle) ===="
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- Evaluating on $SPLIT ---"
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path "$CKPT" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --output "$BASELINES_DIR/doc2lora_${SPLIT}.json" \
        --batch-size 4
done

echo "==== Done: $(date) ===="
