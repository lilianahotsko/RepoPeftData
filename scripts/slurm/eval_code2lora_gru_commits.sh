#!/bin/bash
#SBATCH --job-name=eval_gru_diff
#SBATCH --output=slurm_logs/eval_gru_diff_%j.out
#SBATCH --error=slurm_logs/eval_gru_diff_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian
#
# Evaluate the diff-based Code2LoRA-GRU checkpoint.
# Uses the existing eval_code2lora_gru.py with the diff-trained checkpoint.
# The eval script loads from the .pt file, which is architecture-agnostic.

source scripts/slurm/common.sh
mkdir -p slurm_logs

SUFFIX="${SUFFIX:-default}"
CKPT="${CKPT:-$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}}"

echo "===== Eval: Code2LoRA-GRU (commit-level, diff-based) ====="
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

for SPLIT in cr_test ir_test; do
    echo ""
    echo "--- Evaluating on $SPLIT ---"
    python hypernetwork/eval_code2lora_gru.py \
        --checkpoint "$CKPT" \
        --splits-dir "$SPLITS_DIR" \
        --split "$SPLIT" \
        --mode standard
done

echo "Done: $(date)"
