#!/bin/bash
#SBATCH --job-name=eval_slora
#SBATCH --output=slurm_logs/eval_slora_%j.out
#SBATCH --error=slurm_logs/eval_slora_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate Single LoRA (no-oracle) checkpoint on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: Single LoRA (no-oracle) ====="
echo "Start: $(date)"

ADAPTER_PATH="$CKPT_DIR/SINGLE_LORA/adapter"
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    exit 1
fi

echo "--- cr_test ---"
python baselines/single_lora/test_single_lora.py \
    --adapter "$ADAPTER_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --output "$BASELINES_DIR/single_lora_no_oracle_cr_test.json"

echo "--- ir_test ---"
python baselines/single_lora/test_single_lora.py \
    --adapter "$ADAPTER_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split ir_test \
    --output "$BASELINES_DIR/single_lora_no_oracle_ir_test.json"

echo "Done: $(date)"
