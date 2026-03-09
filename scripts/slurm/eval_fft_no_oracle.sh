#!/bin/bash
#SBATCH --job-name=eval_fft
#SBATCH --output=slurm_logs/eval_fft_%j.out
#SBATCH --error=slurm_logs/eval_fft_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate FFT (no-oracle) checkpoint on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: FFT (no-oracle) ====="
echo "Start: $(date)"

MODEL_PATH="$CKPT_DIR/FFT/final"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "--- cr_test ---"
python baselines/finetuned/test_finetuned.py \
    --model-path "$MODEL_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --output "$BASELINES_DIR/fft_no_oracle_cr_test.json"

echo "--- ir_test ---"
python baselines/finetuned/test_finetuned.py \
    --model-path "$MODEL_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split ir_test \
    --output "$BASELINES_DIR/fft_no_oracle_ir_test.json"

echo "Done: $(date)"
