#!/bin/bash
#SBATCH --job-name=reeval_train
#SBATCH --output=slurm_logs/reeval_trained_%j.out
#SBATCH --error=slurm_logs/reeval_trained_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Re-evaluate FFT + Single LoRA r=64 with fixed postprocess_prediction.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Re-eval: Trained methods (fixed metrics) ====="
echo "Start: $(date)"

echo ""
echo "========== FFT =========="
MODEL_PATH="$CKPT_DIR/FFT/final"
if [[ -d "$MODEL_PATH" ]]; then
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
else
    echo "WARN: FFT model not found at $MODEL_PATH"
fi

echo ""
echo "========== SINGLE LORA r=64 =========="
ADAPTER_PATH="$CKPT_DIR/SINGLE_LORA/adapter"
if [[ -d "$ADAPTER_PATH" ]]; then
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
else
    echo "WARN: Single LoRA adapter not found at $ADAPTER_PATH"
fi

echo ""
echo "Done: $(date)"
