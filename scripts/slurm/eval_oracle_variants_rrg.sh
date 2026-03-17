#!/bin/bash
#SBATCH --job-name=eval_oracle
#SBATCH --output=slurm_logs/eval_oracle_variants_%j.out
#SBATCH --error=slurm_logs/eval_oracle_variants_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Evaluate all +Oracle trained variants on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: All +Oracle variants ====="
echo "Start: $(date)"

echo ""
echo "========== FFT + ORACLE =========="
MODEL_PATH="$CKPT_DIR/FFT_ORACLE/final"
if [[ -d "$MODEL_PATH" ]]; then
    for split in cr_test ir_test; do
        echo "--- $split ---"
        python baselines/finetuned/test_finetuned.py \
            --model-path "$MODEL_PATH" \
            --splits-dir "$SPLITS_DIR" \
            --split $split \
            --output "$BASELINES_DIR/fft_oracle_${split}.json"
    done
else
    echo "WARN: FFT+Oracle not found at $MODEL_PATH"
fi

echo ""
echo "========== SINGLE LORA + ORACLE =========="
ADAPTER_PATH="$CKPT_DIR/SINGLE_LORA_ORACLE/adapter"
if [[ -d "$ADAPTER_PATH" ]]; then
    for split in cr_test ir_test; do
        echo "--- $split ---"
        python baselines/single_lora/test_single_lora.py \
            --adapter "$ADAPTER_PATH" \
            --splits-dir "$SPLITS_DIR" \
            --split $split \
            --output "$BASELINES_DIR/single_lora_oracle_${split}.json"
    done
else
    echo "WARN: sLoRA+Oracle adapter not found at $ADAPTER_PATH"
fi

echo ""
echo "========== CODE2LORA + ORACLE =========="
CKPT_PATH="$CKPT_DIR/HYPERNET/oracle"
if [[ -d "$CKPT_PATH" ]]; then
    python hypernetwork/hypernetwork_sampled_test.py \
        --checkpoint "$CKPT_PATH" \
        --splits-dir "$SPLITS_DIR" \
        --splits cr_test ir_test

    RESULTS_BASE="${CKPT_PATH}_results"
    for split in cr_test ir_test; do
        if [[ -f "$RESULTS_BASE/$split/results.json" ]]; then
            cp "$RESULTS_BASE/$split/results.json" "$BASELINES_DIR/hypernet_oracle_${split}.json"
            echo "Copied $split results"
        fi
    done
else
    echo "WARN: Hypernet+Oracle not found at $CKPT_PATH"
fi

echo ""
echo "========== CODE2LORA PAW + ORACLE =========="
CKPT_PATH="$CKPT_DIR/HYPERNET_PAW/oracle"
if [[ -f "$CKPT_PATH/lora_mapper_best.pt" ]]; then
    python hypernetwork/hypernetwork_paw_test.py \
        --checkpoint "$CKPT_PATH" \
        --splits-dir "$SPLITS_DIR" \
        --splits cr_test ir_test

    RESULTS_BASE="${CKPT_PATH}_results"
    for split in cr_test ir_test; do
        if [[ -f "$RESULTS_BASE/$split/results.json" ]]; then
            cp "$RESULTS_BASE/$split/results.json" "$BASELINES_DIR/hypernet_paw_oracle_${split}.json"
            echo "Copied $split results"
        fi
    done
else
    echo "WARN: PAW+Oracle not found at $CKPT_PATH"
fi

echo ""
echo "========== SINGLE LORA R16 (retrained) =========="
ADAPTER_PATH="$CKPT_DIR/SINGLE_LORA_R16/adapter"
if [[ -d "$ADAPTER_PATH" ]]; then
    for split in cr_test ir_test; do
        echo "--- $split ---"
        python baselines/single_lora/test_single_lora.py \
            --adapter "$ADAPTER_PATH" \
            --splits-dir "$SPLITS_DIR" \
            --split $split \
            --output "$BASELINES_DIR/single_lora_r16_${split}.json"
    done
else
    echo "WARN: sLoRA R16 adapter not found at $ADAPTER_PATH"
fi

echo ""
echo "Done: $(date)"
