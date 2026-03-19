#!/bin/bash
#SBATCH --job-name=eval_new_models
#SBATCH --output=slurm_logs/eval_new_models_%j.out
#SBATCH --error=slurm_logs/eval_new_models_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Evaluate 4 newly trained models on cr_test + ir_test:
#   FFT_4K         (no DRC)  → fft4k_{cr,ir}_test.json
#   SINGLE_LORA_4K (no DRC)  → slora4k_{cr,ir}_test.json
#   FFT_DRC4K      (with DRC) → fft_drc4k_{cr,ir}_test.json
#   SINGLE_LORA_DRC4K (with DRC) → slora_drc4k_{cr,ir}_test.json

source scripts/slurm/common.sh
mkdir -p slurm_logs

ORACLE_CACHE="$SCRATCH/ORACLE_CONTEXT_CACHE_V2"

echo "===== Eval: FFT_4K, sLoRA_4K, FFT_DRC4K, sLoRA_DRC4K ====="
echo "Start: $(date)"

# ── FFT 4K (no DRC) ─────────────────────────────────────────────────────────
echo ""
echo "========== FFT 4K (no DRC) =========="
MODEL="$CKPT_DIR/FFT_4K/final"
for split in cr_test ir_test; do
    echo "--- $split ---"
    python baselines/finetuned/test_finetuned.py \
        --model-path "$MODEL" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --output "$BASELINES_DIR/fft4k_${split}.json"
done

# ── sLoRA 4K (no DRC) ───────────────────────────────────────────────────────
echo ""
echo "========== sLoRA 4K (no DRC) =========="
ADAPTER="$CKPT_DIR/SINGLE_LORA_4K/adapter"
for split in cr_test ir_test; do
    echo "--- $split ---"
    python baselines/single_lora/test_single_lora.py \
        --adapter "$ADAPTER" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --output "$BASELINES_DIR/slora4k_${split}.json"
done

# ── FFT DRC 4K (with oracle DRC context at test time) ───────────────────────
echo ""
echo "========== FFT DRC4K (with DRC) =========="
MODEL="$CKPT_DIR/FFT_DRC4K/final"
for split in cr_test ir_test; do
    echo "--- $split ---"
    python baselines/finetuned/test_finetuned.py \
        --model-path "$MODEL" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --oracle-cache-dir "$ORACLE_CACHE" \
        --output "$BASELINES_DIR/fft_drc4k_${split}.json"
done

# ── sLoRA DRC 4K (with oracle DRC context at test time) ─────────────────────
echo ""
echo "========== sLoRA DRC4K (with DRC) =========="
ADAPTER="$CKPT_DIR/SINGLE_LORA_DRC4K/adapter"
for split in cr_test ir_test; do
    echo "--- $split ---"
    python baselines/single_lora/test_single_lora.py \
        --adapter "$ADAPTER" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --oracle-cache-dir "$ORACLE_CACHE" \
        --output "$BASELINES_DIR/slora_drc4k_${split}.json"
done

echo ""
echo "Done: $(date)"
