#!/bin/bash
#SBATCH --job-name=eval_drc_v4
#SBATCH --account=def-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/eval_drc_v4_%j.out
#SBATCH --error=slurm_logs/eval_drc_v4_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

ORACLE_V4="$SCRATCH/ORACLE_CONTEXT_CACHE_V4"

# ── FFT + DRC v4 ────────────────────────────────────────────────────────────
FFT_MODEL="$CKPT_DIR/FFT_DRC_V4_8K/final"
if [[ -d "$FFT_MODEL" ]]; then
    echo "========== FFT + DRC v4 =========="
    for SPLIT in cr_test ir_test; do
        echo "--- $SPLIT ---"
        python baselines/finetuned/test_finetuned.py \
            --model-path "$FFT_MODEL" \
            --splits-dir "$SPLITS_DIR" \
            --split "$SPLIT" \
            --oracle-cache-dir "$ORACLE_V4" \
            --max-input-tokens 8192 \
            --output "$BASELINES_DIR/fft_drc_v4_${SPLIT}.json"
    done
else
    echo "SKIP: FFT model not found at $FFT_MODEL"
fi

# ── Single LoRA + DRC v4 ────────────────────────────────────────────────────
SLORA_ADAPTER="$CKPT_DIR/SINGLE_LORA_DRC_V4_8K/adapter"
if [[ -d "$SLORA_ADAPTER" ]]; then
    echo ""
    echo "========== Single LoRA + DRC v4 =========="
    for SPLIT in cr_test ir_test; do
        echo "--- $SPLIT ---"
        python baselines/single_lora/test_single_lora.py \
            --adapter "$SLORA_ADAPTER" \
            --splits-dir "$SPLITS_DIR" \
            --split "$SPLIT" \
            --oracle-cache-dir "$ORACLE_V4" \
            --max-input-tokens 8192 \
            --output "$BASELINES_DIR/slora_drc_v4_${SPLIT}.json"
    done
else
    echo "SKIP: LoRA adapter not found at $SLORA_ADAPTER"
fi

echo ""
echo "Done: $(date)"
