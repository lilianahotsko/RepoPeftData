#!/bin/bash
#SBATCH --job-name=phase8_scale
#SBATCH --output=slurm_logs/phase8_scale_%j.out
#SBATCH --error=slurm_logs/phase8_scale_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 8: Scale experiments with 0.5B and 3B base models.
# Uses structured splits.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 8: Scale experiments ====="
echo "Start: $(date)"

# --- 0.5B ---
echo "--- Scale: 0.5B ---"
MODEL_05B="Qwen/Qwen2.5-Coder-0.5B"
OUT_05B="$CKPT_DIR/HYPERNET/scale_0.5B"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --model-name "$MODEL_05B" \
    --output-dir "$OUT_05B" \
    --rank 16 --alpha 32 --hidden-dim 512 \
    --epochs 3 --lr 1e-4 \
    --eval-steps 1000 --save-steps 1000

python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$OUT_05B" \
    --model-name "$MODEL_05B" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test

python baselines/pretrained/test_qwen_coder.py \
    --model-name "$MODEL_05B" \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --output "$BASELINES_DIR/pretrained_0.5B_cr_test.json"

# --- 3B ---
echo "--- Scale: 3B ---"
MODEL_3B="Qwen/Qwen2.5-Coder-3B"
OUT_3B="$CKPT_DIR/HYPERNET/scale_3B"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --model-name "$MODEL_3B" \
    --output-dir "$OUT_3B" \
    --rank 16 --alpha 32 --hidden-dim 512 \
    --epochs 3 --lr 1e-4 \
    --eval-steps 1000 --save-steps 1000

python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$OUT_3B" \
    --model-name "$MODEL_3B" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test

python baselines/pretrained/test_qwen_coder.py \
    --model-name "$MODEL_3B" \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --output "$BASELINES_DIR/pretrained_3B_cr_test.json"

echo "Phase 8 complete: $(date)"
