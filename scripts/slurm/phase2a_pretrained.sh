#!/bin/bash
#SBATCH --job-name=p2a_pretrained
#SBATCH --output=slurm_logs/p2a_pretrained_%j.out
#SBATCH --error=slurm_logs/p2a_pretrained_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Pretrained baseline ====="
echo "Start: $(date)"

echo "--- Pretrained (cr_test) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_cr_test.json"

echo "--- Pretrained (ir_test) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_ir_test.json"

echo "Done: $(date)"
