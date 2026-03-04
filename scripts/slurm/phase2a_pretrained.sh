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

echo "--- Pretrained (cr_test_structured) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split cr_test_structured \
    --output "$BASELINES_DIR/pretrained_cr_test_structured.json"

echo "--- Pretrained (ir_test_structured) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split ir_test_structured \
    --output "$BASELINES_DIR/pretrained_ir_test_structured.json"

echo "Done: $(date)"
