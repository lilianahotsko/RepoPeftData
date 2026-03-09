#!/bin/bash
#SBATCH --job-name=train_fft_orc
#SBATCH --output=slurm_logs/train_fft_oracle_%j.out
#SBATCH --error=slurm_logs/train_fft_oracle_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Full Fine-Tuning + Oracle ====="
echo "Start: $(date)"

python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_ORACLE" \
    --use-oracle \
    --max-seq-length 4096 \
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 2e-5

echo "Done: $(date)"
