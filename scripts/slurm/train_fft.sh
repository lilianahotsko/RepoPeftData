#!/bin/bash
#SBATCH --job-name=train_fft
#SBATCH --output=slurm_logs/train_fft_%j.out
#SBATCH --error=slurm_logs/train_fft_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Full Fine-Tuning ====="
echo "Start: $(date)"

python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT" \
    --max-seq-length 2048 \
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 2e-5

echo "Done: $(date)"
