#!/bin/bash
#SBATCH --job-name=train_fft_4k
#SBATCH --output=slurm_logs/train_fft_4k_%j.out
#SBATCH --error=slurm_logs/train_fft_4k_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: FFT, max_seq=4096, no DRC ====="
echo "Start: $(date)"

# Saves to FFT_4K — does NOT overwrite FFT (2K) checkpoint
python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_4K" \
    --max-seq-length 4096 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-5

echo "Done: $(date)"
