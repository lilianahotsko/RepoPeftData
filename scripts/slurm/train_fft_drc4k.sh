#!/bin/bash
#SBATCH --job-name=train_fft_drc4k
#SBATCH --output=slurm_logs/train_fft_drc4k_%j.out
#SBATCH --error=slurm_logs/train_fft_drc4k_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: FFT + DRC v2, max_seq=4096 ====="
echo "Start: $(date)"

# Saves to FFT_DRC4K — 8K OOM'd; 4K matches no-DRC FFT_4K context window
python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_DRC4K" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V2" \
    --max-seq-length 4096 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-5

echo "Done: $(date)"
