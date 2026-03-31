#!/bin/bash
#SBATCH --job-name=fft_drc_v4
#SBATCH --account=def-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_fft_drc_v4_%j.out
#SBATCH --error=slurm_logs/train_fft_drc_v4_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

echo "===== Train: FFT + DRC v4 (adaptive), max_seq=8192 ====="
echo "Start: $(date)"

python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_DRC_V4_8K" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
    --max-oracle-tokens 4096 \
    --max-seq-length 8192 \
    --no-eval \
    --epochs 1 --batch-size 1 --grad-accum 32 --lr 2e-5

echo "Done: $(date)"
