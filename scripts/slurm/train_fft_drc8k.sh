#!/bin/bash
#SBATCH --job-name=train_fft_drc8k
#SBATCH --output=slurm_logs/train_fft_drc8k_%j.out
#SBATCH --error=slurm_logs/train_fft_drc8k_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: FFT + DRC v3, max_seq=8192 ====="
echo "Start: $(date)"

python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_DRC_V3_8K" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V3" \
    --max-oracle-tokens 6000 \
    --max-seq-length 8192 \
    --no-eval \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-5

echo "Done: $(date)"
