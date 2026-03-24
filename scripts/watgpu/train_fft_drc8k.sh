#!/bin/bash
#SBATCH --job-name=fft_drc8k
#SBATCH --output=slurm_logs/fft_drc8k_%j.out
#SBATCH --error=slurm_logs/fft_drc8k_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

source scripts/watgpu/common.sh

echo "===== Train: FFT + DRC, max_seq=8192 ====="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FFT_DRC8K" \
    --use-oracle \
    --oracle-cache-dir "$ORACLE_CACHE_DIR" \
    --max-seq-length 8192 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-5

echo "Done: $(date)"
