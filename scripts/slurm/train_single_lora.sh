#!/bin/bash
#SBATCH --job-name=train_slora
#SBATCH --output=slurm_logs/train_slora_%j.out
#SBATCH --error=slurm_logs/train_slora_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Single LoRA ====="
echo "Start: $(date)"

python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA" \
    --max-seq-length 2048 \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4

echo "Done: $(date)"
