#!/bin/bash
#SBATCH --job-name=train_slora_orc
#SBATCH --output=slurm_logs/train_slora_oracle_%j.out
#SBATCH --error=slurm_logs/train_slora_oracle_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Single LoRA + Oracle ====="
echo "Start: $(date)"

python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_ORACLE" \
    --use-oracle \
    --max-seq-length 4096 \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4

echo "Done: $(date)"
