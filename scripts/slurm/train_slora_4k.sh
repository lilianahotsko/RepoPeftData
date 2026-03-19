#!/bin/bash
#SBATCH --job-name=train_slora_4k
#SBATCH --output=slurm_logs/train_slora_4k_%j.out
#SBATCH --error=slurm_logs/train_slora_4k_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Single LoRA, max_seq=4096, no DRC ====="
echo "Start: $(date)"

# Saves to SINGLE_LORA_4K — does NOT overwrite SINGLE_LORA (2K) checkpoint
python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_4K" \
    --rank 16 --lora-alpha 32 \
    --max-seq-length 4096 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4

echo "Done: $(date)"
