#!/bin/bash
#SBATCH --job-name=slora_8k
#SBATCH --output=slurm_logs/slora_8k_%j.out
#SBATCH --error=slurm_logs/slora_8k_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

source scripts/watgpu/common.sh

echo "===== Train: Single LoRA, max_seq=8192, no DRC ====="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_8K" \
    --rank 16 --lora-alpha 32 \
    --max-seq-length 8192 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4

echo "Done: $(date)"
