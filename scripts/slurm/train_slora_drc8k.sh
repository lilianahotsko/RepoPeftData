#!/bin/bash
#SBATCH --job-name=train_slora_drc8k
#SBATCH --output=slurm_logs/train_slora_drc8k_%j.out
#SBATCH --error=slurm_logs/train_slora_drc8k_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Single LoRA + DRC v2, max_seq=8192 ====="
echo "Start: $(date)"

# Saves to SINGLE_LORA_DRC8K — does NOT overwrite SINGLE_LORA_ORACLE (4K v1 cache)
python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_DRC8K" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V2" \
    --rank 16 --lora-alpha 32 \
    --max-seq-length 8192 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4

echo "Done: $(date)"
