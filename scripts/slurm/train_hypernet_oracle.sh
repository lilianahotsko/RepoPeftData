#!/bin/bash
#SBATCH --job-name=train_hnet_orc
#SBATCH --output=slurm_logs/train_hnet_oracle_%j.out
#SBATCH --error=slurm_logs/train_hnet_oracle_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Hypernetwork + Oracle ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET/oracle" \
    --use-oracle \
    --max-seq-len 8192 \
    --epochs 3 --grad-accum 8

echo "Done: $(date)"
