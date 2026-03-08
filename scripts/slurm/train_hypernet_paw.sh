#!/bin/bash
#SBATCH --job-name=train_hpaw
#SBATCH --output=slurm_logs/train_hpaw_%j.out
#SBATCH --error=slurm_logs/train_hpaw_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: PAW Hypernetwork ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_paw.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET_PAW/no_oracle" \
    --max-seq-len 8192 \
    --num-bases 16 \
    --trunk-depth 2 \
    --lora-mapper-fp32 \
    --epochs 3 --grad-accum 8

echo "Done: $(date)"
