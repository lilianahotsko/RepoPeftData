#!/bin/bash
#SBATCH --job-name=train_prlora_orc
#SBATCH --output=slurm_logs/train_prlora_oracle_%j.out
#SBATCH --error=slurm_logs/train_prlora_oracle_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Per-repo LoRA + Oracle ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA_ORACLE" \
    --limit-repos 30 \
    --eval-split ir_test \
    --use-oracle \
    --max-seq-length 4096 \
    --epochs 3 \
    --no-wandb

echo "Done: $(date)"
