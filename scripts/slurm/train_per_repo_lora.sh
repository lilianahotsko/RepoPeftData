#!/bin/bash
#SBATCH --job-name=train_prlora
#SBATCH --output=slurm_logs/train_prlora_%j.out
#SBATCH --error=slurm_logs/train_prlora_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Per-repo LoRA ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --limit-repos 10 \
    --eval-split ir_test \
    --epochs 3 \
    --no-wandb

echo "Done: $(date)"
