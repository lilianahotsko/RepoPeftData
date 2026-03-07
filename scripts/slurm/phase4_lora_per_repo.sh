#!/bin/bash
#SBATCH --job-name=phase4_perrepo
#SBATCH --output=slurm_logs/phase4_perrepo_%j.out
#SBATCH --error=slurm_logs/phase4_perrepo_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 4: Per-repo LoRA baseline.
# Train a LoRA adapter per repo (sample of 30 repos), evaluate on ir_test.
# ~6h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 4: Per-repo LoRA baseline ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --limit-repos 30 \
    --eval-split ir_test \
    --epochs 3 \
    --no-wandb

echo "Phase 4 complete: $(date)"
