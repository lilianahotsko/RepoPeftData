#!/bin/bash
#SBATCH --job-name=prlora_c2
#SBATCH --output=slurm_logs/prlora_chunk2_%j.out
#SBATCH --error=slurm_logs/prlora_chunk2_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Per-repo LoRA chunk 2: repos 112-223 ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --repo-offset 112 \
    --limit-repos 112 \
    --eval-split ir_test \
    --epochs 3 \
    --no-wandb

echo "Done: $(date)"
