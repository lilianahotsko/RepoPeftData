#!/bin/bash
#SBATCH --job-name=prlora_c4
#SBATCH --output=slurm_logs/prlora_chunk4_%j.out
#SBATCH --error=slurm_logs/prlora_chunk4_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Per-repo LoRA chunk 4: repos 336-447 ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --repo-offset 336 \
    --limit-repos 111 \
    --eval-split ir_test \
    --epochs 3 \
    --no-wandb

echo "Done: $(date)"
