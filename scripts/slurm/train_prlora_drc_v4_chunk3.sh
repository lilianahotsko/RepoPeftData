#!/bin/bash
#SBATCH --job-name=prlv4_c3
#SBATCH --output=slurm_logs/prlora_drc_v4_chunk3_%j.out
#SBATCH --error=slurm_logs/prlora_drc_v4_chunk3_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Per-repo LoRA + DRC v4 (adaptive) chunk 3: repos 224+ ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA_DRC_V4_8K" \
    --repo-offset 224 \
    --limit-repos 112 \
    --eval-split ir_test \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
    --max-oracle-tokens 4096 \
    --max-seq-length 8192 \
    --epochs 1 \
    --no-eval \
    --no-wandb

echo "Done: $(date)"
