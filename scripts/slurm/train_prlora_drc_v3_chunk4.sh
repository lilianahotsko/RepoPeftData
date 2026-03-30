#!/bin/bash
#SBATCH --job-name=prlora_drc_c4
#SBATCH --output=slurm_logs/prlora_drc_v3_chunk4_%j.out
#SBATCH --error=slurm_logs/prlora_drc_v3_chunk4_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Per-repo LoRA + DRC v3 chunk 4: repos 336-447 ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA_DRC_V3_8K" \
    --repo-offset 336 \
    --limit-repos 112 \
    --eval-split ir_test \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V3" \
    --max-oracle-tokens 6000 \
    --max-seq-length 8192 \
    --no-eval \
    --epochs 3 \
    --no-wandb

echo "Done: $(date)"
