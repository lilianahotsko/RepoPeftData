#!/bin/bash
#SBATCH --job-name=p2d_oracle
#SBATCH --output=slurm_logs/p2d_oracle_%j.out
#SBATCH --error=slurm_logs/p2d_oracle_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Oracle Context baseline ====="
echo "Start: $(date)"

echo "--- Building import-resolved contexts (skips cached) ---"
python baselines/oracle_context/build_context.py \
    --repos-root "$REPOS_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE"

echo "--- Oracle Context: Eval (cr_test) ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE" \
    --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_cr_test.json"

echo "Done: $(date)"
