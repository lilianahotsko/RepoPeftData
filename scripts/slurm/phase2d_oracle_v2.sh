#!/bin/bash
#SBATCH --job-name=p2d_oracv2
#SBATCH --output=slurm_logs/p2d_oracle_v2_%j.out
#SBATCH --error=slurm_logs/p2d_oracle_v2_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

CACHE_V2="$SCRATCH/ORACLE_CONTEXT_CACHE_V2"

echo "===== Oracle Context v2 (function-scoped) ====="
echo "Start: $(date)"

echo "--- Building function-scoped contexts ---"
python baselines/oracle_context/build_context_v2.py \
    --repos-root "$REPOS_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$CACHE_V2"

echo "--- Oracle v2: Eval (cr_test) ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$CACHE_V2" \
    --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_v2_cr_test.json"

echo "--- Oracle v2: Eval (ir_test) ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$CACHE_V2" \
    --split ir_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_v2_ir_test.json"

echo "Done: $(date)"
