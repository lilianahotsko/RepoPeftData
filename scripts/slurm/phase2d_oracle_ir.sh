#!/bin/bash
#SBATCH --job-name=p2d_oracle_ir
#SBATCH --output=slurm_logs/p2d_oracle_ir_%j.out
#SBATCH --error=slurm_logs/p2d_oracle_ir_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Oracle Context baseline (IR test) ====="
echo "Start: $(date)"

python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE" \
    --split ir_test_structured \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_ir_test_structured.json"

echo "Done: $(date)"
