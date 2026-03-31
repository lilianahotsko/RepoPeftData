#!/bin/bash
#SBATCH --job-name=pre-drc-v4
#SBATCH --account=def-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pretrained_drc_v4_%j.out
#SBATCH --error=logs/pretrained_drc_v4_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Pretrained + DRC v4 (adaptive, 8k context) ===="
echo "Start: $(date)"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- $SPLIT ---"
    python baselines/oracle_context/test_oracle_context.py \
        --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
        --splits-dir "$SPLITS_DIR" \
        --split "$SPLIT" \
        --max-input-tokens 8192 \
        --output "$BASELINES_DIR/pretrained_drc_v4_8k_${SPLIT}.json"
done

echo -e "\nDone: $(date)"
