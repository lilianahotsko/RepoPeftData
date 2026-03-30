#!/bin/bash
#SBATCH --job-name=slora_drc8k_tuned
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_slora_drc8k_tuned_%j.out
#SBATCH --error=logs/train_slora_drc8k_tuned_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Train Single LoRA + DRC v3 8K (Optuna-tuned HPs) ===="
echo "Start: $(date)"

python baselines/train_with_best_params.py \
    --method lora \
    --hpo-dir "$CKPT_DIR/HPO_SLORA_DRC_V3" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_DRC_V3_8K" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V3" \
    --max-oracle-tokens 6000 \
    --max-seq-length 8192

echo "Done: $(date)"
