#!/bin/bash
#SBATCH --job-name=hpo_slora
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpo_slora_drc_v3_%j.out
#SBATCH --error=logs/hpo_slora_drc_v3_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Optuna HPO: Single LoRA + DRC v3 8K ===="
echo "Start: $(date)"

python baselines/hpo_optuna.py \
    --method lora \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HPO_SLORA_DRC_V3" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V3" \
    --max-oracle-tokens 6000 \
    --max-seq-length 8192 \
    --n-trials 20

echo "Done: $(date)"
