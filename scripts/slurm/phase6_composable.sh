#!/bin/bash
#SBATCH --job-name=phase6_compose
#SBATCH --output=slurm_logs/phase6_compose_%j.out
#SBATCH --error=slurm_logs/phase6_compose_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 6: Train composable hypernetwork (weighted strategy -- primary).
# ~10h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 6: Composable hypernetwork (weighted) ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_composable.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET_COMPOSABLE" \
    --composition weighted \
    --rank 16 --alpha 32 --hidden-dim 512 \
    --max-files 30 \
    --epochs 3 --lr 1e-4 \
    --eval-steps 1000 --save-steps 1000

echo "Phase 6 complete: $(date)"
