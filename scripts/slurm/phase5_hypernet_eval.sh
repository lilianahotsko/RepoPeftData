#!/bin/bash
#SBATCH --job-name=phase5_hneval
#SBATCH --output=slurm_logs/phase5_hneval_%j.out
#SBATCH --error=slurm_logs/phase5_hneval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 5: Evaluate existing hypernetwork checkpoint on structured splits.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 5: Hypernetwork evaluation ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$CKPT_DIR/HYPERNET/full_repos" \
    --splits-dir "$SPLITS_DIR" \
    --splits cr_test cr_val ir_test ir_val

echo "Phase 5 complete: $(date)"
