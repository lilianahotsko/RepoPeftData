#!/bin/bash
#SBATCH --job-name=phase7_compvar
#SBATCH --output=slurm_logs/phase7_compvar_%j.out
#SBATCH --error=slurm_logs/phase7_compvar_%j.err
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 7: Composable variants (additive, gated) + incremental adaptation.
# Uses structured splits.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 7: Composable variants + incremental ====="
echo "Start: $(date)"

# --- Additive ---
echo "--- Composition: additive ---"
python hypernetwork/hypernetwork_composable.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET_COMPOSABLE" \
    --composition additive \
    --rank 16 --alpha 32 --hidden-dim 512 \
    --max-files 30 \
    --epochs 3 --lr 1e-4

# --- Gated ---
echo "--- Composition: gated ---"
python hypernetwork/hypernetwork_composable.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET_COMPOSABLE" \
    --composition gated \
    --rank 16 --alpha 32 --hidden-dim 512 \
    --max-files 30 \
    --epochs 3 --lr 1e-4

# --- Incremental adaptation experiment ---
echo "--- Incremental adaptation ---"
python hypernetwork/eval_incremental.py \
    --checkpoint "$CKPT_DIR/HYPERNET_COMPOSABLE_weighted" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --limit-repos 20 \
    --max-files-to-test 20

echo "Phase 7 complete: $(date)"
