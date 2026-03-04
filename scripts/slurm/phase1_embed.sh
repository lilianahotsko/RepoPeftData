#!/bin/bash
#SBATCH --job-name=phase1_embed
#SBATCH --output=slurm_logs/phase1_embed_%j.out
#SBATCH --error=slurm_logs/phase1_embed_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 1: Re-embed repos with file-level embeddings, then recreate splits.
# ~2h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 1: Re-embed repos with file-level embeddings ====="
echo "Start: $(date)"

python create_dataset/embed_repos.py \
    --repos-root "$REPOS_ROOT" \
    --overwrite

echo "Recreating splits with file embeddings..."
python create_dataset/create_splits.py \
    --repos-root "$REPOS_ROOT" \
    --out-dir "$SPLITS_DIR"

echo "Phase 1 complete: $(date)"
