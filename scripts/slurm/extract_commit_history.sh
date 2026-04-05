#!/bin/bash
#SBATCH --job-name=extract_history
#SBATCH --output=slurm_logs/extract_history_%j.out
#SBATCH --error=slurm_logs/extract_history_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Extract Commit History for Code2LoRA-GRU ====="
echo "Start: $(date)"

python create_dataset/extract_commit_history.py \
    --splits-dir "$SPLITS_DIR" \
    --repos-root "$REPOS_ROOT" \
    --output-dir "$SPLITS_DIR" \
    --max-commits 200 \
    --preamble-frac 0.1

echo "Done: $(date)"
