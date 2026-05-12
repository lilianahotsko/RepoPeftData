#!/bin/bash
# Merge sharded suite JSONs into single per-suite files.
#
#SBATCH --job-name=merge_split_shards
#SBATCH --output=slurm_logs/merge_split_shards_%j.out
#SBATCH --error=slurm_logs/merge_split_shards_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET/static_commit/splits_at_anchor}"
SUITES="${SUITES:-ir_test cr_val}"

# shellcheck disable=SC2086
python create_dataset/merge_split_shards.py \
    --splits-dir "$SPLITS_DIR" \
    --suites $SUITES

ls -la "$SPLITS_DIR/"
