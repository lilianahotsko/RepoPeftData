#!/bin/bash
# Build ALL splits for the static-commit Code2LoRA pipeline.
# CPU-only; re-extracts QnAs at each (repo, commit) via v2 extractor.
#
# Requires: manifest.tsv + snapshot_embeddings.json
# Output:   $SPLITS_DIR/{train,ir_val,ir_test,cr_val,cr_test,ood_test}.json

#SBATCH --job-name=build_static_commit_splits
#SBATCH --output=slurm_logs/build_static_commit_splits_%j.out
#SBATCH --error=slurm_logs/build_static_commit_splits_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

MANIFEST="${MANIFEST:-$SCRATCH/REPO_DATASET/static_commit/manifest.tsv}"
SNAPSHOT_EMBEDDINGS="${SNAPSHOT_EMBEDDINGS:-$SCRATCH/REPO_DATASET/static_commit/snapshot_embeddings.json}"
PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/static_commit/splits_at_anchor}"

echo "Manifest : $MANIFEST"
echo "Embed    : $SNAPSHOT_EMBEDDINGS"
echo "Parquet  : $PARQUET_DIR"
echo "Out dir  : $OUT_DIR"

python create_dataset/build_static_commit_all_splits.py \
    --manifest "$MANIFEST" \
    --snapshot-embeddings "$SNAPSHOT_EMBEDDINGS" \
    --parquet-dir "$PARQUET_DIR" \
    --out-dir "$OUT_DIR" \
    "$@"

echo "Done: $(date)"
ls -la "$OUT_DIR/"
