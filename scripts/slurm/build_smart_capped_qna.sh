#!/bin/bash
# Build the smart-capped GRU-commit QnA parquet snapshot.
#
# Drops trivial-target rows (~17%), then round-robin samples per
# (test_file, test_function) inside each commit, capped at --max-per-file
# and --max-per-commit. Only in_repo_split=='train' rows are touched;
# val/test rows pass through verbatim.
#
# Usage:
#   sbatch scripts/slurm/build_smart_capped_qna.sh
#
# Optional env overrides:
#   PARQUET_DIR (default: $SCRATCH/REPO_DATASET/commit_parquet_hf)
#   OUT_DIR     (default: $SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap)
#   MAX_PER_FILE   (default: 4)
#   MAX_PER_COMMIT (default: 8)
#   MIN_TARGET_CHARS (default: 4)
#   SEED        (default: 3407)
#   BATCH_SIZE  (default: 2048)

#SBATCH --job-name=build_smartcap_qna
#SBATCH --output=slurm_logs/build_smartcap_qna_%j.out
#SBATCH --error=slurm_logs/build_smartcap_qna_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap}"
MAX_PER_FILE="${MAX_PER_FILE:-4}"
MAX_PER_COMMIT="${MAX_PER_COMMIT:-8}"
MIN_TARGET_CHARS="${MIN_TARGET_CHARS:-4}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-2048}"

echo "===== Smart-cap GRU-commit QnA parquet ====="
echo "In:                  $PARQUET_DIR"
echo "Out:                 $OUT_DIR"
echo "Max per file:        $MAX_PER_FILE"
echo "Max per commit:      $MAX_PER_COMMIT"
echo "Min target chars:    $MIN_TARGET_CHARS"
echo "Seed:                $SEED"
echo "Batch size:          $BATCH_SIZE"
echo "Start: $(date)"

python create_dataset/build_smart_capped_qna_parquet.py \
    --in-dir "$PARQUET_DIR" \
    --out-dir "$OUT_DIR" \
    --max-per-file "$MAX_PER_FILE" \
    --max-per-commit "$MAX_PER_COMMIT" \
    --min-target-chars "$MIN_TARGET_CHARS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --splits train

echo "Done: $(date)"
ls -lah "$OUT_DIR/qna/"
