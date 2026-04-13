#!/bin/bash
#SBATCH --job-name=build_commit_db
#SBATCH --output=slurm_logs/build_commit_db_%j.out
#SBATCH --error=slurm_logs/build_commit_db_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian
#
# CPU-only: builds SQLite commits + assertions DB with blame-based mapping.
# Much faster than v1 (no per-file git show, no production_code column).
#
# Submit from repo root:  sbatch scripts/slurm/build_commit_assertion_db.sh
#
# Optional env overrides:
#   DB_PATH           (default: $SPLITS_DIR/commits_assertions.db)
#   LIMIT_REPOS       e.g. 10 for a smoke test
#   MAX_COMMITS       per repo sampling (default 200)
#   DIFF_MODE         single_commit (default) | inter_sample
#   NO_GRU_SPLITS=1   use train/cr_*/ir_*.json only, not gru_*.json

source scripts/slurm/common.sh
mkdir -p slurm_logs

DB_PATH="${DB_PATH:-$SPLITS_DIR/commits_assertions.db}"
MAX_COMMITS="${MAX_COMMITS:-200}"
DIFF_MODE="${DIFF_MODE:-single_commit}"

EXTRA=()
if [ -n "$LIMIT_REPOS" ]; then
  EXTRA+=(--limit-repos "$LIMIT_REPOS")
fi
if [ "${NO_GRU_SPLITS:-0}" = "1" ]; then
  EXTRA+=(--no-gru-splits)
fi

echo "===== Build: commits + assertions SQLite (v2) ====="
echo "Splits dir:    $SPLITS_DIR"
echo "Repos root:    $REPOS_ROOT"
echo "DB path:       $DB_PATH"
echo "Max commits:   $MAX_COMMITS"
echo "Diff mode:     $DIFF_MODE"
echo "Extra args:    ${EXTRA[*]:-(none)}"
echo "Start: $(date)"

python create_dataset/build_commit_assertion_db.py \
    --splits-dir "$SPLITS_DIR" \
    --repos-root "$REPOS_ROOT" \
    --db-path "$DB_PATH" \
    --max-commits "$MAX_COMMITS" \
    --diff-mode "$DIFF_MODE" \
    "${EXTRA[@]}"

echo "Done: $(date)"
