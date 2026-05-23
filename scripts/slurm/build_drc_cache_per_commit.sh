#!/bin/bash
#SBATCH --job-name=build_drc_per_commit
#SBATCH --output=slurm_logs/build_drc_per_commit_%A_%a.out
#SBATCH --error=slurm_logs/build_drc_per_commit_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

# Precompute per-(repo_id, commit_sha) dependency-resolved-context (DRC)
# caches for the v2 commit-derived QnA suites. CPU-only -- no GPU needed.
# Sharding is by REPO (round-robin over sorted repo_ids); each array task
# materializes ``git archive | tar -x`` into SLURM_TMPDIR and runs the v2
# function-scoped DRC pipeline.
#
# Usage examples
# --------------
#   # 4-way sharded build over all four suites:
#   NUM_SHARDS=4 SUITES="cr_val cr_test ir_val ir_test" \
#     sbatch --array=0-3 scripts/slurm/build_drc_cache_per_commit.sh
#
#   # Single-shard smoke test on 1 repo:
#   LIMIT=1 NUM_SHARDS=1 SUITES="cr_val" \
#     sbatch --array=0 scripts/slurm/build_drc_cache_per_commit.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
REPOS_ROOT="${REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories}"
OUT_DIR="${OUT_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS}"
TMP_ROOT="${TMP_ROOT:-${SLURM_TMPDIR:-/tmp}}"
SUITES_STR="${SUITES:-cr_val cr_test ir_val ir_test}"
read -r -a SUITES <<< "$SUITES_STR"

LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"

echo "===== build DRC per-commit cache  shard ${SHARD_I}/${NUM_SHARDS} ====="
echo "QnA dir   : $QNA_DIR"
echo "Suites    : ${SUITES[*]}"
echo "Repos root: $REPOS_ROOT"
echo "Out dir   : $OUT_DIR"
echo "Tmp root  : $TMP_ROOT"
echo "Start     : $(date)"

EXTRA_ARGS=()
if [ "$LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

python evaluation/build_drc_cache_per_commit.py \
    --qna-dir "$QNA_DIR" \
    --suites "${SUITES[@]}" \
    --repos-root "$REPOS_ROOT" \
    --out-dir "$OUT_DIR" \
    --tmp-root "$TMP_ROOT" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
