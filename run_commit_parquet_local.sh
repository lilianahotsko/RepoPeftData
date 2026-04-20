#!/bin/bash
# Build the commit-level Parquet dataset on a single host using N local
# worker processes. Intended for running either on the login node (short
# tests / small slices) or inside an `salloc` / sbatch compute allocation
# for the real thing.
#
# Usage:
#   # 8 workers, full dataset, resume where shard 0 left off:
#   ./run_commit_parquet_local.sh 8
#
#   # 16 workers inside a compute allocation:
#   salloc --time=04:00:00 --cpus-per-task=16 --mem=32G --account=def-yuntian
#   ./run_commit_parquet_local.sh 16
#
#   # Extra flags are forwarded to build_commit_parquet_db.py, e.g.:
#   ./run_commit_parquet_local.sh 8 --limit-repos 20 --out-dir /tmp/smoke

set -euo pipefail

N="${1:-8}"
shift || true

: "${SCRATCH:?SCRATCH must be set}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
# shellcheck disable=SC1091
source "$SCRATCH/venvs/qwen-cu126-py312/bin/activate"

cd "$(dirname "$0")"
mkdir -p logs

echo "Launching ${N} local workers at $(date)"
echo "Out dir: $SCRATCH/REPO_DATASET/commit_parquet"

python create_dataset/build_commit_parquet_db.py \
    --splits-dir "$SCRATCH/REPO_DATASET" \
    --out-dir    "$SCRATCH/REPO_DATASET/commit_parquet" \
    --workers    "$N" \
    --resume \
    --no-concat \
    "$@" \
    2>&1 | tee -a "logs/commit_parquet_local_$(date +%Y%m%d_%H%M%S).log"

echo "Workers done at $(date). Run once to concatenate:"
echo "  python create_dataset/build_commit_parquet_db.py \\"
echo "      --out-dir \"\$SCRATCH/REPO_DATASET/commit_parquet\" --concat-only"
