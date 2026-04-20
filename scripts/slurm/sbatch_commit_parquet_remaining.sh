#!/bin/bash
# Extract commit Parquet shards for repos listed in missing_parquet_repos.txt
# (backfill after the main array job), then concatenate all shards into global
# commits.parquet + qna_pairs.parquet (same run).
#
# Submit from repo root:
#   sbatch scripts/slurm/sbatch_commit_parquet_remaining.sh
#
# Override list path:
#   REPO_LIST=/path/to/repos.txt sbatch scripts/slurm/sbatch_commit_parquet_remaining.sh
#
# After success, verify:
#   python -c "import pyarrow.parquet as pq; d='$SCRATCH/REPO_DATASET/commit_parquet'; print(pq.read_table(d+'/commits.parquet').num_rows)"

#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-yuntian
#SBATCH --job-name=parquet_remaining
#SBATCH --output=logs/commit_parquet_remaining_%j.out
#SBATCH --error=logs/commit_parquet_remaining_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source "${SCRATCH:-$HOME/scratch}/venvs/qwen-cu126-py312/bin/activate"
export PIP_CACHE_DIR="${SCRATCH:-$HOME/scratch}/.cache/pip"

cd /home/lhotsko/RepoPeftData
mkdir -p logs

REPO_LIST="${REPO_LIST:-$(pwd)/scripts/slurm/missing_parquet_repos.txt}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "Repo list:   $REPO_LIST"
echo "Out dir:     $OUT_DIR"
echo "Workers:     $WORKERS"
echo "Start:       $(date)"

# One process: backfill shards for listed repos, then concat all shards (global Parquet).
python create_dataset/build_commit_parquet_db.py \
    --splits-dir "$SCRATCH/REPO_DATASET" \
    --out-dir    "$OUT_DIR" \
    --repo-list  "$REPO_LIST" \
    --workers    "$WORKERS" \
    --resume

echo "Done: $(date)"
