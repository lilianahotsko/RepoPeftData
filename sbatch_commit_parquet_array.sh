#!/bin/bash
# Array-job variant of the commit-parquet builder.
#
# Each task grabs a round-robin subset of repos via --shard-index /
# --num-shards. Each task writes its own _done.NNNN.jsonl so there is no
# coordination between workers. Each repo is independent and produces its own
# per-repo Parquet shard in $OUT_DIR/shards/, so writes never collide.
#
# Submit:
#   sbatch --array=0-3 sbatch_commit_parquet_array.sh   # 4 tasks x 8 workers = 32-way
#   sbatch --array=0-7 sbatch_commit_parquet_array.sh   # 8 tasks x 8 workers = 64-way
#
# When the array finishes, run once on the login node (or as a dependent
# follow-up job) to produce the global parquets:
#   module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
#   source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
#   python create_dataset/build_commit_parquet_db.py \
#       --out-dir $SCRATCH/REPO_DATASET/commit_parquet --concat-only

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=def-yuntian
#SBATCH --job-name=commit_parquet_db
#SBATCH --output=logs/commit_parquet_db_%A_%a.out
#SBATCH --error=logs/commit_parquet_db_%A_%a.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source "$SCRATCH/venvs/qwen-cu126-py312/bin/activate"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

cd /home/lhotsko/RepoPeftData
mkdir -p logs

: "${SLURM_ARRAY_TASK_COUNT:=${SLURM_ARRAY_TASK_MAX:-0}+1}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-4}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
# One task uses --cpus-per-task local workers in parallel on top of sharding.
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} (x${WORKERS} workers) starting at $(date)"

python create_dataset/build_commit_parquet_db.py \
    --splits-dir "$SCRATCH/REPO_DATASET" \
    --out-dir    "$SCRATCH/REPO_DATASET/commit_parquet" \
    --shard-index "$SHARD_INDEX" \
    --num-shards  "$NUM_SHARDS" \
    --workers     "$WORKERS" \
    --resume \
    --no-concat

echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} finished at $(date)"
