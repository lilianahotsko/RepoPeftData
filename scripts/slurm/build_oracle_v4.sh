#!/bin/bash
#SBATCH --job-name=build_v4
#SBATCH --account=def-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_oracle_v4_%j.out
#SBATCH --error=logs/build_oracle_v4_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Build adaptive oracle context cache (v4) ===="
echo "Start: $(date)"

python baselines/oracle_context/build_adaptive_cache.py \
    --v2-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V2" \
    --output-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
    --splits-dir "$SPLITS_DIR" \
    --max-seq-len 8192 \
    --margin 64 \
    --min-oracle-tokens 128 \
    --max-oracle-tokens 4096

echo "Done: $(date)"
