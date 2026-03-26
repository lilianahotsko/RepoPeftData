#!/bin/bash
#SBATCH --job-name=build_v3_cache
#SBATCH --output=slurm_logs/build_v3_cache_%j.out
#SBATCH --error=slurm_logs/build_v3_cache_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source scripts/watgpu/common.sh

echo "===== Build Compressed Oracle Context Cache (v3) ====="
echo "Start: $(date)"

python baselines/oracle_context/build_compressed_cache.py \
    --v2-cache-dir "$ORACLE_CACHE_DIR" \
    --output-dir "${SCRATCH:-$HOME/scratch}/repopeft_data/oracle_context_cache_v3" \
    --splits-dir "$SPLITS_DIR" \
    --max-tokens 6000

echo "Done: $(date)"
