#!/bin/bash
#SBATCH --job-name=rag_build_256
#SBATCH --output=slurm_logs/rag_build_256_%j.out
#SBATCH --error=slurm_logs/rag_build_256_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Build RAG chunk indices: 256-tok chunks, max=500 ====="
echo "Start: $(date)"

python baselines/rag/build_indices.py \
    --repos-root  "$REPOS_ROOT" \
    --splits-dir  "$SPLITS_DIR" \
    --cache-dir   "$SCRATCH/RAG_CHUNK_CACHE_256" \
    --chunk-tokens 256 \
    --max-chunks   1000 \
    --device cuda

echo "Done: $(date)"
