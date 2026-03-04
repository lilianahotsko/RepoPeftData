#!/bin/bash
#SBATCH --job-name=p2b_rag
#SBATCH --output=slurm_logs/p2b_rag_%j.out
#SBATCH --error=slurm_logs/p2b_rag_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== RAG baseline ====="
echo "Start: $(date)"

echo "--- Building chunk indices (skips cached) ---"
python baselines/rag/build_indices.py \
    --repos-root "$REPOS_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/RAG_CHUNK_CACHE"

for k in 3 5 10; do
    echo "--- RAG top-$k (cr_test_structured) ---"
    python baselines/rag/test_rag.py \
        --splits-dir "$SPLITS_DIR" \
        --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
        --split cr_test_structured --top-k $k \
        --max-input-tokens 16384 \
        --output "$BASELINES_DIR/rag_top${k}_cr_test_structured.json"
done

echo "Done: $(date)"
