#!/bin/bash
#SBATCH --job-name=p2b_rag_ir
#SBATCH --output=slurm_logs/p2b_rag_ir_%j.out
#SBATCH --error=slurm_logs/p2b_rag_ir_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== RAG baseline (IR test) ====="
echo "Start: $(date)"

for k in 3 5 10; do
    echo "--- RAG top-$k (ir_test_structured) ---"
    python baselines/rag/test_rag.py \
        --splits-dir "$SPLITS_DIR" \
        --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
        --split ir_test_structured --top-k $k \
        --max-input-tokens 16384 \
        --output "$BASELINES_DIR/rag_top${k}_ir_test_structured.json"
done

echo "Done: $(date)"
