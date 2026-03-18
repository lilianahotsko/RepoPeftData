#!/bin/bash
#SBATCH --job-name=rag_eval_256
#SBATCH --output=slurm_logs/rag_eval_256_%j.out
#SBATCH --error=slurm_logs/rag_eval_256_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

CHUNK_CACHE="$SCRATCH/RAG_CHUNK_CACHE_256"

echo "===== Eval RAG: 256-tok chunks, k=5,10 on cr_test + ir_test ====="
echo "Start: $(date)"

for split in cr_test ir_test; do
    for k in 5 10; do
        echo ""
        echo "--- split=$split  k=$k ---"
        python baselines/rag/test_rag.py \
            --splits-dir "$SPLITS_DIR" \
            --cache-dir "$CHUNK_CACHE" \
            --split "$split" \
            --top-k "$k" \
            --output "$BASELINES_DIR/rag256_k${k}_${split}.json"
    done
done

echo ""
echo "Done: $(date)"
