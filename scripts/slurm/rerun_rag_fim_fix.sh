#!/bin/bash
#SBATCH --job-name=rag_fim_fix
#SBATCH --output=slurm_logs/rag_fim_fix_%j.out
#SBATCH --error=slurm_logs/rag_fim_fix_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== RAG re-evaluation with FIM stop tokens ====="
echo "Start: $(date)"

# Indices already built, skip rebuild
for split in cr_test ir_test; do
    for k in 3 5 10; do
        echo "--- RAG top-$k ($split) ---"
        python baselines/rag/test_rag.py \
            --splits-dir "$SPLITS_DIR" \
            --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
            --split $split --top-k $k \
            --max-input-tokens 16384 \
            --output "$BASELINES_DIR/rag_top${k}_${split}.json"
    done
done

echo "Done: $(date)"
