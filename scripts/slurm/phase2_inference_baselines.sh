#!/bin/bash

#SBATCH --job-name=phase2_infer
#SBATCH --output=slurm_logs/phase2_infer_%j.out
#SBATCH --error=slurm_logs/phase2_infer_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 2: Inference-only baselines (no training).
# Uses structured splits throughout.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 2: Inference-only baselines ====="
echo "Start: $(date)"

# --- Pretrained ---
echo "--- Pretrained (cr_test) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_cr_test.json"

echo "--- Pretrained (ir_test) ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_ir_test.json"

# --- RAG: build indices once, then eval at different k ---
echo "--- RAG: Building chunk indices ---"
python baselines/rag/build_indices.py \
    --repos-root "$REPOS_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/RAG_CHUNK_CACHE"

for k in 3 5 10; do
    echo "--- RAG top-$k (cr_test) ---"
    python baselines/rag/test_rag.py \
        --splits-dir "$SPLITS_DIR" \
        --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
        --split cr_test --top-k $k \
        --max-input-tokens 16384 \
        --output "$BASELINES_DIR/rag_top${k}_cr_test.json"
done

# --- ICL ---
for shots in 3 5; do
    echo "--- ICL ${shots}-shot (cr_test) ---"
    python baselines/icl/test_icl.py \
        --splits-dir "$SPLITS_DIR" --split cr_test --n-shots $shots \
        --max-input-tokens 16384 \
        --output "$BASELINES_DIR/icl_${shots}shot_cr_test.json"
done

# --- Oracle Context (import-aware) ---
echo "--- Oracle Context: Building import-resolved contexts ---"
python baselines/oracle_context/build_context.py \
    --repos-root "$REPOS_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE"

echo "--- Oracle Context: Eval (cr_test) ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE" \
    --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_cr_test.json"

echo "Phase 2 complete: $(date)"
