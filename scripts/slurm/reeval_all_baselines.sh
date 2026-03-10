#!/bin/bash
#SBATCH --job-name=reeval_all
#SBATCH --output=slurm_logs/reeval_all_%j.out
#SBATCH --error=slurm_logs/reeval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Re-evaluate ALL inference-only baselines with fixed postprocess_prediction.
# Trained methods are re-evaluated via their own scripts.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Re-eval: ALL baselines (fixed metrics) ====="
echo "Start: $(date)"

echo ""
echo "========== PRETRAINED =========="
echo "--- cr_test ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_cr_test.json"

echo "--- ir_test ---"
python baselines/pretrained/test_qwen_coder.py \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/pretrained_ir_test.json"

echo ""
echo "========== RAG (k=3) =========="
echo "--- cr_test ---"
python baselines/rag/test_rag.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
    --split cr_test --top-k 3 \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/rag_top3_cr_test.json"

echo "--- ir_test ---"
python baselines/rag/test_rag.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$SCRATCH/RAG_CHUNK_CACHE" \
    --split ir_test --top-k 3 \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/rag_top3_ir_test.json"

echo ""
echo "========== ICL (3-shot) =========="
echo "--- cr_test ---"
python baselines/icl/test_icl.py \
    --splits-dir "$SPLITS_DIR" --split cr_test --n-shots 3 \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/icl_3shot_cr_test.json"

echo "--- ir_test ---"
python baselines/icl/test_icl.py \
    --splits-dir "$SPLITS_DIR" --split ir_test --n-shots 3 \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/icl_3shot_ir_test.json"

echo ""
echo "========== ORACLE CONTEXT =========="
echo "--- cr_test ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_cr_test.json"

echo "--- ir_test ---"
python baselines/oracle_context/test_oracle_context.py \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --max-input-tokens 16384 \
    --output "$BASELINES_DIR/oracle_context_ir_test.json"

echo ""
echo "Done: $(date)"
