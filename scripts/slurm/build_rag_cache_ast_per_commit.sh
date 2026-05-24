#!/bin/bash
#SBATCH --job-name=build_rag_ast
#SBATCH --output=slurm_logs/build_rag_ast_%A_%a.out
#SBATCH --error=slurm_logs/build_rag_ast_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# AST function/class chunks + dense embeddings + BM25 token lists per commit.
# Output: $SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS/<repo>__<sha>.pt
#
# Usage:
#   NUM_SHARDS=8 SUITES="cr_val cr_test ir_val ir_test ood_test" \
#     sbatch --account=rrg-yuntian --array=0-7 \
#       scripts/slurm/build_rag_cache_ast_per_commit.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
REPOS_ROOT="${REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories}"
OUT_DIR="${OUT_DIR:-$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS}"
SUITES_STR="${SUITES:-cr_val cr_test ir_val ir_test ood_test}"
read -r -a SUITES <<< "$SUITES_STR"

EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
CHUNK_MODE="${CHUNK_MODE:-ast}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"
FORCE="${FORCE:-0}"

echo "===== build AST RAG cache  shard ${SHARD_I}/${NUM_SHARDS} ====="
echo "Out dir   : $OUT_DIR"
echo "Suites    : ${SUITES[*]}"
echo "Chunk mode: $CHUNK_MODE"
echo "Start     : $(date)"
nvidia-smi -L || true

EXTRA_ARGS=(--chunk-mode "$CHUNK_MODE" --chunk-tokens "$CHUNK_TOKENS" --max-chunks "$MAX_CHUNKS")
if [ "$LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [ "$FORCE" = "1" ]; then
    EXTRA_ARGS+=(--force)
fi

python evaluation/build_rag_cache_per_commit.py \
    --qna-dir "$QNA_DIR" \
    --suites "${SUITES[@]}" \
    --repos-root "$REPOS_ROOT" \
    --out-dir "$OUT_DIR" \
    --embed-model-name "$EMBED_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
