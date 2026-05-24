#!/bin/bash
# Submit AST-hybrid RAG cache build + evaluations for static and commit-derived
# protocols. Run from repo root:
#   bash scripts/slurm/eval_rag_ast_v2_pipeline.sh
#
# Static snapshot QnAs: code2lora_snapshots_hf/qna (Table-1-style dataset).
# GRU commit-history QnAs: commit_parquet_hf/qna (Table-2-style dataset).
# Shared per-commit AST RAG cache; GRU eval uses a tighter context budget.

set -euo pipefail
cd /home/lhotsko/RepoPeftData
source scripts/slurm/common.sh
mkdir -p logs slurm_logs

CACHE_DIR="${CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS}"
BUILD_SUITES="${BUILD_SUITES:-cr_test ir_test}"
NUM_BUILD_SHARDS="${NUM_BUILD_SHARDS:-8}"
ACCOUNT="${ACCOUNT:-rrg-yuntian}"

echo "=== Submit AST RAG cache build (${NUM_BUILD_SHARDS} shards, suites: ${BUILD_SUITES}) ==="
BUILD_ID=$(NUM_SHARDS="$NUM_BUILD_SHARDS" SUITES="$BUILD_SUITES" OUT_DIR="$CACHE_DIR" FORCE="${FORCE:-0}" \
  sbatch --parsable --account="$ACCOUNT" \
  --array=0-$((NUM_BUILD_SHARDS - 1)) \
  scripts/slurm/build_rag_cache_ast_per_commit.sh)
echo "Build job: $BUILD_ID"
echo "$BUILD_ID" > .last_rag_ast_build_jobid

echo "=== Submit static-protocol RAG eval (cr_test + ir_test) ==="
EVAL_STATIC=$(METHOD=rag \
  RAG_CACHE_DIR="$CACHE_DIR" \
  RAG_MAX_CONTEXT_TOKENS="${RAG_STATIC_CTX:-1536}" \
  NUM_SHARDS=4 SUITES="cr_test ir_test" \
  SUFFIX=rag_ast_v2_static \
  sbatch --parsable --account="$ACCOUNT" \
  --dependency=afterok:"${BUILD_ID}" --array=0-7 \
  scripts/slurm/eval_baselines_v2_sharded.sh)
echo "Static eval: $EVAL_STATIC"

echo "=== Submit GRU commit-parquet RAG eval (commit_parquet_hf/qna) ==="
EVAL_GRU=$(METHOD=rag \
  QNA_DIR="$SCRATCH/REPO_DATASET/commit_parquet_hf/qna" \
  RAG_CACHE_DIR="$CACHE_DIR" \
  RAG_MAX_CONTEXT_TOKENS="${RAG_COMMIT_CTX:-1024}" \
  NUM_SHARDS=4 SUITES="cr_test ir_test" \
  SUFFIX=rag_ast_v2_gru_parquet \
  sbatch --parsable --account="$ACCOUNT" \
  --dependency=afterok:"${BUILD_ID}" --array=0-7 \
  scripts/slurm/eval_baselines_v2_sharded.sh)
echo "GRU parquet eval: $EVAL_GRU"

echo "$BUILD_ID $EVAL_STATIC $EVAL_GRU" > .last_rag_ast_eval_jobids
squeue -u "$USER" 2>/dev/null | head -20
