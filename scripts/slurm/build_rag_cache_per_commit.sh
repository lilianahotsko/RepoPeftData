#!/bin/bash
#SBATCH --job-name=build_rag_per_commit
#SBATCH --output=slurm_logs/build_rag_per_commit_%A_%a.out
#SBATCH --error=slurm_logs/build_rag_per_commit_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Precompute per-(repo_id, commit_sha) RAG chunk indices for the v2
# commit-derived QnA suites. Each array task processes one shard of the
# unique (repo, commit) keys via deterministic round-robin sharding; the
# Python driver writes the per-(repo, sha) .pt file atomically so a
# wall-time kill never corrupts the cache and a rerun resumes
# automatically.
#
# Usage examples
# --------------
#   # 4-way sharded build over cr_val + cr_test + ir_val + ir_test:
#   NUM_SHARDS=4 SUITES="cr_val cr_test ir_val ir_test" \
#     sbatch --array=0-3 scripts/slurm/build_rag_cache_per_commit.sh
#
#   # Single-suite, single-shard smoke test on 16 keys:
#   LIMIT=16 SUITES="cr_val" NUM_SHARDS=1 \
#     sbatch --array=0 scripts/slurm/build_rag_cache_per_commit.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
REPOS_ROOT="${REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories}"
OUT_DIR="${OUT_DIR:-$SCRATCH/RAG_CHUNK_CACHE_COMMITS}"
SUITES_STR="${SUITES:-cr_val cr_test ir_val ir_test}"
read -r -a SUITES <<< "$SUITES_STR"

EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
OVERLAP="${OVERLAP:-64}"
MAX_CHUNKS="${MAX_CHUNKS:-300}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-0}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"

echo "===== build RAG per-commit cache  shard ${SHARD_I}/${NUM_SHARDS} ====="
echo "QnA dir   : $QNA_DIR"
echo "Suites    : ${SUITES[*]}"
echo "Repos root: $REPOS_ROOT"
echo "Out dir   : $OUT_DIR"
echo "Embedder  : $EMBED_MODEL"
echo "Chunks    : $CHUNK_TOKENS tokens, overlap $OVERLAP, max $MAX_CHUNKS"
echo "Start     : $(date)"
nvidia-smi -L || true

EXTRA_ARGS=()
if [ "$LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

python evaluation/build_rag_cache_per_commit.py \
    --qna-dir "$QNA_DIR" \
    --suites "${SUITES[@]}" \
    --repos-root "$REPOS_ROOT" \
    --out-dir "$OUT_DIR" \
    --embed-model-name "$EMBED_MODEL" \
    --chunk-tokens "$CHUNK_TOKENS" \
    --overlap "$OVERLAP" \
    --max-chunks "$MAX_CHUNKS" \
    --batch-size "$BATCH_SIZE" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
