#!/bin/bash
# Precompute per-repo caches for Code2LoRA-GRU commit-parquet training:
# - diff embeddings (Qwen3-Embedding)
# - tokenized assertions (Qwen2.5-Coder tokenizer)
#
# Writes: $COMMIT_CACHE_DIR/commit_cache_<fingerprint>/*.pt
#
# Usage:
#   sbatch scripts/slurm/precompute_gru_commit_cache.sh
#
# Optional env overrides:
#   PARQUET_DIR, PARQUET_PREFER, COMMIT_CACHE_DIR, LIMIT_TRAIN_REPOS, LIMIT_EVAL_REPOS,
#   MAX_SEQ_LEN, CHUNK_TOKENS, CHUNK_OVERLAP, EMBED_MODEL, MODEL_NAME

#SBATCH --job-name=precompute_gru_cache
#SBATCH --output=slurm_logs/precompute_gru_cache_%j.out
#SBATCH --error=slurm_logs/precompute_gru_cache_%j.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export COMMIT_CACHE_DIR="${COMMIT_CACHE_DIR:-$SCRATCH/REPO_DATASET/commit_cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$COMMIT_CACHE_DIR" "$HF_HOME"

PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
PARQUET_PREFER="${PARQUET_PREFER:-hf}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"  # 0 = all train repos
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-0}"    # 0 = all eval repos

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"

SPLITS_ARG="$PARQUET_DIR/splits"
[ -d "$SPLITS_ARG" ] || SPLITS_ARG="$PARQUET_DIR"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
  EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi
if [ "$LIMIT_EVAL_REPOS" != "0" ]; then
  EXTRA_ARGS+=(--limit-eval-repos "$LIMIT_EVAL_REPOS")
fi

echo "===== Precompute: Code2LoRA-GRU commit cache ====="
echo "Parquet dir:             $PARQUET_DIR (prefer=$PARQUET_PREFER)"
echo "Cache dir:               $COMMIT_CACHE_DIR"
echo "Model / embed:           $MODEL_NAME / $EMBED_MODEL"
echo "Max seq len:             $MAX_SEQ_LEN"
echo "Chunk tokens / overlap:  $CHUNK_TOKENS / $CHUNK_OVERLAP"
echo "Limit train / eval:      $LIMIT_TRAIN_REPOS / $LIMIT_EVAL_REPOS"
echo "GPU:                     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
echo "Start: $(date)"

python hypernetwork/train_code2lora_gru_commits.py \
  --data-source parquet \
  --parquet-dir "$PARQUET_DIR" \
  --parquet-prefer "$PARQUET_PREFER" \
  --splits-dir "$SPLITS_ARG" \
  --cache-dir "$COMMIT_CACHE_DIR" \
  --precompute-cache-only \
  --defer-parquet-qna \
  --model-name "$MODEL_NAME" \
  --embed-model "$EMBED_MODEL" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --chunk-tokens "$CHUNK_TOKENS" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --train-in-repo-splits train \
  --in-repo-val-splits val \
  --in-repo-test-splits test \
  --cross-repo-eval-splits cr_val cr_test \
  --no-initial-eval \
  --output-dir "$CKPT_DIR/CODE2LORA_GRU/cache_warmup_full" \
  --seed 3407 \
  "${EXTRA_ARGS[@]}"

echo "Done: $(date)"

