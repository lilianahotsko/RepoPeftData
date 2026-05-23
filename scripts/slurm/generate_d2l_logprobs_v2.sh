#!/bin/bash
# ============================================================
# Generate Doc2LoRA teacher logprobs against the v2 commit-derived
# RepoPeft dataset. Sharded by repo so a SLURM array fans out the
# 400 anchor-commit groups across multiple GPUs.
#
# Output: doc2lora-compatible parquets at
#   doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train_v2/
#
# Prerequisites
# -------------
# 1. DRC source for the 400 train anchor commits. Two layouts supported
#    (controlled by DRC_CACHE_MODE):
#      per_repo (default)  : $SCRATCH/ORACLE_CONTEXT_CACHE_V4/<repo>.json
#                            (v1 cache; covers all 400 train repos
#                            out-of-the-box, no extra build step).
#      per_commit          : $SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS/
#                            <repo>__<sha>.json (requires train.parquet
#                            to expose lineno/col_offset, which the
#                            current schema does not -- use per_repo).
# 2. v2 qna parquets at $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/.
#
# Submit
# ------
#   NUM_SHARDS=4 sbatch --array=0-3 scripts/slurm/generate_d2l_logprobs_v2.sh
# ============================================================
#SBATCH --job-name=d2l-logprobs-v2
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/d2l_logprobs_v2_%A_%a.out
#SBATCH --error=slurm_logs/d2l_logprobs_v2_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

QNA_PARQUET="${QNA_PARQUET:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/train.parquet}"
DRC_CACHE_MODE="${DRC_CACHE_MODE:-per_repo}"
if [ "$DRC_CACHE_MODE" = "per_repo" ]; then
    DRC_CACHE_DIR="${DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_V4}"
else
    DRC_CACHE_DIR="${DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train_v2/train}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B}"

MAX_CTX_TOKENS="${MAX_CTX_TOKENS:-4096}"
MAX_TEACHER_TOKENS="${MAX_TEACHER_TOKENS:-8192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-2048}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-0}"
SHARD_SIZE="${SHARD_SIZE:-50}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$OUTPUT_DIR"

echo "===== generate D2L logprobs v2  shard ${SHARD_I}/${NUM_SHARDS} ====="
echo "Qna parquet  : $QNA_PARQUET"
echo "DRC cache    : $DRC_CACHE_DIR  (mode=$DRC_CACHE_MODE)"
echo "Output dir   : $OUTPUT_DIR"
echo "Model        : $MODEL"
echo "Start        : $(date)"
nvidia-smi -L || true

if [ ! -d "$DRC_CACHE_DIR" ]; then
    echo "[error] DRC_CACHE_DIR not found: $DRC_CACHE_DIR" >&2
    if [ "$DRC_CACHE_MODE" = "per_commit" ]; then
        echo "        Build it first with:" >&2
        echo "          NUM_SHARDS=4 SUITES=train sbatch --array=0-3 \\" >&2
        echo "            scripts/slurm/build_drc_cache_per_commit.sh" >&2
    else
        echo "        Expected the v1 ORACLE_CONTEXT_CACHE_V4 cache." >&2
    fi
    exit 1
fi
if [ ! -f "$QNA_PARQUET" ]; then
    echo "[error] QNA_PARQUET not found: $QNA_PARQUET" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [ "$QNAS_PER_COMMIT_LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT")
fi

python baselines/doc2lora/generate_teacher_logprobs_v2.py \
    --qna-parquet "$QNA_PARQUET" \
    --drc-cache-dir "$DRC_CACHE_DIR" \
    --drc-cache-mode "$DRC_CACHE_MODE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --max-ctx-tokens "$MAX_CTX_TOKENS" \
    --max-teacher-tokens "$MAX_TEACHER_TOKENS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --shard-size "$SHARD_SIZE" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
ls -la "$OUTPUT_DIR" | head -20
