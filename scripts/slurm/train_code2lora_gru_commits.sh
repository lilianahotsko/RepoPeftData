#!/bin/bash
#SBATCH --job-name=train_gru_parquet
#SBATCH --output=slurm_logs/train_gru_parquet_%j.out
#SBATCH --error=slurm_logs/train_gru_parquet_%j.err
#SBATCH --time=72:00:00
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$COMMIT_CACHE_DIR" "$HF_HOME"

PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
PARQUET_PREFER="${PARQUET_PREFER:-hf}"
# Default suffix matches the paper-grade run: 5 epochs over the full 409 train
# repos. Override via SUFFIX=... when launching ablations.
SUFFIX="${SUFFIX:-h100_5ep_full}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}"

ASSERTION_MODE="${ASSERTION_MODE:-new}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-8}"
LM_MICRO_BATCH="${LM_MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"  # 0 = all train repos
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-64}"   # set 0 for full eval
NO_INITIAL_EVAL="${NO_INITIAL_EVAL:-1}"
# Avoid OOM during startup on large HF QnA tables (stream per repo in cache phase).
DEFER_PARQUET_QNA="${DEFER_PARQUET_QNA:-1}"

if [ ! -d "$PARQUET_DIR/commits" ] || [ ! -d "$PARQUET_DIR/qna" ]; then
    echo "ERROR: expected HF parquet layout at $PARQUET_DIR with commits/ and qna/."
    exit 1
fi

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi
if [ "$LIMIT_EVAL_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-eval-repos "$LIMIT_EVAL_REPOS")
fi
if [ "$NO_INITIAL_EVAL" = "1" ]; then
    EXTRA_ARGS+=(--no-initial-eval)
fi
if [ "$DEFER_PARQUET_QNA" = "1" ]; then
    EXTRA_ARGS+=(--defer-parquet-qna)
fi

SPLITS_ARG="$PARQUET_DIR/splits"
[ -d "$SPLITS_ARG" ] || SPLITS_ARG="$PARQUET_DIR"

echo "===== Train: Code2LoRA-GRU (commit-level, HF parquet, H100) ====="
echo "Parquet dir:             $PARQUET_DIR (prefer=$PARQUET_PREFER)"
echo "Output dir:              $OUT_DIR"
echo "Cache dir:               $COMMIT_CACHE_DIR"
echo "Assertion mode:          $ASSERTION_MODE"
echo "Max seq len:             $MAX_SEQ_LEN"
echo "Max assertions/commit:   $MAX_ASSERTIONS_PER_COMMIT"
echo "Epochs / grad accum:     $EPOCHS / $GRAD_ACCUM"
echo "LM micro-batch:          $LM_MICRO_BATCH"
echo "Limit train / eval:      $LIMIT_TRAIN_REPOS / $LIMIT_EVAL_REPOS"
echo "No initial eval:         $NO_INITIAL_EVAL"
echo "Defer parquet QnA:     $DEFER_PARQUET_QNA"
echo "GPU:                     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
echo "Start: $(date)"

python hypernetwork/train_code2lora_gru_commits.py \
    --data-source parquet \
    --parquet-dir "$PARQUET_DIR" \
    --parquet-prefer "$PARQUET_PREFER" \
    --splits-dir "$SPLITS_ARG" \
    --output-dir "$OUT_DIR" \
    --model-name "Qwen/Qwen2.5-Coder-1.5B" \
    --embed-model "Qwen/Qwen3-Embedding-0.6B" \
    --init-type zeros \
    --gru-hidden-dim 1024 \
    --gru-num-layers 1 \
    --rank 16 \
    --alpha 32 \
    --num-bases 16 \
    --trunk-depth 2 \
    --lora-hidden-dim 512 \
    --chunk-tokens 512 \
    --chunk-overlap 64 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --train-in-repo-splits train \
    --in-repo-val-splits val \
    --in-repo-test-splits test \
    --cross-repo-eval-splits cr_val cr_test \
    --assertion-mode "$ASSERTION_MODE" \
    --max-assertions-per-commit "$MAX_ASSERTIONS_PER_COMMIT" \
    --lm-micro-batch "$LM_MICRO_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --max-grad-norm 1.0 \
    --epochs "$EPOCHS" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps 10 \
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
