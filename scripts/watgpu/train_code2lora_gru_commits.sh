#!/bin/bash
#SBATCH --job-name=train_gru_parquet
#SBATCH --output=slurm_logs/train_gru_parquet_%j.out
#SBATCH --error=slurm_logs/train_gru_parquet_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# watgpu variant of train_code2lora_gru_commits.sh. Reads the commit-level
# Parquet dataset downloaded from the HuggingFace Hub (HF layout:
# commits/*.parquet + qna/*.parquet + splits/*.json).
#
# One-time prerequisites:
#   bash scripts/watgpu/setup_env.sh
#   bash scripts/watgpu/setup_commit_dataset.sh nanigock/code2lora-gru-commits
#
# Overridable environment variables:
#   PARQUET_DIR          (default: $COMMIT_DATA_DIR from common.sh)
#   PARQUET_PREFER       auto | hf | shards | concat   (default: hf)
#   SUFFIX               tag appended to the checkpoint dir (default: parquet_hf)
#   ASSERTION_MODE       cumulative | new              (default: cumulative)
#   MAX_ASSERTIONS_PER_COMMIT    (default: 32)
#   EPOCHS               (default: 1)
#   GRAD_ACCUM           (default: 4)
#   LR                   (default: 1e-4)
#   EVAL_STEPS           (default: 100)
#   SAVE_STEPS           (default: 100)
#   LIMIT_TRAIN_REPOS    0 = no limit                  (default: 0)
#   LIMIT_EVAL_REPOS     0 = no limit                  (default: 64)
#   NO_INITIAL_EVAL      1 to pass --no-initial-eval   (default: 0)

source scripts/watgpu/common.sh
mkdir -p slurm_logs

# Force line-buffered stdout so cache-precompute progress, step logs, etc.
# show up live in the slurm .out file (and in `tee` pipes for interactive runs)
# instead of being stuck in Python's 8 KB block buffer.
export PYTHONUNBUFFERED=1

PARQUET_DIR="${PARQUET_DIR:-$COMMIT_DATA_DIR}"
PARQUET_PREFER="${PARQUET_PREFER:-hf}"
SUFFIX="${SUFFIX:-parquet_hf}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}"

ASSERTION_MODE="${ASSERTION_MODE:-cumulative}"
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-32}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-64}"
LM_MICRO_BATCH="${LM_MICRO_BATCH:-4}"
NO_INITIAL_EVAL="${NO_INITIAL_EVAL:-0}"

if [ ! -d "$PARQUET_DIR/commits" ] || [ ! -d "$PARQUET_DIR/qna" ]; then
    echo "ERROR: expected HF layout at $PARQUET_DIR with commits/ and qna/ subdirs."
    echo "Run: bash scripts/watgpu/setup_commit_dataset.sh"
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

# --splits-dir is only read by the (legacy) SQLite backend. The HF parquet
# dataset carries cross_repo_split / in_repo_split columns, so this value
# is just an existing directory that doesn't break argument parsing.
SPLITS_ARG="$PARQUET_DIR/splits"
[ -d "$SPLITS_ARG" ] || SPLITS_ARG="$PARQUET_DIR"

echo "===== Train: Code2LoRA-GRU (commit-level, HF parquet) ====="
echo "Parquet dir:             $PARQUET_DIR (prefer=$PARQUET_PREFER)"
echo "Output dir:              $OUT_DIR"
echo "Cache dir:               ${COMMIT_CACHE_DIR:-<disabled>}"
echo "Assertion mode:          $ASSERTION_MODE"
echo "Max assertions/commit:   $MAX_ASSERTIONS_PER_COMMIT"
echo "Epochs / grad accum:     $EPOCHS / $GRAD_ACCUM"
echo "Learning rate:           $LR"
echo "Eval / Save steps:       $EVAL_STEPS / $SAVE_STEPS"
echo "Limit train / eval:      $LIMIT_TRAIN_REPOS / $LIMIT_EVAL_REPOS"
echo "LM micro-batch:          $LM_MICRO_BATCH"
echo "No initial eval:         $NO_INITIAL_EVAL"
echo "GPU:                     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
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
    --max-seq-len 8192 \
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
