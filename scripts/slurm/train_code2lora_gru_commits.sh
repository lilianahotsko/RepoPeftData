#!/bin/bash
#SBATCH --job-name=train_gru_parquet
#SBATCH --output=slurm_logs/train_gru_parquet_%j.out
#SBATCH --error=slurm_logs/train_gru_parquet_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
module load arrow/18.1.0
mkdir -p slurm_logs

# --------- Defaults (overridable via env) ----------
PARQUET_DIR="${PARQUET_DIR:-$SPLITS_DIR/commit_parquet}"
PARQUET_PREFER="${PARQUET_PREFER:-auto}"            # auto | concat | shards
SUFFIX="${SUFFIX:-parquet}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}"
ASSERTION_MODE="${ASSERTION_MODE:-cumulative}"      # cumulative | new
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-32}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"         # 0 = no limit
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-64}"          # cap held-out eval size
LM_MICRO_BATCH="${LM_MICRO_BATCH:-4}"               # assertions per LM forward

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi
if [ "$LIMIT_EVAL_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-eval-repos "$LIMIT_EVAL_REPOS")
fi

echo "===== Train: Code2LoRA-GRU (commit-level, Parquet-backed) ====="
echo "Parquet dir:             $PARQUET_DIR (prefer=$PARQUET_PREFER)"
echo "Output dir:              $OUT_DIR"
echo "Assertion mode:          $ASSERTION_MODE"
echo "Max assertions/commit:   $MAX_ASSERTIONS_PER_COMMIT"
echo "Epochs:                  $EPOCHS"
echo "Grad accum:              $GRAD_ACCUM"
echo "Learning rate:           $LR"
echo "Limit train repos:       $LIMIT_TRAIN_REPOS"
echo "Limit eval repos:        $LIMIT_EVAL_REPOS"
echo "LM micro-batch:          $LM_MICRO_BATCH"
echo "Start: $(date)"

python hypernetwork/train_code2lora_gru_commits.py \
    --data-source parquet \
    --parquet-dir "$PARQUET_DIR" \
    --parquet-prefer "$PARQUET_PREFER" \
    --splits-dir "$SPLITS_DIR" \
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
