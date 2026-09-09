#!/bin/bash
#SBATCH --job-name=c2l_evo_multitask
#SBATCH --output=slurm_logs/c2l_evo_multitask_%j.out
#SBATCH --error=slurm_logs/c2l_evo_multitask_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Finetune an existing single-task Code2LoRA-GRU checkpoint into a multi-task
# (assert_rhs + qa) head, training **only on the qa dataset**.
#
# What this does:
#   * Warm-starts GRU + head from $INIT_CKPT (a single-task assert_rhs run).
#     The head is grown to 2 tasks: the trunk's first layer is column-padded
#     with zeros for the task-embedding input, so at init the loaded behaviour
#     is byte-identical to the source for BOTH task ids (no disturbance).
#   * Trains on the qa QnAs only (task=qa, routed to task index 1). The
#     assertion task (index 0) is preserved from the warm-start; it is NOT
#     trained here (we are not mixing in the assert_rhs data).
#
# Requires:
#   - Diff/repo-state embeddings: $SCRATCH/REPO_DATASET/commit_parquet_hf_v2
#   - qa QnAs:                     $SCRATCH/REPO_DATASET/qna_qa_hf/qna/*.parquet
#   - A trained single-task ckpt at $INIT_CKPT

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
# qa-only: both the training QnAs and the eval suites come from the qa dataset.
QNAS_DIR="${QNAS_DIR:-$SCRATCH/REPO_DATASET/qna_qa_hf}"
EVAL_QNAS_DIR="${EVAL_QNAS_DIR:-$SCRATCH/REPO_DATASET/qna_qa_hf}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"

# Warm-start source: the canonical single-task assert_rhs GRU (head 1024 /
# gru 2048, per-step diff). Override INIT_CKPT to finetune a different run.
INIT_CKPT="${INIT_CKPT:-$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep_h1024_g2048/gru_head.best.pt}"

# Task order is the *stable index map*: assert_rhs=0 (preserved), qa=1 (trained).
TASKS="${TASKS:-assert_rhs qa}"
TASK_DIM="${TASK_DIM:-64}"

# Finetune knobs: shorter + gentler than a from-scratch run to limit drift.
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
# Must match the warm-start source's dims.
GRU_HIDDEN="${GRU_HIDDEN:-2048}"
HEAD_HIDDEN="${HEAD_HIDDEN:-1024}"
SUFFIX="${SUFFIX:-c2l_evo_multitask}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/${SUFFIX}"
mkdir -p "$OUT_DIR"
BPTT_WINDOW="${BPTT_WINDOW:-16}"
MAX_QNA_PER_COMMIT="${MAX_QNA_PER_COMMIT:-8}"
LM_MICRO_BATCH="${LM_MICRO_BATCH:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PER_STEP_INPUT="${PER_STEP_INPUT:-diff}"
EVAL_EVERY_REPOS="${EVAL_EVERY_REPOS:-80}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-10}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"
SEED="${SEED:-3407}"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi

echo "===== Finetune: Code2LoRA-GRU -> c2l_evo_multitask (qa-only) ====="
echo "Model             : $MODEL_NAME"
echo "Warm-start ckpt   : $INIT_CKPT"
echo "Tasks (idx order) : $TASKS"
echo "Commits dir       : $COMMITS_DIR"
echo "Train QnAs dir    : $QNAS_DIR  (qa only)"
echo "Eval suite QnAs   : $EVAL_QNAS_DIR"
echo "Output dir        : $OUT_DIR"
echo "Head / GRU hidden : $HEAD_HIDDEN / $GRU_HIDDEN"
echo "Epochs / LR       : $EPOCHS / $LR"
echo "Per-step input    : $PER_STEP_INPUT"
echo "Seed              : $SEED"
echo "Start             : $(date)"

python hypernetwork/train_code2lora_gru_v2.py \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --eval-qnas-dir "$EVAL_QNAS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --init-checkpoint "$INIT_CKPT" \
    --tasks $TASKS \
    --task-dim "$TASK_DIM" \
    --rank 16 --alpha 32 \
    --head-hidden-dim "$HEAD_HIDDEN" \
    --gru-hidden-dim "$GRU_HIDDEN" \
    --epochs "$EPOCHS" --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --per-step-input "$PER_STEP_INPUT" \
    --bptt-window "$BPTT_WINDOW" \
    --lm-micro-batch "$LM_MICRO_BATCH" \
    --max-qna-per-commit "$MAX_QNA_PER_COMMIT" \
    --eval-every-repos "$EVAL_EVERY_REPOS" \
    --eval-suites cr_val \
    --primary-eval-suite cr_val \
    --limit-eval-repos "$LIMIT_EVAL_REPOS" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
