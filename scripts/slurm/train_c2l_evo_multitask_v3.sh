#!/bin/bash
#SBATCH --job-name=c2l_evo_multitask_v3
#SBATCH --output=slurm_logs/c2l_evo_multitask_v3_%j.out
#SBATCH --error=slurm_logs/c2l_evo_multitask_v3_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Finetune the single-task Code2LoRA-GRU into a multi-task (assert_rhs + qa)
# head using the FULL v2 QA set (diff + repo-state) *with assert_rhs rehearsal*.
#
# This is the missing combination:
#   * c2l_evo_multitask_v2   = full v2 QA (129k), NO rehearsal  -> assert drifts
#   * c2l_evo_multitask_rehearsal = old diff-QA (58k) + rehearsal
#   * c2l_evo_multitask_v3   = full v2 QA (129k) + rehearsal    <-- HERE
#
# Recipe:
#   * Warm-start GRU + head from the single-task assert_rhs GRU (grown 0->2).
#   * Train on a mixed train.parquet = ALL v2 qa rows + a light assert_rhs
#     rehearsal sample (0.25 of train commit-groups, 1 QnA/commit), so the
#     assertion task (index 0) keeps receiving gradient and does not drift.

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"

# ---- Build the v3 rehearsal-mix training set (idempotent) ----
# qa side  = full v2 QA (diff + repo-state); assert side = smart-capped train.
QA_PARQUET="${QA_PARQUET:-$SCRATCH/REPO_DATASET/qna_qa_hf_v2/qna/train.parquet}"
ASSERT_PARQUET="${ASSERT_PARQUET:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet}"
QNAS_DIR="${QNAS_DIR:-$SCRATCH/REPO_DATASET/qna_qa_hf_v3_rehearsal}"
REHEARSAL_FRAC="${REHEARSAL_FRAC:-0.25}"

if [ ! -f "$QNAS_DIR/qna/train.parquet" ]; then
    echo "===== Building v3 rehearsal mix (qa=full v2 + assert rehearsal) ====="
    python create_dataset/build_qa_rehearsal_mix.py \
        --qa-parquet "$QA_PARQUET" \
        --assert-parquet "$ASSERT_PARQUET" \
        --out-dir "$QNAS_DIR" \
        --commit-frac "$REHEARSAL_FRAC"
else
    echo "===== Reusing existing v3 rehearsal mix at $QNAS_DIR ====="
fi

# Eval on the v2 qa suites (comparable to c2l_evo_multitask_v2 val loss).
EVAL_QNAS_DIR="${EVAL_QNAS_DIR:-$SCRATCH/REPO_DATASET/qna_qa_hf_v2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"

# Warm-start source: the canonical single-task assert_rhs GRU (the "original"
# GRU checkpoint). The head is grown 0->2 tasks at load.
INIT_CKPT="${INIT_CKPT:-$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep_h1024_g2048/gru_head.best.pt}"

TASKS="${TASKS:-assert_rhs qa}"
TASK_DIM="${TASK_DIM:-64}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
GRU_HIDDEN="${GRU_HIDDEN:-2048}"
HEAD_HIDDEN="${HEAD_HIDDEN:-1024}"
SUFFIX="${SUFFIX:-c2l_evo_multitask_v3}"
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

echo "===== Finetune: Code2LoRA-GRU -> c2l_evo_multitask_v3 (full v2 qa + rehearsal) ====="
echo "Model             : $MODEL_NAME"
echo "Warm-start ckpt   : $INIT_CKPT"
echo "Tasks (idx order) : $TASKS"
echo "Commits dir       : $COMMITS_DIR"
echo "Train QnAs dir    : $QNAS_DIR  (full v2 qa + assert rehearsal @ $REHEARSAL_FRAC)"
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
