#!/bin/bash
#SBATCH --job-name=smoke_c2l_gru_v2
#SBATCH --output=slurm_logs/smoke_c2l_gru_v2_%j.out
#SBATCH --error=slurm_logs/smoke_c2l_gru_v2_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Smoke test for the v2 *GRU* Code2LoRA trainer. Runs 1 epoch on a few train
# repos with a tiny eval slice. Purpose: verify, before committing to the full
# multi-day run, that
#   (a) the (frozen) base model loads + LoRA wrappers attach,
#   (b) memory fits at the real training max-seq-len (default 4096),
#   (c) loss computes and autograd reaches the GRU + head + LoRA,
#   (d) end-of-epoch checkpoint saves.
#
# Override the frozen base via MODEL_NAME, e.g.:
#   MODEL_NAME=Qwen/Qwen2.5-Coder-3B sbatch --account=rrg-yuntian_gpu \
#       scripts/slurm/smoke_train_code2lora_gru_v2.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
QNAS_DIR="${QNAS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap}"
EVAL_QNAS_DIR="${EVAL_QNAS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"
SUFFIX="${SUFFIX:-smoke_v2_gru_5repos}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/${SUFFIX}"
mkdir -p "$OUT_DIR"

# Smoke knobs: small enough to finish in <1h, but max-seq-len matches the real
# run so we also de-risk memory at training scale.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-5}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-3}"

echo "===== SMOKE: Code2LoRA-GRU v2 ====="
echo "Model             : $MODEL_NAME"
echo "Commits dir       : $COMMITS_DIR"
echo "Train QnAs dir    : $QNAS_DIR"
echo "Eval suite QnAs   : $EVAL_QNAS_DIR"
echo "Output dir        : $OUT_DIR"
echo "Max seq len       : $MAX_SEQ_LEN"
echo "Start             : $(date)"
nvidia-smi -L || true

python hypernetwork/train_code2lora_gru_v2.py \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --eval-qnas-dir "$EVAL_QNAS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --rank 16 --alpha 32 \
    --head-hidden-dim 1024 \
    --gru-hidden-dim 2048 \
    --epochs 1 --lr 5e-5 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --bptt-window 16 \
    --lm-micro-batch 1 \
    --max-qna-per-commit 8 \
    --eval-every-repos 0 \
    --eval-suites cr_val \
    --primary-eval-suite cr_val \
    --limit-eval-repos "$LIMIT_EVAL_REPOS" \
    --limit-train-repos "$LIMIT_TRAIN_REPOS" \
    --seed 3407

echo "Done: $(date)"
ls -la "$OUT_DIR"
echo
echo "=== metrics.jsonl ==="
cat "$OUT_DIR/metrics.jsonl" 2>/dev/null || echo "(no metrics file)"
