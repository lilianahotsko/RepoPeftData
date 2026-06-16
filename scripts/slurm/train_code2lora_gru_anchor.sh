#!/bin/bash
#SBATCH --job-name=train_c2l_gru_anchor
#SBATCH --output=slurm_logs/train_c2l_gru_anchor_%j.out
#SBATCH --error=slurm_logs/train_c2l_gru_anchor_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Ablation: train Code2LoRA-GRU using ONLY the anchor commit's
# `repo_state_embedding` (replicated across all GRU steps), supervised
# by the static-trainer's anchor QnAs. The GRU still unrolls N steps per
# repo (N = number of kept commits in v2 chronology), but every step
# receives the same input.
#
# Requires:
#   - $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits/*.parquet
#     (only commit_sha / commit_index / in_repo_split are read).
#   - $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/*.parquet
#     (provides anchor repo_state_embedding per repo) and
#     $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/*.parquet
#     (provides train + eval QnAs).

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
SUFFIX="${SUFFIX:-h100_v2_gru_anchor_3ep}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/${SUFFIX}"
mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
GRU_HIDDEN="${GRU_HIDDEN:-2048}"
HEAD_HIDDEN="${HEAD_HIDDEN:-1024}"
BPTT_WINDOW="${BPTT_WINDOW:-16}"
MAX_QNA_PER_COMMIT="${MAX_QNA_PER_COMMIT:-8}"
LM_MICRO_BATCH="${LM_MICRO_BATCH:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
EVAL_EVERY_REPOS="${EVAL_EVERY_REPOS:-80}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-10}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"
SEED="${SEED:-3407}"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi

echo "===== Train: Code2LoRA-GRU (anchor-only ablation) ====="
echo "Commits dir   : $COMMITS_DIR  (chronology only)"
echo "Snapshots dir : $SNAPSHOTS_DIR (anchor emb + QnAs)"
echo "Output dir    : $OUT_DIR"
echo "Epochs / LR   : $EPOCHS / $LR"
echo "Seed          : $SEED"
echo "Start         : $(date)"

python hypernetwork/train_code2lora_gru_anchor.py \
    --commits-dir "$COMMITS_DIR" \
    --snapshots-dir "$SNAPSHOTS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name Qwen/Qwen2.5-Coder-1.5B \
    --rank 16 --alpha 32 \
    --head-hidden-dim "$HEAD_HIDDEN" \
    --gru-hidden-dim "$GRU_HIDDEN" \
    --epochs "$EPOCHS" --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
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
