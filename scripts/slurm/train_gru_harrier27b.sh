#!/bin/bash
# Retrain Code2LoRA-GRU on the harrier-oss-v1-27b embeddings (scaled-up encoder
# ablation). Base LLM (Qwen2.5-Coder-1.5B) and hypernet size are held at the
# default (m=1); only the input embeddings change (2048-d Qwen3 -> 5376-d
# harrier). The trainer infers the embedding dims from the parquet data, so no
# code change is needed -- only the dataset path.
#
# This is a *submit* launcher: run it on the login node, NOT via sbatch.
# It requires the harrier embedding pipeline to have finished
# (run_harrier27b_pipeline.sh), i.e. the merged dataset must exist.
#
# Usage:
#   ACCOUNT=rrg-yuntian bash scripts/slurm/train_gru_harrier27b.sh

set -euo pipefail
cd /home/lhotsko/RepoPeftData
source scripts/slurm/common.sh

ACCOUNT="${ACCOUNT:-}"
ACCT_ARGS=()
[ -n "$ACCOUNT" ] && ACCT_ARGS+=(--account "$ACCOUNT")

COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b}"
# Train QnAs (smart-cap) are text-only and encoder-independent -> reuse.
QNAS_DIR="${QNAS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap}"
# Eval-suite QnAs come from the harrier snapshots dataset (QnAs identical to
# the Qwen3 snapshots; the embeddings differ but eval reads commits-dir for
# those). Falls back to the Qwen3 snapshots QnAs if the harrier one is absent.
EVAL_QNAS_DIR="${EVAL_QNAS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_harrier27b_hf}"
SUFFIX="${SUFFIX:-h100_v2_gru_3ep_harrier27b}"

if [ ! -f "$COMMITS_DIR/commits/train.parquet" ]; then
    echo "ERROR: harrier merged dataset not found at $COMMITS_DIR/commits/train.parquet"
    echo "Run the embedding pipeline first:"
    echo "  ACCOUNT=$ACCOUNT bash scripts/slurm/run_harrier27b_pipeline.sh"
    exit 1
fi
if [ ! -d "$EVAL_QNAS_DIR/qna" ]; then
    echo "[note] $EVAL_QNAS_DIR/qna missing; falling back to Qwen3 snapshots QnAs."
    EVAL_QNAS_DIR="$SCRATCH/REPO_DATASET/code2lora_snapshots_hf"
fi

echo "===== Submit Code2LoRA-GRU (harrier-27b embeddings) ====="
echo "Commits dir   : $COMMITS_DIR"
echo "Eval QnAs dir : $EVAL_QNAS_DIR"
echo "Suffix        : $SUFFIX"

COMMITS_DIR="$COMMITS_DIR" QNAS_DIR="$QNAS_DIR" EVAL_QNAS_DIR="$EVAL_QNAS_DIR" \
SUFFIX="$SUFFIX" \
    sbatch "${ACCT_ARGS[@]}" --job-name="c2l_gru_harrier27b" \
           scripts/slurm/train_code2lora_gru_v2.sh
