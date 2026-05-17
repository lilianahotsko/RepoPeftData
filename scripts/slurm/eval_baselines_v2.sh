#!/bin/bash
#SBATCH --job-name=eval_baselines_v2
#SBATCH --output=slurm_logs/eval_baselines_v2_%j.out
#SBATCH --error=slurm_logs/eval_baselines_v2_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate one baseline (pretrained / fft / slora) on the v2 per-commit
# eval suites. Parameterized by env vars METHOD and CKPT.
#
# Examples:
#   METHOD=pretrained                                  sbatch scripts/slurm/eval_baselines_v2.sh
#   METHOD=fft   CKPT=$CKPT_DIR/FFT_V2/h100_v2/final   sbatch scripts/slurm/eval_baselines_v2.sh
#   METHOD=slora CKPT=$CKPT_DIR/SLORA_V2/h100_v2/final sbatch scripts/slurm/eval_baselines_v2.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

METHOD="${METHOD:-pretrained}"
CKPT="${CKPT:-}"
QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2/qna}"
SUITES="${SUITES:-ir_val ir_test cr_val cr_test}"
SUFFIX="${SUFFIX:-h100_v2}"
OUT_DIR="$CKPT_DIR/BASELINES_V2/${METHOD}_${SUFFIX}"
mkdir -p "$OUT_DIR"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REPO_LIMIT="${REPO_LIMIT:-0}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

EXTRA_ARGS=()
if [ -n "$CKPT" ]; then
    EXTRA_ARGS+=(--ckpt "$CKPT")
fi
if [ "$REPO_LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--repo-limit "$REPO_LIMIT")
fi
if [ "$QNAS_PER_COMMIT_LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT")
fi

echo "===== Eval baseline v2 ====="
echo "Method        : $METHOD"
echo "Checkpoint    : ${CKPT:-<none>}"
echo "QnA dir       : $QNA_DIR"
echo "Suites        : $SUITES"
echo "Output dir    : $OUT_DIR"
echo "Max input tok : $MAX_INPUT_TOKENS"
echo "Batch size    : $BATCH_SIZE"
echo "Start         : $(date)"

python evaluation/run_baselines_v2.py \
    --method "$METHOD" \
    --qna-dir "$QNA_DIR" \
    --suites $SUITES \
    --output-dir "$OUT_DIR" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --bootstrap "$BOOTSTRAP" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
