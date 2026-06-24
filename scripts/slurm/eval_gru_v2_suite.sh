#!/bin/bash
# Generic, array-aware v2 Code2LoRA-GRU eval for any suite (cr_test, ir_test,
# cr_val, ir_val, ood_test) at fp16 or bitsandbytes 4/8-bit, using the released
# checkpoint. Submit as a SLURM array; SHARD_I = $SLURM_ARRAY_TASK_ID.
#
# Required env:
#   SUITE        cr_test | ir_test | cr_val | ir_val | ood_test
#   NUM_SHARDS   must equal the array width (e.g. 0-7 => NUM_SHARDS=8)
# Optional env:
#   QUANTIZE     "" (fp16) | 4bit | 8bit
#   COMMITS_DIR  default $SCRATCH/REPO_DATASET/commit_parquet_hf_v2
#   QNAS_DIR     default $SCRATCH/REPO_DATASET/code2lora_snapshots_hf
#   QPC          qnas-per-commit-limit (default 8)
#
# Output: results_indist/<prec>/<suite>/gru_v2_<suite>_shard<i>of<N>.json
# Merge afterwards with scripts/merge_gru_v2_shards.py.

#SBATCH --job-name=eval_gru_v2_suite
#SBATCH --output=slurm_logs/eval_gru_v2_suite_%A_%a.out
#SBATCH --error=slurm_logs/eval_gru_v2_suite_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian_gpu

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HUB="$HOME/.cache/huggingface/hub"
CKPT_SNAP="$HUB/models--code2lora--code2lora-gru/snapshots/26460cba5f0de9b708277488ae0b1a826435961e"
DS="$SCRATCH/REPO_DATASET"

CHECKPOINT="${CHECKPOINT:-$CKPT_SNAP/code2lora_gru.pt}"
COMMITS_DIR="${COMMITS_DIR:-$DS/commit_parquet_hf_v2}"
QNAS_DIR="${QNAS_DIR:-$DS/code2lora_snapshots_hf}"

: "${SUITE:?must set SUITE=cr_test|ir_test|...}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-${SHARD_I:-0}}"
QPC="${QPC:-8}"

QUANTIZE="${QUANTIZE:-}"
QUANT_ARGS=()
PREC_TAG="fp16"
if [[ -n "$QUANTIZE" ]]; then
    QUANT_ARGS=(--quantize "$QUANTIZE")
    PREC_TAG="$QUANTIZE"
fi

OUT_DIR="${OUT_DIR:-/home/lhotsko/RepoPeftData/evaluation/results_indist/$PREC_TAG/$SUITE}"
mkdir -p "$OUT_DIR"

echo "Checkpoint  : $CHECKPOINT"
echo "Suite       : $SUITE   precision=$PREC_TAG"
echo "Commits dir : $COMMITS_DIR"
echo "QnAs dir    : $QNAS_DIR"
echo "Output dir  : $OUT_DIR"
echo "Shard       : $SHARD_I / $NUM_SHARDS   qnas-per-commit=$QPC"

python evaluation/run_code2lora_gru_v2_eval.py \
    --checkpoint "$CHECKPOINT" \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --suite "$SUITE" \
    --output-dir "$OUT_DIR" \
    --device cuda \
    --qnas-per-commit-limit "$QPC" \
    --bootstrap 0 \
    --num-shards "$NUM_SHARDS" \
    --shard-i "$SHARD_I" \
    "${QUANT_ARGS[@]}" \
    "$@"

echo "Done shard $SHARD_I/$NUM_SHARDS: $(date)"
