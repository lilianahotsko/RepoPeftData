#!/bin/bash
# OOD eval of the released v2 Code2LoRA-GRU checkpoint, using the precomputed
# v2 OOD commits parquet (diff + repo-state embeddings) and the canonical OOD
# QnA set. Per-commit protocol (h_k walk), same harness used for cr_*/ir_*.
#
# Env knobs:
#   CHECKPOINT   path to code2lora_gru.pt        (default: cached HF snapshot)
#   COMMITS_DIR  dir with commits/ood_test.parquet
#   QNAS_DIR     dir with qna/ood_test.parquet
#   SMOKE=1      quick validation: 1/16 shard, qnas-per-commit-limit=4
#   QPC          qnas-per-commit-limit (default 8 = trainer eval cap)

#SBATCH --job-name=eval_gru_v2_ood
#SBATCH --output=slurm_logs/eval_gru_v2_ood_%j.out
#SBATCH --error=slurm_logs/eval_gru_v2_ood_%j.err
#SBATCH --gres=gpu:h100:1
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HUB="$HOME/.cache/huggingface/hub"
CKPT_SNAP="$HUB/models--code2lora--code2lora-gru/snapshots/26460cba5f0de9b708277488ae0b1a826435961e"
DS="$SCRATCH/REPO_DATASET"

CHECKPOINT="${CHECKPOINT:-$CKPT_SNAP/code2lora_gru.pt}"
COMMITS_DIR="${COMMITS_DIR:-$DS/commit_parquet_hf_v2}"
QNAS_DIR="${QNAS_DIR:-$DS/commit_parquet_hf_v2_ood}"

# QUANTIZE in {"", "4bit", "8bit"}. A separate output dir per precision keeps
# the resumable per-(repo,commit) cache from colliding with the fp run.
QUANTIZE="${QUANTIZE:-}"
QUANT_ARGS=()
PREC_TAG="fp16"
if [[ -n "$QUANTIZE" ]]; then
    QUANT_ARGS=(--quantize "$QUANTIZE")
    PREC_TAG="$QUANTIZE"
fi

OUT_DIR="${OUT_DIR:-/home/lhotsko/RepoPeftData/evaluation/results_ood_published/$PREC_TAG}"
mkdir -p "$OUT_DIR"

QPC="${QPC:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_I="${SHARD_I:-0}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    NUM_SHARDS=16
    SHARD_I=0
    QPC=4
fi

echo "Checkpoint  : $CHECKPOINT"
echo "Commits dir : $COMMITS_DIR  (commits/ood_test.parquet)"
echo "QnAs dir    : $QNAS_DIR  (qna/ood_test.parquet)"
echo "Output dir  : $OUT_DIR"
echo "Precision   : $PREC_TAG  (quantize='${QUANTIZE}')"
echo "Shard       : $SHARD_I / $NUM_SHARDS   qnas-per-commit=$QPC   smoke=${SMOKE:-0}"

python evaluation/run_code2lora_gru_v2_eval.py \
    --checkpoint "$CHECKPOINT" \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --suite ood_test \
    --output-dir "$OUT_DIR" \
    --device cuda \
    --qnas-per-commit-limit "$QPC" \
    --bootstrap 5000 \
    --num-shards "$NUM_SHARDS" \
    --shard-i "$SHARD_I" \
    "${QUANT_ARGS[@]}" \
    "$@"

echo "Done: $(date)"
SHARD_SUFFIX=""
if [[ "$NUM_SHARDS" -gt 1 ]]; then SHARD_SUFFIX="_shard${SHARD_I}of${NUM_SHARDS}"; fi
OUT_JSON="$OUT_DIR/gru_v2_ood_test${SHARD_SUFFIX}.json"
echo "Summary ($OUT_JSON):"
python3 - "$OUT_JSON" <<'PY'
import json, sys, os
p=sys.argv[1]
if not os.path.exists(p):
    print("  (no output json found)"); sys.exit(0)
d=json.load(open(p))
s=d.get("summary") or {}
print(f"  suite        = {s.get('suite','?')}")
print(f"  exact_match  = {s.get('exact_match','?')}  CI={s.get('exact_match_ci','?')}")
print(f"  edit_sim     = {s.get('edit_similarity','?')}")
print(f"  code_bleu    = {s.get('code_bleu','?')}")
print(f"  n_qnas       = {s.get('n','?') or s.get('n_qnas','?')}")
PY
