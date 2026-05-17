#!/bin/bash
#SBATCH --job-name=train_c2l_static_v2
#SBATCH --output=slurm_logs/train_c2l_static_v2_%j.out
#SBATCH --error=slurm_logs/train_c2l_static_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

# Train the v2 *static* Code2LoRA hypernet on precomputed
# `repo_state_embedding`s from the `repopeft-code2lora-snapshots` dataset.
#
# Requires:
#   - `build_code2lora_snapshots` has produced
#     $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/{commits,qna}/*.parquet
#   - Working Qwen2.5-Coder-1.5B in the HF cache.

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
SUFFIX="${SUFFIX:-h100_v2_static_3ep}"
OUT_DIR="$CKPT_DIR/CODE2LORA_STATIC/${SUFFIX}"
mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LM_MICRO_BATCH="${LM_MICRO_BATCH:-2}"
MAX_QNA_PER_SNAPSHOT="${MAX_QNA_PER_SNAPSHOT:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LIMIT_EVAL_SNAPS="${LIMIT_EVAL_SNAPS:-200}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi

echo "===== Train: Code2LoRA static v2 ====="
echo "Snapshots dir : $SNAPSHOTS_DIR"
echo "Output dir    : $OUT_DIR"
echo "Epochs / LR   : $EPOCHS / $LR"
echo "Start         : $(date)"

python hypernetwork/train_code2lora_static_v2.py \
    --snapshots-dir "$SNAPSHOTS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name Qwen/Qwen2.5-Coder-1.5B \
    --rank 16 --alpha 32 --head-hidden-dim 1024 \
    --epochs "$EPOCHS" --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lm-micro-batch "$LM_MICRO_BATCH" \
    --max-qna-per-snapshot "$MAX_QNA_PER_SNAPSHOT" \
    --eval-every-steps "$EVAL_EVERY" \
    --eval-suites cr_val ir_val \
    --primary-eval-suite cr_val \
    --limit-eval-snapshots "$LIMIT_EVAL_SNAPS" \
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
