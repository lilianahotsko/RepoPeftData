#!/bin/bash
#SBATCH --job-name=train_c2l_static_v2
#SBATCH --output=slurm_logs/train_c2l_static_v2_%j.out
#SBATCH --error=slurm_logs/train_c2l_static_v2_%j.err
#SBATCH --time=03:00:00
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
MAX_QNA_PER_SNAPSHOT="${MAX_QNA_PER_SNAPSHOT:-16}"
SEED="${SEED:-3407}"
# 4096 = verified-safe ceiling for the 720M-param Code2LoRAHead alongside
# the 1.5B base + LoRA + gradient checkpointing on a single H100-80GB.
# At seq_len=8192 the LM logits alone are ~9 GB at micro_batch=2 and the
# backward OOMs in step 1. 4096 also matches the v2 GRU trainer (fair
# architectural comparison).
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LIMIT_EVAL_SNAPS="${LIMIT_EVAL_SNAPS:-200}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"
LOG_EVERY_ITERS="${LOG_EVERY_ITERS:-50}"
# Default to skipping validation during training: just save per-epoch ckpts
# and run evaluation offline via the sharded baseline / static-eval pipeline.
SKIP_EVAL="${SKIP_EVAL:-1}"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi
if [ "$SKIP_EVAL" = "1" ]; then
    EXTRA_ARGS+=(--skip-eval)
fi

echo "===== Train: Code2LoRA static v2 ====="
echo "Snapshots dir : $SNAPSHOTS_DIR"
echo "Output dir    : $OUT_DIR"
echo "Epochs / LR   : $EPOCHS / $LR"
echo "Seed          : $SEED"
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
    --log-every-iters "$LOG_EVERY_ITERS" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
