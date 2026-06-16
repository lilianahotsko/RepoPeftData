#!/bin/bash
#SBATCH --job-name=smoke_c2l_gru_mgpu
#SBATCH --output=slurm_logs/smoke_c2l_gru_mgpu_%j.out
#SBATCH --error=slurm_logs/smoke_c2l_gru_mgpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --account=def-yuntian

# Smoke test for the multi-GPU (sharded base) Code2LoRA-GRU v2 trainer.
# Runs 1 epoch on a few train repos at the real max-seq-len so it verifies the
# 3B base shards across 2 GPUs and that the OOM at seq-len 4096 is resolved.
#
#   MODEL_NAME=Qwen/Qwen2.5-Coder-3B SUFFIX=smoke_mgpu_gru_3b \
#       sbatch --account=rrg-yuntian_gpu \
#       scripts/slurm/smoke_train_code2lora_gru_v2_mgpu.sh

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
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-3B}"
SUFFIX="${SUFFIX:-smoke_mgpu_gru_3b}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/${SUFFIX}"
mkdir -p "$OUT_DIR"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-5}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-3}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

EXTRA_ARGS=()
if [ "${GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
    EXTRA_ARGS+=(--gradient-checkpointing)
fi

echo "===== SMOKE: Code2LoRA-GRU v2 (multi-GPU / sharded base) ====="
echo "Model             : $MODEL_NAME"
echo "Device map        : $DEVICE_MAP"
echo "Output dir        : $OUT_DIR"
echo "Max seq len       : $MAX_SEQ_LEN"
echo "Start             : $(date)"
nvidia-smi -L || true

python hypernetwork/train_code2lora_gru_v2_mgpu.py \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --eval-qnas-dir "$EVAL_QNAS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --device-map "$DEVICE_MAP" \
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
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
ls -la "$OUT_DIR"
echo
echo "=== metrics.jsonl ==="
cat "$OUT_DIR/metrics.jsonl" 2>/dev/null || echo "(no metrics file)"
