#!/bin/bash
#SBATCH --job-name=train_c2l_gru_mgpu
#SBATCH --output=slurm_logs/train_c2l_gru_mgpu_%j.out
#SBATCH --error=slurm_logs/train_c2l_gru_mgpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --account=def-yuntian

# Multi-GPU (model-sharded) Code2LoRA-GRU v2 trainer.
#
# The frozen base LLM is sharded across the 2 requested GPUs via device_map,
# while the trainable Code2LoRAHead + CommitGRU + optimizer sit on cuda:0.
# All training hyperparameters match scripts/slurm/train_code2lora_gru_v2.sh;
# the only difference is the base is split across GPUs (option 2 for the 3B OOM).
#
# Submit (e.g. to the rrg allocation) with::
#
#   MODEL_NAME=Qwen/Qwen2.5-Coder-3B SUFFIX=h100x2_gru_3ep_qwen3b \
#       sbatch --account=rrg-yuntian_gpu scripts/slurm/train_code2lora_gru_v2_mgpu.sh

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
SUFFIX="${SUFFIX:-h100x2_gru_3ep}"
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
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-3407}"

EXTRA_ARGS=()
if [ "$LIMIT_TRAIN_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
fi
if [ "${GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
    EXTRA_ARGS+=(--gradient-checkpointing)
fi

echo "===== Train: Code2LoRA-GRU v2 (multi-GPU / sharded base) ====="
echo "Model             : $MODEL_NAME"
echo "Device map        : $DEVICE_MAP"
echo "Commits dir       : $COMMITS_DIR"
echo "Train QnAs dir    : $QNAS_DIR"
echo "Eval suite QnAs   : $EVAL_QNAS_DIR"
echo "Output dir        : $OUT_DIR"
echo "Epochs / LR       : $EPOCHS / $LR"
echo "Max seq len       : $MAX_SEQ_LEN"
echo "Seed              : $SEED"
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
