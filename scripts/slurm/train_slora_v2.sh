#!/bin/bash
#SBATCH --job-name=train_slora_v2
#SBATCH --output=slurm_logs/train_slora_v2_%j.out
#SBATCH --error=slurm_logs/train_slora_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

# Train single-LoRA baseline on the v2 smart-cap train QnAs (same dataset
# as Code2LoRA-GRU v2). Outputs a PEFT adapter consumed by
# evaluation/run_baselines_v2.py --method slora.

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QNA_PARQUET="${QNA_PARQUET:-$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"
SUFFIX="${SUFFIX:-h100_v2}"
OUT_DIR="$CKPT_DIR/SLORA_V2/${SUFFIX}"
mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-2000}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"

EXTRA_ARGS=()
if [ "$MAX_TRAIN_ROWS" != "0" ]; then
    EXTRA_ARGS+=(--max-train-rows "$MAX_TRAIN_ROWS")
fi

echo "===== Train: SLoRA v2 (smart-cap train QnAs) ====="
echo "Model       : $MODEL_NAME"
echo "QnA parquet : $QNA_PARQUET"
echo "Output dir  : $OUT_DIR"
echo "Rank / Alph : $RANK / $ALPHA"
echo "Epochs / LR : $EPOCHS / $LR"
echo "MAX_SEQ_LEN : $MAX_SEQ_LEN"
echo "Batch / GA  : $BATCH_SIZE / $GRAD_ACCUM"
echo "Start       : $(date)"

python baselines/single_lora/train_slora_v2.py \
    --qna-parquet "$QNA_PARQUET" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --rank "$RANK" --alpha "$ALPHA" \
    --epochs "$EPOCHS" --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --save-every-steps "$SAVE_EVERY_STEPS" \
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
