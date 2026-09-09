#!/bin/bash
#SBATCH --job-name=train_slora_v2_res
#SBATCH --output=slurm_logs/train_slora_v2_resume_%j.out
#SBATCH --error=slurm_logs/train_slora_v2_resume_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

# Resume SLoRA v2 training from an existing adapter checkpoint and run until
# the planned total_steps for the epoch, then write adapter/final.
#
# Required:
#   RESUME_ADAPTER  Path to adapter-XXXXX dir (e.g. adapter-16000)
#
# Example (3B, continue from step 16000):
#   RESUME_ADAPTER=$CKPT_DIR/SLORA_V2/h100_v2_qwen3b/adapter-16000 \
#   MODEL_NAME=Qwen/Qwen2.5-Coder-3B SUFFIX=h100_v2_qwen3b \
#     sbatch --account=rrg-yuntian_gpu scripts/slurm/train_slora_v2_resume.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESUME_ADAPTER="${RESUME_ADAPTER:?RESUME_ADAPTER is required}"
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

echo "===== Resume: SLoRA v2 ====="
echo "Model          : $MODEL_NAME"
echo "Resume adapter : $RESUME_ADAPTER"
echo "QnA parquet   : $QNA_PARQUET"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python baselines/single_lora/train_slora_v2.py \
    --qna-parquet "$QNA_PARQUET" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL_NAME" \
    --resume-adapter "$RESUME_ADAPTER" \
    --rank "$RANK" --alpha "$ALPHA" \
    --epochs "$EPOCHS" --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --save-every-steps "$SAVE_EVERY_STEPS" \
    --seed 3407

echo "Done: $(date)"
