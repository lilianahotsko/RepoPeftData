#!/bin/bash
# Sharded diff embeddings with the scaled-up encoder microsoft/harrier-oss-v1-27b
# (decoder embedder: last-token pooling + L2-norm -> 5376-d). Writes to a NEW
# dataset path so the Qwen3-0.6B (2048-d) embeddings are left untouched.
#
# Submit (16 shards):
#   ACCOUNT=rrg-yuntian sbatch --array=0-15 \
#       scripts/slurm/build_diff_embeddings_harrier27b.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_shards/diff/<split>/shard_<I>_of_16.parquet
#
# 27B bf16 (~54 GB) fits one H100-80GB for inference. Wall is encoder-bound and
# much slower than the 0.6B model -- budget several hours per shard.

#SBATCH --job-name=diff_emb_harrier27b
#SBATCH --output=slurm_logs/diff_emb_harrier27b_%A_%a.out
#SBATCH --error=slurm_logs/diff_emb_harrier27b_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-microsoft/harrier-oss-v1-27b}"
POOLING="${POOLING:-lasttoken}"
DTYPE="${DTYPE:-bfloat16}"
SHARD_TOTAL="${SHARD_TOTAL:-16}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DIFF_BATCH="${DIFF_BATCH:-64}"
LIMIT="${LIMIT:-0}"
# Splits to embed (override via SPLITS env). ir_test is derived downstream from
# the train split, so only train/cr_val/cr_test need raw embeddings.
SPLITS="${SPLITS:-train cr_val cr_test}"

INPUT_DIR="${INPUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf/commits}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_shards/diff}"

echo "===== Diff embeddings (harrier-27b) shard ${SHARD_INDEX}/${SHARD_TOTAL} ====="
echo "Model   : $MODEL_NAME (pooling=$POOLING dtype=$DTYPE)"
echo "Input   : $INPUT_DIR"
echo "Output  : $OUT_DIR"
echo "Start   : $(date)"
nvidia-smi -L || true

python create_dataset/build_diff_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --splits $SPLITS \
    --shard-index "$SHARD_INDEX" \
    --shard-total "$SHARD_TOTAL" \
    --model-name "$MODEL_NAME" \
    --pooling "$POOLING" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --diff-batch "$DIFF_BATCH" \
    --device cuda \
    --limit "$LIMIT"

echo "Done: $(date)"
