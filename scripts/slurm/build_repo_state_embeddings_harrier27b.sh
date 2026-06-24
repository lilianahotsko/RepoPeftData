#!/bin/bash
# Sharded whole-repo (repo-state) embeddings with microsoft/harrier-oss-v1-27b
# (decoder embedder: last-token pooling + L2-norm -> 5376-d). Uses a SEPARATE
# blob cache dir (cache vectors are encoder-specific) and a NEW dataset path.
#
# Submit (16 shards):
#   ACCOUNT=rrg-yuntian sbatch --array=0-15 \
#       scripts/slurm/build_repo_state_embeddings_harrier27b.sh
#
# Output:
#   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_shards/repo_state/<split>/shard_<I>_of_16.parquet
#
# Unlike the Qwen3 run, the harrier blob cache starts EMPTY, so every .py blob
# at every snapshot is embedded with the 27B model -- this is the heaviest job
# in the pipeline. Budget the full wall-time and many shards.

#SBATCH --job-name=repo_state_harrier27b
#SBATCH --output=slurm_logs/repo_state_harrier27b_%A_%a.out
#SBATCH --error=slurm_logs/repo_state_harrier27b_%A_%a.err
#SBATCH --time=24:00:00
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
# 27B at 2048-token chunks: keep the per-forward chunk batch small.
BLOB_BATCH="${BLOB_BATCH:-2}"
# Splits to embed (override via SPLITS env). ir_test is derived downstream from
# the train split, so only train/cr_val/cr_test need raw embeddings.
SPLITS="${SPLITS:-train cr_val cr_test}"

INPUT_DIR="${INPUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf/commits}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_shards/repo_state}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/REPO_DATASET/static_commit/cache_harrier27b}"

echo "===== Repo-state embeddings (harrier-27b) shard ${SHARD_INDEX}/${SHARD_TOTAL} ====="
echo "Model   : $MODEL_NAME (pooling=$POOLING dtype=$DTYPE blob_batch=$BLOB_BATCH)"
echo "Input   : $INPUT_DIR"
echo "Output  : $OUT_DIR"
echo "Cache   : $CACHE_ROOT"
echo "Start   : $(date)"
nvidia-smi -L || true

python create_dataset/build_repo_state_embeddings_shard.py \
    --input-dir "$INPUT_DIR" \
    --out-dir "$OUT_DIR" \
    --cache-root "$CACHE_ROOT" \
    --splits $SPLITS \
    --shard-index "$SHARD_INDEX" \
    --shard-total "$SHARD_TOTAL" \
    --model-name "$MODEL_NAME" \
    --pooling "$POOLING" \
    --dtype "$DTYPE" \
    --blob-batch "$BLOB_BATCH" \
    --device cuda

echo "Done: $(date)"
