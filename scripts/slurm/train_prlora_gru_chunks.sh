#!/bin/bash
#SBATCH --job-name=prlora_gru
#SBATCH --output=slurm_logs/prlora_gru_chunk%a_%A.out
#SBATCH --error=slurm_logs/prlora_gru_chunk%a_%A.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Per-repo LoRA on the GRU (commit-derived) dataset variant - Table 2 row.
# Splits the 409 training repos of gru_train.json into 8 chunks (~52 repos each)
# launched as SLURM array tasks. Adapters write to:
#   $CKPT_DIR/PER_REPO_LORA_GRU/<author>/<repo>/adapter
#
# Submit:
#   sbatch --array=0-7 scripts/slurm/train_prlora_gru_chunks.sh
# Resume a single chunk:
#   sbatch --array=3 scripts/slurm/train_prlora_gru_chunks.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

NUM_CHUNKS="${NUM_CHUNKS:-8}"
TOTAL_REPOS="${TOTAL_REPOS:-409}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ceil division so the last chunk picks up any remainder
CHUNK_SIZE=$(( (TOTAL_REPOS + NUM_CHUNKS - 1) / NUM_CHUNKS ))
REPO_OFFSET=$(( CHUNK_ID * CHUNK_SIZE ))
LIMIT_REPOS=$CHUNK_SIZE

GRU_SPLITS_DIR="${GRU_SPLITS_DIR:-$SCRATCH/REPO_DATASET_GRU}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"
OUTPUT_BASE="${OUTPUT_BASE:-$CKPT_DIR/PER_REPO_LORA_GRU}"
EPOCHS="${EPOCHS:-3}"

echo "===== Per-repo LoRA GRU chunk ${CHUNK_ID}/${NUM_CHUNKS}: offset ${REPO_OFFSET}, limit ${LIMIT_REPOS} ====="
echo "Model      : $MODEL_NAME"
echo "Splits dir : $GRU_SPLITS_DIR"
echo "Output dir : $OUTPUT_BASE"
echo "Epochs     : $EPOCHS"
echo "Start      : $(date)"
nvidia-smi -L || true

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$GRU_SPLITS_DIR" \
    --output-base "$OUTPUT_BASE" \
    --model-name "$MODEL_NAME" \
    --repo-offset "$REPO_OFFSET" \
    --limit-repos "$LIMIT_REPOS" \
    --eval-split ir_test \
    --epochs "$EPOCHS" \
    --no-eval \
    --no-wandb

echo "Done: $(date)"
