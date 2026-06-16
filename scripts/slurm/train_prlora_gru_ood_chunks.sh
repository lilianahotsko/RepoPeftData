#!/bin/bash
#SBATCH --job-name=prlora_gru_ood
#SBATCH --output=slurm_logs/prlora_gru_ood_chunk%a_%A.out
#SBATCH --error=slurm_logs/prlora_gru_ood_chunk%a_%A.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Per-repo LoRA on the GRU OOD dataset variant - Table 3 row.
# Each of the 92 OOD repos was internally split by commit_idx (or randomly if
# only one commit exists) into a train/test fraction via
#   scripts/preprocess_gru_ood_temporal_split.py
# producing $SCRATCH/REPO_DATASET_GRU_OOD/{train,ir_test}.json.
# We then train a per-repo LoRA on the early-commit train portion and (later)
# evaluate it on the late-commit ir_test portion using the standard pipeline.
#
# Submit:
#   sbatch --array=0-1 scripts/slurm/train_prlora_gru_ood_chunks.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

NUM_CHUNKS="${NUM_CHUNKS:-2}"
TOTAL_REPOS="${TOTAL_REPOS:-92}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

CHUNK_SIZE=$(( (TOTAL_REPOS + NUM_CHUNKS - 1) / NUM_CHUNKS ))
REPO_OFFSET=$(( CHUNK_ID * CHUNK_SIZE ))
LIMIT_REPOS=$CHUNK_SIZE

GRU_OOD_SPLITS_DIR="${GRU_OOD_SPLITS_DIR:-$SCRATCH/REPO_DATASET_GRU_OOD}"
OUTPUT_BASE="${OUTPUT_BASE:-$CKPT_DIR/PER_REPO_LORA_GRU_OOD}"
EPOCHS="${EPOCHS:-3}"

if [[ ! -f "$GRU_OOD_SPLITS_DIR/train.json" ]] || [[ ! -f "$GRU_OOD_SPLITS_DIR/ir_test.json" ]]; then
    echo "[error] missing $GRU_OOD_SPLITS_DIR/{train,ir_test}.json" >&2
    echo "       run: python scripts/preprocess_gru_ood_temporal_split.py" >&2
    exit 1
fi

echo "===== Per-repo LoRA GRU-OOD chunk ${CHUNK_ID}/${NUM_CHUNKS}: offset ${REPO_OFFSET}, limit ${LIMIT_REPOS} ====="
echo "Splits dir : $GRU_OOD_SPLITS_DIR"
echo "Output dir : $OUTPUT_BASE"
echo "Epochs     : $EPOCHS"
echo "Start      : $(date)"
nvidia-smi -L || true

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$GRU_OOD_SPLITS_DIR" \
    --output-base "$OUTPUT_BASE" \
    --repo-offset "$REPO_OFFSET" \
    --limit-repos "$LIMIT_REPOS" \
    --eval-split ir_test \
    --epochs "$EPOCHS" \
    --no-eval \
    --no-wandb

echo "Done: $(date)"
