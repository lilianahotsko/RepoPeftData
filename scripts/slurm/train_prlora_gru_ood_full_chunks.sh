#!/bin/bash
#SBATCH --job-name=prlora_gru_ood_full
#SBATCH --output=slurm_logs/prlora_gru_ood_full_chunk%a_%A.out
#SBATCH --error=slurm_logs/prlora_gru_ood_full_chunk%a_%A.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Per-repo LoRA for the OOD suite (Table 3), comparable-eval variant.
# Train each per-repo adapter on qnas the standard V2 OOD eval did NOT use
# (positions 9..N within each (repo, commit_sha) group, capped per repo via
# scripts/preprocess_gru_ood_full_split.py) and evaluate on the exact 14,783-
# qna subset that sLoRA / FFT / Text2LoRA / Code2LoRA-direct / Code2LoRA-GRU
# were scored on in Table 3.
#
# Inputs (created by preprocess_gru_ood_full_split.py):
#   $SCRATCH/REPO_DATASET_GRU_OOD_FULL/train.json      (~57.7k qnas, 89 repos)
#   $SCRATCH/REPO_DATASET_GRU_OOD_FULL/ir_test.json    (~14.8k qnas, 89 repos)
#
# Submit:
#   sbatch --array=0-7 scripts/slurm/train_prlora_gru_ood_full_chunks.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

NUM_CHUNKS="${NUM_CHUNKS:-8}"
TOTAL_REPOS="${TOTAL_REPOS:-89}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

CHUNK_SIZE=$(( (TOTAL_REPOS + NUM_CHUNKS - 1) / NUM_CHUNKS ))
REPO_OFFSET=$(( CHUNK_ID * CHUNK_SIZE ))
LIMIT_REPOS=$CHUNK_SIZE

SPLITS_DIR_LOCAL="${GRU_OOD_FULL_SPLITS_DIR:-$SCRATCH/REPO_DATASET_GRU_OOD_FULL}"
OUTPUT_BASE="${OUTPUT_BASE:-$CKPT_DIR/PER_REPO_LORA_GRU_OOD_FULL}"
EPOCHS="${EPOCHS:-3}"

if [[ ! -f "$SPLITS_DIR_LOCAL/train.json" ]] || [[ ! -f "$SPLITS_DIR_LOCAL/ir_test.json" ]]; then
    echo "[error] missing $SPLITS_DIR_LOCAL/{train,ir_test}.json" >&2
    echo "       run: python scripts/preprocess_gru_ood_full_split.py" >&2
    exit 1
fi

echo "===== Per-repo LoRA GRU-OOD-full chunk ${CHUNK_ID}/${NUM_CHUNKS}: offset ${REPO_OFFSET}, limit ${LIMIT_REPOS} ====="
echo "Splits dir : $SPLITS_DIR_LOCAL"
echo "Output dir : $OUTPUT_BASE"
echo "Epochs     : $EPOCHS"
echo "Start      : $(date)"
nvidia-smi -L || true

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR_LOCAL" \
    --output-base "$OUTPUT_BASE" \
    --repo-offset "$REPO_OFFSET" \
    --limit-repos "$LIMIT_REPOS" \
    --eval-split ir_test \
    --epochs "$EPOCHS" \
    --no-eval \
    --no-wandb

echo "Done: $(date)"
