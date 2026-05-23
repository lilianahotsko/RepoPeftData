#!/bin/bash
# Train Code2LoRA v1 (legacy hypernetwork_sampled.py) on the v2 dataset:
# the anchor-snapshot splits at
#   $SCRATCH/REPO_DATASET/static_commit/splits_at_anchor/{train,cr_val,cr_test}.json
#
# This is the same data that fed `nanigock/repopeft-code2lora-snapshots`
# (the static-v2 trainer reads its parquet form), so v1 and v2 trainers
# train on identical anchor snapshots + identical QnAs.
#
# Output goes to a *new* location, kept separate from any prior v1 ckpts:
#   $CKPT_DIR/CODE2LORA_DIRECT/c2l_v1_data_v2/

#SBATCH --job-name=train_c2l_v1_dataV2
#SBATCH --output=slurm_logs/train_c2l_v1_dataV2_%j.out
#SBATCH --error=slurm_logs/train_c2l_v1_dataV2_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Inputs: v2 anchor-snapshot splits in legacy JSON layout ---------------
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET/static_commit/splits_at_anchor}"
REPOS_ROOT="${REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories}"

# --- Output: new location, never overwrite previous v1 ckpts ---------------
SUFFIX="${SUFFIX:-c2l_v1_data_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/CODE2LORA_DIRECT/$SUFFIX}"
mkdir -p "$OUTPUT_DIR"

# --- Hyperparams: keep the v1 defaults (paper-baseline settings) -----------
EPOCHS="${EPOCHS:-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"   # v1 hypernet head: small (512), unlike v2 (1024)
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-3407}"

# Optional bounds (useful for smoke tests) -- 0 means "no limit"
LIMIT_TRAIN_REPOS="${LIMIT_TRAIN_REPOS:-0}"
LIMIT_EVAL_REPOS="${LIMIT_EVAL_REPOS:-100}"
LIMIT_TEST_REPOS="${LIMIT_TEST_REPOS:-100}"

EXTRA_ARGS=()
[ "$LIMIT_TRAIN_REPOS" != "0" ] && EXTRA_ARGS+=(--limit-train-repos "$LIMIT_TRAIN_REPOS")
[ "$LIMIT_EVAL_REPOS"  != "0" ] && EXTRA_ARGS+=(--limit-eval-repos  "$LIMIT_EVAL_REPOS")
[ "$LIMIT_TEST_REPOS"  != "0" ] && EXTRA_ARGS+=(--limit-test-repos  "$LIMIT_TEST_REPOS")

echo "===== Train: Code2LoRA v1 on v2 data ====="
echo "Splits dir   : $SPLITS_DIR"
echo "Repos root   : $REPOS_ROOT"
echo "Output dir   : $OUTPUT_DIR"
echo "Epochs / LR  : $EPOCHS / $LR"
echo "max_seq_len  : $MAX_SEQ_LEN  | grad_accum=$GRAD_ACCUM"
echo "hidden_dim   : $HIDDEN_DIM  | rank=$RANK  alpha=$ALPHA"
echo "Start        : $(date)"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir   "$SPLITS_DIR" \
    --repos-root   "$REPOS_ROOT" \
    --output-dir   "$OUTPUT_DIR" \
    --model-name   Qwen/Qwen2.5-Coder-1.5B \
    --rank         "$RANK" \
    --alpha        "$ALPHA" \
    --hidden-dim   "$HIDDEN_DIM" \
    --max-seq-len  "$MAX_SEQ_LEN" \
    --epochs       "$EPOCHS" \
    --grad-accum   "$GRAD_ACCUM" \
    --lr           "$LR" \
    --eval-steps   "$EVAL_STEPS" \
    --save-steps   "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --seed         "$SEED" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
