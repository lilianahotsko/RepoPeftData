#!/bin/bash
#SBATCH --job-name=eval_baseline_v2_sh
#SBATCH --output=slurm_logs/eval_baseline_v2_sh_%A_%a.out
#SBATCH --error=slurm_logs/eval_baseline_v2_sh_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Sharded v2 baseline (pretrained / fft / slora) evaluation.
#
# Uses a SLURM array. Each array task evaluates ONE suite x ONE shard of the
# repos in that suite. The Python driver writes a JSON file after EVERY
# (repo, commit) group, so a wall-time kill never loses more than the one
# commit currently in flight. Re-launching the same array picks up where
# each shard left off (resume is automatic).
#
# Usage examples
# --------------
#   # Pretrained on all 4 suites with 4 repo shards each -> 16 array tasks:
#   METHOD=pretrained NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # FFT with the LATEST checkpoint, 4 suites x 4 shards:
#   METHOD=fft  CKPT=$CKPT_DIR/FFT_V2/h100_v2/checkpoint-44000 \
#     NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # SLoRA on the latest adapter:
#   METHOD=slora CKPT=$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000 \
#     NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
# Array-index decoding: idx = suite_index * NUM_SHARDS + shard_i.
# After all shards finish, merge with:
#   python evaluation/merge_eval_shards.py --auto-detect --input-dir $OUT_DIR

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

METHOD="${METHOD:-pretrained}"
CKPT="${CKPT:-}"
QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
SUITES_STR="${SUITES:-ir_val ir_test cr_val cr_test}"
read -r -a SUITES <<< "$SUITES_STR"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUFFIX="${SUFFIX:-h100_v2_sharded}"
OUT_DIR="$CKPT_DIR/BASELINES_V2/${METHOD}_${SUFFIX}"
mkdir -p "$OUT_DIR"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-8}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_SUITES=${#SUITES[@]}
N_TASKS=$(( N_SUITES * NUM_SHARDS ))
SUITE_I=$(( IDX / NUM_SHARDS ))
SHARD_I=$(( IDX % NUM_SHARDS ))
if [ "$SUITE_I" -ge "$N_SUITES" ]; then
    echo "[skip] array index $IDX out of range (n_tasks=$N_TASKS)"
    exit 0
fi
SUITE="${SUITES[$SUITE_I]}"

EXTRA_ARGS=()
if [ -n "$CKPT" ]; then
    EXTRA_ARGS+=(--ckpt "$CKPT")
fi
if [ "$QNAS_PER_COMMIT_LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT")
fi

echo "===== Eval baseline v2 SHARDED  task ${IDX}/${N_TASKS} ====="
echo "Method        : $METHOD"
echo "Checkpoint    : ${CKPT:-<none>}"
echo "Suite         : $SUITE (suite_i=$SUITE_I)"
echo "Shard         : $SHARD_I of $NUM_SHARDS"
echo "QnA dir       : $QNA_DIR"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python evaluation/run_baselines_v2.py \
    --method "$METHOD" \
    --qna-dir "$QNA_DIR" \
    --suites "$SUITE" \
    --output-dir "$OUT_DIR" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --bootstrap "$BOOTSTRAP" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
