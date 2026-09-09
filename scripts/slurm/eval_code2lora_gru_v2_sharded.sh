#!/bin/bash
#SBATCH --job-name=eval_c2l_gru_v2_sh
#SBATCH --output=slurm_logs/eval_c2l_gru_v2_sh_%A_%a.out
#SBATCH --error=slurm_logs/eval_c2l_gru_v2_sh_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Sharded evaluation of a *v2* Code2LoRA-GRU checkpoint with task metrics
# (EM / EditSim / CodeBLEU). One array task = one suite x one repo shard.
# Per-(repo, commit) atomic writes mean a wall-time kill never loses more
# than the one commit currently in flight, and re-launching auto-resumes.
#
# Required env:
#   CKPT       Path to gru_head.*.pt produced by train_code2lora_gru_v2.py
#
# Optional:
#   NUM_SHARDS (default 4)
#   SUITES     (default "ir_val ir_test cr_val cr_test")
#   COMMITS_DIR / QNAS_DIR / SUFFIX / ...
#
# Submit:
#   CKPT=$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/gru_head.best.pt \
#     NUM_SHARDS=4 \
#     sbatch --array=0-15 scripts/slurm/eval_code2lora_gru_v2_sharded.sh
#
# Merge after array completes:
#   python evaluation/merge_eval_shards.py --auto-detect --input-dir $OUT_DIR

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

CKPT="${CKPT:?CKPT is required (path to gru_head.*.pt)}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B}"
COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
QNAS_DIR="${QNAS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
SUITES_STR="${SUITES:-ir_val ir_test cr_val cr_test}"
read -r -a SUITES <<< "$SUITES_STR"
# Task-index order used at TRAINING time. Single-task checkpoints keep the
# default (a no-op). Multi-task checkpoints (e.g. c2l_evo_multitask) MUST pass
# the same order so each QnA's 'task' column selects the right task embedding,
# e.g. TASKS="assert_rhs qa".
TASKS_STR="${TASKS:-assert_rhs}"
read -r -a TASKS_ARR <<< "$TASKS_STR"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUFFIX="${SUFFIX:-h100_v2_gru_3ep_best}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU_EVAL_V2/${SUFFIX}"
mkdir -p "$OUT_DIR"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-8}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
# Optional: dump per-QnA predictions (question/target/prediction + metrics) as
# JSONL for qualitative inspection. Set PREDICTIONS_OUT to enable.
PREDICTIONS_OUT="${PREDICTIONS_OUT:-}"

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

PRED_ARGS=()
if [ -n "$PREDICTIONS_OUT" ]; then
    PRED_ARGS+=(--predictions-out "${PREDICTIONS_OUT%.jsonl}_${SUITE}_shard${SHARD_I}of${NUM_SHARDS}.jsonl")
fi

echo "===== Eval v2 GRU SHARDED  task ${IDX}/${N_TASKS} ====="
echo "Checkpoint    : $CKPT"
echo "Base model    : $BASE_MODEL"
echo "Suite         : $SUITE (suite_i=$SUITE_I)"
echo "Shard         : $SHARD_I of $NUM_SHARDS"
echo "Tasks         : ${TASKS_ARR[*]}"
echo "Commits dir   : $COMMITS_DIR"
echo "QnAs dir      : $QNAS_DIR"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python evaluation/run_code2lora_gru_v2_eval.py \
    --checkpoint "$CKPT" \
    --base-model "$BASE_MODEL" \
    --commits-dir "$COMMITS_DIR" \
    --qnas-dir "$QNAS_DIR" \
    --suite "$SUITE" \
    --output-dir "$OUT_DIR" \
    --tasks "${TASKS_ARR[@]}" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
    --bootstrap "$BOOTSTRAP" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${PRED_ARGS[@]}"

echo "Done: $(date)"
