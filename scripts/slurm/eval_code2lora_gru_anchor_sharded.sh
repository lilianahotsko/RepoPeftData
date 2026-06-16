#!/bin/bash
#SBATCH --job-name=eval_c2l_gru_anchor_sh
#SBATCH --output=slurm_logs/eval_c2l_gru_anchor_sh_%A_%a.out
#SBATCH --error=slurm_logs/eval_c2l_gru_anchor_sh_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Sharded evaluation of an anchor-only Code2LoRA-GRU checkpoint with task
# metrics (EM / EditSim / CodeBLEU). One array task = one suite x one repo
# shard. Same atomic-write / resume semantics as the v2 GRU sharded eval.
#
# Required env:
#   CKPT       Path to gru_head.*.pt produced by
#              hypernetwork/train_code2lora_gru_anchor.py
#
# Optional:
#   NUM_SHARDS    (default 4)
#   SUITES        (default "ir_test cr_test")
#   COMMITS_DIR / SNAPSHOTS_DIR / SUFFIX / ...
#
# Submit (2 suites x 4 shards = 8 array tasks):
#   CKPT=$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_anchor_3ep/gru_head.best.pt \
#     SUFFIX=h100_v2_gru_anchor_3ep_best \
#     sbatch --array=0-7 scripts/slurm/eval_code2lora_gru_anchor_sharded.sh

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

CKPT="${CKPT:?CKPT is required (path to gru_head.*.pt)}"
COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
SUITES_STR="${SUITES:-ir_test cr_test}"
read -r -a SUITES <<< "$SUITES_STR"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUFFIX="${SUFFIX:-h100_v2_gru_anchor_3ep_best}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU_EVAL_V2/${SUFFIX}"
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

echo "===== Eval anchor-only GRU SHARDED  task ${IDX}/${N_TASKS} ====="
echo "Checkpoint    : $CKPT"
echo "Suite         : $SUITE (suite_i=$SUITE_I)"
echo "Shard         : $SHARD_I of $NUM_SHARDS"
echo "Commits dir   : $COMMITS_DIR"
echo "Snapshots dir : $SNAPSHOTS_DIR"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python evaluation/run_code2lora_gru_anchor_eval.py \
    --checkpoint "$CKPT" \
    --commits-dir "$COMMITS_DIR" \
    --snapshots-dir "$SNAPSHOTS_DIR" \
    --suite "$SUITE" \
    --output-dir "$OUT_DIR" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
    --bootstrap "$BOOTSTRAP" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS"

echo "Done: $(date)"
