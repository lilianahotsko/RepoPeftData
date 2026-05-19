#!/bin/bash
#SBATCH --job-name=eval_gru_legacy_sh
#SBATCH --output=slurm_logs/eval_gru_legacy_sh_%A_%a.out
#SBATCH --error=slurm_logs/eval_gru_legacy_sh_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Sharded evaluation of the LEGACY best Code2LoRA-GRU (smart-cap 5ep_full,
# PAW head). Per-(repo) atomic incremental writes => a wall-time kill never
# loses more than the one repo currently in flight; resume is automatic.
#
# One array task = one suite x one repo shard.
#
# Optional env:
#   CKPT, PARQUET_DIR, NUM_SHARDS, SUITES, SUFFIX, TIMELINE_MODE, ASSERTION_MODE
#
# Submit:
#   NUM_SHARDS=4 sbatch --array=0-15 scripts/slurm/eval_gru_legacy_sharded.sh
#
# Merge with:
#   python evaluation/merge_eval_shards.py --auto-detect --input-dir $OUT_DIR --legacy-gru

set -euo pipefail

source scripts/slurm/common.sh
module load arrow/24.0.0 2>/dev/null || module load arrow/18.1.0 2>/dev/null || true
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

CKPT="${CKPT:-$CKPT_DIR/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/code2lora_gru_best.pt}"
PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
SUITES_STR="${SUITES:-in_repo_val in_repo_test cross_repo_cr_val cross_repo_cr_test}"
read -r -a SUITES <<< "$SUITES_STR"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUFFIX="${SUFFIX:-legacy_5ep_smartcap_sh}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU_EVAL_V2/${SUFFIX}"
mkdir -p "$OUT_DIR"

TIMELINE_MODE="${TIMELINE_MODE:-all}"
ASSERTION_MODE="${ASSERTION_MODE:-new}"
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-32}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LIMIT_REPOS="${LIMIT_REPOS:-0}"

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
OUTPUT_JSON="$OUT_DIR/eval_metrics_per_commit.json"

EXTRA_ARGS=()
if [ "$LIMIT_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-repos "$LIMIT_REPOS")
fi

echo "===== Eval Legacy GRU SHARDED  task ${IDX}/${N_TASKS} ====="
echo "Checkpoint    : $CKPT"
echo "Parquet dir   : $PARQUET_DIR"
echo "Suite         : $SUITE (suite_i=$SUITE_I)"
echo "Shard         : $SHARD_I of $NUM_SHARDS"
echo "Timeline      : $TIMELINE_MODE"
echo "Output JSON   : $OUTPUT_JSON  (suffixed per-suite per-shard)"
echo "Start         : $(date)"
nvidia-smi -L || true

python hypernetwork/eval_code2lora_gru_commits_metrics.py \
    --checkpoint "$CKPT" \
    --parquet-dir "$PARQUET_DIR" \
    --suites "$SUITE" \
    --timeline-mode "$TIMELINE_MODE" \
    --assertion-mode "$ASSERTION_MODE" \
    --max-assertions-per-commit "$MAX_ASSERTIONS_PER_COMMIT" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-json "$OUTPUT_JSON" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
