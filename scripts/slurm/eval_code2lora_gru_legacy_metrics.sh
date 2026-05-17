#!/bin/bash
#SBATCH --job-name=eval_gru_legacy
#SBATCH --output=slurm_logs/eval_gru_legacy_%j.out
#SBATCH --error=slurm_logs/eval_gru_legacy_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

# Evaluate the legacy best Code2LoRA-GRU checkpoint (smart-cap 5ep_full, PAW
# head) with full task metrics (EM / EditSim / CodeBLEU) and per-commit
# timeline tracking, on the canonical eval suites used by the v2 trainers
# and baselines. Apples-to-apples comparison for Table 1.
#
# Triples scored: every (repo, commit_sha, qna) in
#     cr_val / cr_test / ir_val / ir_test
# read from $SCRATCH/REPO_DATASET/commit_parquet_hf/{commits,qna}/*.parquet
# (which has the same commit set + the same canonical val/test QnAs as the
# v2 datasets; smart-cap only filters TRAIN QnAs so eval suites are byte-
# identical).

set -euo pipefail

source scripts/slurm/common.sh
module load arrow/24.0.0 2>/dev/null || module load arrow/18.1.0 2>/dev/null || true
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# --- Inputs --------------------------------------------------------------
CKPT="${CKPT:-$CKPT_DIR/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/code2lora_gru_best.pt}"
PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
SUFFIX="${SUFFIX:-legacy_5ep_smartcap}"
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU_EVAL_V2/${SUFFIX}"
mkdir -p "$OUT_DIR"

# --- Scoring config ------------------------------------------------------
SUITES="${SUITES:-in_repo_val in_repo_test cross_repo_cr_val cross_repo_cr_test}"
TIMELINE_MODE="${TIMELINE_MODE:-all}"   # decay curve over every commit
ASSERTION_MODE="${ASSERTION_MODE:-new}"          # only score the QnAs newly
                                                  # introduced/changed at each
                                                  # commit (matches v2 trainers).
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-32}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LIMIT_REPOS="${LIMIT_REPOS:-0}"

OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/eval_metrics_per_commit.json}"

EXTRA_ARGS=()
if [ "$LIMIT_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-repos "$LIMIT_REPOS")
fi

echo "===== Eval: Legacy Code2LoRA-GRU best (per-commit metrics) ====="
echo "Checkpoint    : $CKPT"
echo "Parquet dir   : $PARQUET_DIR"
echo "Suites        : $SUITES"
echo "Timeline      : $TIMELINE_MODE"
echo "Assertion mode: $ASSERTION_MODE"
echo "Max assertions/commit: $MAX_ASSERTIONS_PER_COMMIT"
echo "Max input tok : $MAX_INPUT_TOKENS"
echo "Max new tok   : $MAX_NEW_TOKENS"
echo "Output JSON   : $OUTPUT_JSON"
echo "Start         : $(date)"

python hypernetwork/eval_code2lora_gru_commits_metrics.py \
    --checkpoint "$CKPT" \
    --parquet-dir "$PARQUET_DIR" \
    --suites $SUITES \
    --timeline-mode "$TIMELINE_MODE" \
    --assertion-mode "$ASSERTION_MODE" \
    --max-assertions-per-commit "$MAX_ASSERTIONS_PER_COMMIT" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-json "$OUTPUT_JSON" \
    --seed 3407 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
