#!/bin/bash
#SBATCH --job-name=eval_gru_parquet
#SBATCH --output=slurm_logs/eval_gru_parquet_%j.out
#SBATCH --error=slurm_logs/eval_gru_parquet_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
module load arrow/18.1.0
mkdir -p slurm_logs

SUFFIX="${SUFFIX:-parquet}"
CKPT_BASE="${CKPT_BASE:-$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}}"
CKPT="${CKPT:-$CKPT_BASE/code2lora_gru_best.pt}"
PARQUET_DIR="${PARQUET_DIR:-$SPLITS_DIR/commit_parquet}"
ASSERTION_MODE="${ASSERTION_MODE:-cumulative}"
MAX_ASSERTIONS_PER_COMMIT="${MAX_ASSERTIONS_PER_COMMIT:-32}"
LIMIT_REPOS="${LIMIT_REPOS:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-$CKPT_BASE/eval_results.json}"

EXTRA_ARGS=()
if [ "$LIMIT_REPOS" != "0" ]; then
    EXTRA_ARGS+=(--limit-repos "$LIMIT_REPOS")
fi

echo "===== Eval: Code2LoRA-GRU (commit-level, Parquet) ====="
echo "Checkpoint:       $CKPT"
echo "Parquet dir:      $PARQUET_DIR"
echo "Assertion mode:   $ASSERTION_MODE"
echo "Output JSON:      $OUTPUT_JSON"
echo "Start: $(date)"

python hypernetwork/eval_code2lora_gru_commits.py \
    --checkpoint "$CKPT" \
    --parquet-dir "$PARQUET_DIR" \
    --assertion-mode "$ASSERTION_MODE" \
    --max-assertions-per-commit "$MAX_ASSERTIONS_PER_COMMIT" \
    --output-json "$OUTPUT_JSON" \
    --suites in_repo_val in_repo_test cross_repo_cr_val cross_repo_cr_test \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
