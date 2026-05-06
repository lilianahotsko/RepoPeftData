#!/bin/bash
# Run the unified RepoPeftBench eval driver across the canonical bench
# (cr_test, ir_test) and the OOD bench (ood_test from commit_parquet_ood).
#
# Outputs JSON results into $OUT_DIR named ``bench_<method>_<benchname>.json``.
#
# Required env: METHOD (direct|gru_file|gru_commit), CHECKPOINT.
# Optional env: PARQUET_DIR, OOD_PARQUET_DIR, MAX_INPUT_TOKENS, BOOTSTRAP, OUT_DIR.

#SBATCH --job-name=eval_repopeft_bench
#SBATCH --output=slurm_logs/eval_repopeft_bench_%j.out
#SBATCH --error=slurm_logs/eval_repopeft_bench_%j.err
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

METHOD="${METHOD:-gru_commit}"
CHECKPOINT="${CHECKPOINT:?must set CHECKPOINT=/path/to/code2lora_gru_best.pt}"
OUT_DIR="${OUT_DIR:-$(dirname "$CHECKPOINT")/bench_results}"
mkdir -p "$OUT_DIR"

# Allow opting into / out of individual suites. Default = all three. Useful for
# avoiding the (memory-hungry) IR-test gru_commit pass when only OOD is needed,
# and for resuming after a partial crash.
SUITES="${SUITES:-cr ir ood}"
have_suite() { [[ " $SUITES " == *" $1 "* ]]; }

PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
OOD_PARQUET_DIR="${OOD_PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SEED="${SEED:-3407}"

# Bench JSONs (canonical RepoPeftBench)
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET}"
CR_TEST_JSON="${CR_TEST_JSON:-$SPLITS_DIR/gru_cr_test.json}"
IR_TEST_JSON="${IR_TEST_JSON:-$SPLITS_DIR/gru_ir_test.json}"

# Common args ------------------------------------------------------------
COMMON=(
  --method        "$METHOD"
  --checkpoint    "$CHECKPOINT"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --bootstrap     "$BOOTSTRAP"
  --seed          "$SEED"
)

# For gru_commit we additionally need the commit-parquet for the canonical
# bench (to replay diffs of cr_test/ir_test repos) and a separate OOD parquet
# for the OOD column.
if [ "$METHOD" = "gru_commit" ]; then
  COMMON+=(--parquet-dir "$PARQUET_DIR")
fi

run_eval() {
  local label="$1"
  shift
  local out="$OUT_DIR/bench_${METHOD}_${label}.json"
  echo
  echo "===== ${METHOD^^} on ${label} ====="
  python evaluation/run_repopeft_bench.py "${COMMON[@]}" "$@" \
    --output-json "$out"
}

# Canonical CR-test
if have_suite cr; then
  if [ "$METHOD" = "gru_commit" ]; then
    run_eval "cr_test" --bench-json "$CR_TEST_JSON" --cross-repo-splits cr_test
  else
    run_eval "cr_test" --bench-json "$CR_TEST_JSON"
  fi
fi

# Canonical IR-test
if have_suite ir; then
  if [ "$METHOD" = "gru_commit" ]; then
    run_eval "ir_test" --bench-json "$IR_TEST_JSON" --cross-repo-splits train
  else
    run_eval "ir_test" --bench-json "$IR_TEST_JSON"
  fi
fi

# OOD-test (only run if the OOD parquet exists)
if have_suite ood && { [ -f "$OOD_PARQUET_DIR/commits.parquet" ] || [ -d "$OOD_PARQUET_DIR/commits" ]; }; then
  if [ "$METHOD" = "gru_commit" ]; then
    # gru_commit needs to replay diffs from the OOD parquet -- it must use the
    # OOD parquet as the parquet source (not the canonical one).
    OOD_COMMON=(
      --method        "$METHOD"
      --checkpoint    "$CHECKPOINT"
      --max-input-tokens "$MAX_INPUT_TOKENS"
      --bootstrap     "$BOOTSTRAP"
      --seed          "$SEED"
      --parquet-dir   "$OOD_PARQUET_DIR"
      --cross-repo-splits ood_test
    )
    out="$OUT_DIR/bench_${METHOD}_ood_test.json"
    echo
    echo "===== ${METHOD^^} on ood_test ====="
    python evaluation/run_repopeft_bench.py "${OOD_COMMON[@]}" \
      --bench-parquet-dir "$OOD_PARQUET_DIR" \
      --bench-cross-repo-splits ood_test \
      --output-json "$out"
  else
    out="$OUT_DIR/bench_${METHOD}_ood_test.json"
    echo
    echo "===== ${METHOD^^} on ood_test ====="
    python evaluation/run_repopeft_bench.py "${COMMON[@]}" \
      --bench-parquet-dir "$OOD_PARQUET_DIR" \
      --bench-cross-repo-splits ood_test \
      --output-json "$out"
  fi
else
  if ! have_suite ood; then
    echo "[skip] ood suite disabled by SUITES=$SUITES"
  else
    echo "[skip] OOD parquet not found at $OOD_PARQUET_DIR -- run build_ood_parquet.sh first."
  fi
fi

echo "Done: $(date)"
