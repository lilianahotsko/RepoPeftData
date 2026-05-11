#!/bin/bash
# Per-commit GRU eval on the canonical RepoPeftBench suites
# (in_repo_val, in_repo_test, cross_repo_cr_val, cross_repo_cr_test) with
# FULL per-commit timeline (every commit, not percentiles). Outputs the same
# JSON schema as the OOD per-commit eval, so the same plotter works.
#
# Inputs:
#   CHECKPOINT (env)   = path to code2lora_gru_best.pt
#   PARQUET_DIR (env)  = $SCRATCH/REPO_DATASET/commit_parquet_hf (default)
#   SUITES (env)       = space-separated suite names; defaults to all four
# Output:
#   $OUTPUT_JSON       = bench_per_commit_timeline_full.json

#SBATCH --job-name=eval_gru_timeline
#SBATCH --output=slurm_logs/eval_gru_timeline_%j.out
#SBATCH --error=slurm_logs/eval_gru_timeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

: "${CHECKPOINT:?must set CHECKPOINT=/path/to/code2lora_gru_best.pt}"
PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf}"
OUT_DIR="${OUT_DIR:-$(dirname "$CHECKPOINT")/bench_results}"
mkdir -p "$OUT_DIR"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/bench_per_commit_timeline_full.json}"

# Default = all four canonical suites. Pass SUITES="cross_repo_cr_test" etc.
# to restrict.
SUITES="${SUITES:-in_repo_val in_repo_test cross_repo_cr_val cross_repo_cr_test}"

# Timeline knobs:
#   --timeline-mode=all keeps every commit's per-commit row.
#   --max-assertions-per-commit=64 matches the prelim run; bump to 0 for "all"
#     (much slower, paper-grade).
TIMELINE_MODE="${TIMELINE_MODE:-all}"
MAX_PC="${MAX_PC:-64}"
MAX_FINAL="${MAX_FINAL:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

echo "Checkpoint : $CHECKPOINT"
echo "Parquet    : $PARQUET_DIR"
echo "Suites     : $SUITES"
echo "Output     : $OUTPUT_JSON"
echo "Timeline   : $TIMELINE_MODE  | max_per_commit=$MAX_PC  | max_final=$MAX_FINAL"

# shellcheck disable=SC2086
python hypernetwork/eval_code2lora_gru_commits_metrics.py \
    --checkpoint   "$CHECKPOINT" \
    --parquet-dir  "$PARQUET_DIR" \
    --parquet-prefer hf \
    --suites $SUITES \
    --assertion-mode new \
    --final-mode last_lora_all_assertions \
    --timeline-mode "$TIMELINE_MODE" \
    --max-assertions-per-commit "$MAX_PC" \
    --max-assertions-final "$MAX_FINAL" \
    --bootstrap "$BOOTSTRAP" \
    --output-json "$OUTPUT_JSON" \
    "$@"

echo "Done: $(date)"
echo
echo "Summary:"
python3 - "$OUTPUT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in d.get("suites", []):
    s = d.get(k) or {}
    fin = s.get("final") or {}
    pr = s.get("per_repo") or {}
    tl_rows = sum(len((r or {}).get("per_commit_timeline") or (r or {}).get("timeline") or []) for r in pr.values())
    print(f"  {k}: final EM={fin.get('em_pct','?')} n={fin.get('n','?')} repos={len(pr)} timeline_rows={tl_rows:,}")
PY
