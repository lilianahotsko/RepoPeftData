#!/bin/bash
# Per-commit GRU eval on the matched-distribution OOD set.
# Uses the SAME protocol as training (h_k per commit, not h_T), and records
# the full per-commit timeline so we can plot accuracy vs. commit position.
#
# Inputs:
#   CHECKPOINT (env)   = path to code2lora_gru_best.pt
#   PARQUET_DIR (env)  = $SCRATCH/REPO_DATASET/commit_parquet_ood_matched (default)
# Outputs:
#   $OUTPUT_JSON       = bench_ood_matched_per_commit.json
#                         (suites.cross_repo_ood_test.{final, per_commit_timeline, per_repo})

#SBATCH --job-name=eval_gru_ood_matched
#SBATCH --output=slurm_logs/eval_gru_ood_matched_%j.out
#SBATCH --error=slurm_logs/eval_gru_ood_matched_%j.err
#SBATCH --time=24:00:00
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

: "${CHECKPOINT:?must set CHECKPOINT=/path/to/code2lora_gru_best.pt}"
PARQUET_DIR="${PARQUET_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood_matched}"
OUT_DIR="${OUT_DIR:-$(dirname "$CHECKPOINT")/bench_results}"
mkdir -p "$OUT_DIR"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/bench_ood_matched_per_commit.json}"

# Timeline knobs:
#   - timeline-mode=all records every commit (best for the decay plot)
#   - max-assertions-per-commit=0 means "score everything at each commit"
#   - max-assertions-final=0 = no cap on the headline pass either
TIMELINE_MODE="${TIMELINE_MODE:-all}"
MAX_PC="${MAX_PC:-64}"     # per-commit assertion cap (default 64 = matches paper-protocol cost)
MAX_FINAL="${MAX_FINAL:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

echo "Checkpoint : $CHECKPOINT"
echo "Parquet    : $PARQUET_DIR"
echo "Output     : $OUTPUT_JSON"
echo "Timeline   : $TIMELINE_MODE  | max_per_commit=$MAX_PC  | max_final=$MAX_FINAL  | bootstrap=$BOOTSTRAP"

python hypernetwork/eval_code2lora_gru_commits_metrics.py \
    --checkpoint   "$CHECKPOINT" \
    --parquet-dir  "$PARQUET_DIR" \
    --parquet-prefer concat \
    --suites       cross_repo_ood_test \
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
s = d.get("cross_repo_ood_test") or {}
fin = s.get("final") or {}
print(f"  final  EM = {fin.get('em_pct','?'):>6}%")
print(f"  final  EditSim = {fin.get('edit_similarity','?')}")
print(f"  final  CodeBLEU = {fin.get('code_bleu','?')}")
print(f"  n_repos_scored = {fin.get('n_repos_scored','?')}")
PY
