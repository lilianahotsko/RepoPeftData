#!/bin/bash
# Render per-commit decay overlay plots for the smart-cap GRU ckpt.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/snapshots/epoch0_for_eval
BENCH="$CKPT_DIR/bench_results/bench_per_commit_timeline_full.json"
OOD_BENCH="${OOD_BENCH:-}"   # optionally point at the matched-OOD per-commit JSON

OUT_DIR=analysis/figures/gru_smartcap_decay
mkdir -p "$OUT_DIR"

args=( --bench-result "$BENCH" )
if [[ -n "$OOD_BENCH" && -f "$OOD_BENCH" ]]; then
  args+=( --bench-result "$OOD_BENCH" )
fi

module load python/3.12 arrow/24.0.0 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

python analysis/plot_per_commit_decay_all_suites.py \
    "${args[@]}" \
    --out-prefix "$OUT_DIR/gru_smartcap_decay"

echo
echo "Figures: $OUT_DIR/"
ls -la "$OUT_DIR/"
