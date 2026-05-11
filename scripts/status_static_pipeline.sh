#!/bin/bash
# Quick status check for the static-commit Code2LoRA pipeline.
set -uo pipefail
: "${CKPT_DIR:=/scratch/lhotsko/TRAINING_CHECKPOINTS}"

JOBS=(
  "13688463:GRU per-commit eval (in_repo/cr suites)"
  "13699664:Static embed shard 0 (extras)"
  "13699665:Static embed shard 1 (extras)"
  "13699666:Static embed shard 2 (extras)"
  "13699667:Static embed shard 3 (extras)"
  "13747157:Splits build (ir_test)"
  "13747158:Splits build (ood_test)"
  "13747159:Splits build (ir_val)"
  "13747160:Splits build (cr_val)"
  "13747161:Train Code2LoRA-direct"
  "13747162:Static per-commit eval"
)

echo "=== SLURM job statuses ==="
for entry in "${JOBS[@]}"; do
  jid="${entry%%:*}"
  desc="${entry#*:}"
  state=$(sacct -j "$jid" --format=State -n -P 2>/dev/null | head -1)
  elapsed=$(sacct -j "$jid" --format=Elapsed -n -P 2>/dev/null | head -1)
  printf "  %s  %-40s  state=%-12s elapsed=%s\n" "$jid" "$desc" "${state:-?}" "${elapsed:-?}"
done

echo
echo "=== Active queue ==="
squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %.10L %R" 2>/dev/null

echo
echo "=== Output artifacts ==="
for f in \
    /scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv \
    /scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json \
    /scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor/train.json \
    /scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor/cr_val.json \
    /scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor/ir_test.json \
    "$CKPT_DIR/CODE2LORA_DIRECT/static_commit_at_anchor/hypernetwork_best.pt" \
    "$CKPT_DIR/CODE2LORA_DIRECT/static_commit_at_anchor/bench_results/static_per_commit_timeline.json" \
    "$CKPT_DIR/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/snapshots/epoch0_for_eval/bench_results/bench_per_commit_timeline_full.json" \
; do
  if [ -f "$f" ]; then
    sz=$(du -h "$f" 2>/dev/null | cut -f1)
    printf "  %-100s  EXISTS  %s\n" "$f" "$sz"
  else
    printf "  %-100s  missing\n" "$f"
  fi
done

echo
echo "=== Latest log tails (last 5 lines each) ==="
for f in \
    slurm_logs/eval_gru_timeline_13688463.out \
    slurm_logs/build_static_commit_embed_13699664.out \
    slurm_logs/build_static_commit_splits_13699701.out \
    slurm_logs/train_static_commit_13699702.out \
    slurm_logs/eval_static_timeline_13699708.out \
; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    tail -5 "$f"
  fi
done
