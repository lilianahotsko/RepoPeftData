#!/bin/bash
# Quick status check for the static-commit Code2LoRA pipeline.
set -uo pipefail
: "${CKPT_DIR:=/scratch/lhotsko/TRAINING_CHECKPOINTS}"

JOBS=(
  "13805695:GRU per-commit eval v2 (incremental writes)"
  "13805696:Static embed (re-fill ir_val)"
  "13805698:Splits ir_test shard 0/2"
  "13805700:Splits ir_test shard 1/2"
  "13805701:Splits cr_val shard 0/2"
  "13805703:Splits cr_val shard 1/2"
  "13805704:Splits ir_val (after embed)"
  "13805970:Merge sharded splits"
  "13805972:Train Code2LoRA-direct"
  "13805974:Static per-commit eval"
  "13747158:Splits build (ood_test)  [done earlier]"
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
