#!/bin/bash
# Model-scaling probe: train and evaluate the direct-projection \method{}
# hypernetwork on Qwen2.5-Coder-0.5B / 1.5B / 3B and emit a 3-point scaling
# curve for Section 6 of the paper.
#
# Submit with::
#
#   sbatch --array=0-2 scripts/slurm/scaling_direct_hypernet.sh
#
# Index map:
#   0 -> Qwen2.5-Coder-0.5B
#   1 -> Qwen2.5-Coder-1.5B    (matches paper headline)
#   2 -> Qwen2.5-Coder-3B

#SBATCH --job-name=hnet_scale
#SBATCH --output=slurm_logs/hnet_scale_%A_%a.out
#SBATCH --error=slurm_logs/hnet_scale_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

case "${SLURM_ARRAY_TASK_ID:-1}" in
  0) MODEL_NAME="Qwen/Qwen2.5-Coder-0.5B" ; TAG=qwen0p5b ;;
  1) MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B" ; TAG=qwen1p5b ;;
  2) MODEL_NAME="Qwen/Qwen2.5-Coder-3B"   ; TAG=qwen3b   ;;
  *) echo "ERROR: unhandled SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2 ; exit 2 ;;
esac

OUT_DIR="$CKPT_DIR/HYPERNET_SCALING/${TAG}"
mkdir -p "$OUT_DIR"

echo "===== Direct hypernet scaling: $MODEL_NAME ====="
echo "  -> $OUT_DIR"
echo "Start: $(date)"

python hypernetwork/hypernetwork_sampled.py \
  --splits-dir "$SPLITS_DIR" \
  --output-dir "$OUT_DIR" \
  --model-name "$MODEL_NAME" \
  --max-seq-len 8192 \
  --epochs 3 \
  --grad-accum 8

# Eval the resulting checkpoint with bootstrap CIs through the unified driver.
CKPT="$OUT_DIR/best.pt"
[ -f "$CKPT" ] || CKPT=$(ls -t "$OUT_DIR"/checkpoint-*/best.pt 2>/dev/null | head -1 || true)
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
  python evaluation/run_repopeft_bench.py \
    --method direct \
    --checkpoint "$CKPT" \
    --bench-json "$SPLITS_DIR/cr_test.json" \
    --model-name "$MODEL_NAME" \
    --bootstrap 5000 \
    --output-json "$OUT_DIR/bench_cr_test.json"
  python evaluation/run_repopeft_bench.py \
    --method direct \
    --checkpoint "$CKPT" \
    --bench-json "$SPLITS_DIR/ir_test.json" \
    --model-name "$MODEL_NAME" \
    --bootstrap 5000 \
    --output-json "$OUT_DIR/bench_ir_test.json"
fi

echo "Done: $(date)"
