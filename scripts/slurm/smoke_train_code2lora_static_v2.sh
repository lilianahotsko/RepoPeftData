#!/bin/bash
#SBATCH --job-name=smoke_c2l_static_v2
#SBATCH --output=slurm_logs/smoke_c2l_static_v2_%j.out
#SBATCH --error=slurm_logs/smoke_c2l_static_v2_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

# Smoke test for the v2 static Code2LoRA trainer. Runs 1 epoch on 5 train
# repos with a tiny eval slice and small max-seq-len. Purpose: verify that
#   (a) the retain_graph bug is fixed,
#   (b) loss decreases (autograd reaches head + LoRA),
#   (c) gradient checkpointing works alongside the LoRA wrappers,
#   (d) end-of-epoch checkpoint saves before eval.

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf}"
OUT_DIR="$CKPT_DIR/CODE2LORA_STATIC/smoke_v2_5repos"
mkdir -p "$OUT_DIR"

echo "===== SMOKE: Code2LoRA static v2 ====="
echo "Snapshots dir : $SNAPSHOTS_DIR"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python hypernetwork/train_code2lora_static_v2.py \
    --snapshots-dir "$SNAPSHOTS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name Qwen/Qwen2.5-Coder-1.5B \
    --rank 16 --alpha 32 --head-hidden-dim 1024 \
    --epochs 1 --lr 1e-4 \
    --max-seq-len 2048 \
    --lm-micro-batch 2 \
    --max-qna-per-snapshot 8 \
    --eval-every-steps 0 \
    --eval-suites cr_val \
    --primary-eval-suite cr_val \
    --limit-eval-snapshots 5 \
    --limit-train-repos 5 \
    --seed 3407

echo "Done: $(date)"
ls -la "$OUT_DIR"
echo
echo "=== metrics.jsonl ==="
cat "$OUT_DIR/metrics.jsonl" 2>/dev/null || echo "(no metrics file)"
