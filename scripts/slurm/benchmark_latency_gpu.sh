#!/bin/bash
#SBATCH --job-name=bench_latency_gpu
#SBATCH --output=slurm_logs/bench_latency_gpu_%j.out
#SBATCH --error=slurm_logs/bench_latency_gpu_%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-yuntian_gpu

# Per-query inference latency for all methods on the shared backbone (yeRc-W1/W6).
set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LORA="${LORA:-$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000}"

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python analysis/rebuttal/latency_bench.py \
    --model Qwen/Qwen2.5-Coder-1.5B \
    --lora "$LORA" \
    --batch-size 4 --reps 20

echo "Done: $(date)"
