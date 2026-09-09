#!/bin/bash
#SBATCH --job-name=compute_budget
#SBATCH --output=slurm_logs/compute_budget_%j.out
#SBATCH --error=slurm_logs/compute_budget_%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-yuntian_gpu

# Per-query compute budget (input tokens + approximate FLOPs + measured latency)
# per method, to show Code2LoRA uses far less inference compute than RAG/DRC
# [yeRc-W2 / Suggestion 2].
#
# The tokens/FLOPs analysis is CPU-only and deterministic; add --measure-latency
# to also time ms/query on the shared backbone (this is why the job requests a GPU).
#
# CPU-only (tokens + FLOPs, no GPU needed):
#   module load StdEnv/2023 python/3.12 gcc/12.3 arrow
#   source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
#   QNA_DIR=$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna \
#     python analysis/rebuttal/compute_budget.py
#
# With measured latency (GPU):
#   sbatch scripts/slurm/compute_budget.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

QNA_DIR="$QNA_DIR" python analysis/rebuttal/compute_budget.py \
    --base-model Qwen/Qwen2.5-Coder-1.5B \
    --qna-dir "$QNA_DIR" \
    --out-lens 8 64 \
    --measure-latency --latency-batch 4 --latency-reps 20

echo "Done: $(date)"
