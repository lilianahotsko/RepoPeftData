#!/bin/bash
#SBATCH --job-name=c2l_latency
#SBATCH --output=slurm_logs/c2l_latency_%j.out
#SBATCH --error=slurm_logs/c2l_latency_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Latency benchmark for Code2LoRA serving backends x workflow stages, on the
# demo repo (like-dbg). Runs one process per backend (infer) plus one adapt
# timing for the node's device.
#
# Submit (GPU: hf/8bit/4bit):
#   DEVICE=cuda BACKENDS="hf 8bit 4bit" \
#     sbatch --gres=gpu:h100:1 --account=rrg-yuntian_gpu scripts/slurm/latency_bench.sh
# Submit (CPU: gguf):
#   DEVICE=cpu BACKENDS="gguf" \
#     sbatch --account=def-yuntian_cpu scripts/slurm/latency_bench.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

REPO=/home/lhotsko/RepoPeftData
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/c2l_terminal${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
# Local multitask checkpoint so adapt timing needs no network.
export C2L_CKPT="${C2L_CKPT:-$CKPT_DIR/CODE2LORA_GRU/c2l_evo_multitask_v2/gru_head.best.pt}"
export C2L_LLAMACPP="${C2L_LLAMACPP:-/scratch/lhotsko/llama.cpp}"

DEVICE="${DEVICE:-cuda}"
BACKENDS="${BACKENDS:-hf 8bit 4bit}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
OUTDIR="${OUTDIR:-$SCRATCH/demo_repos/latency}"
mkdir -p "$OUTDIR"

echo "===== Code2LoRA latency bench  device=$DEVICE backends=[$BACKENDS] ====="
echo "Checkpoint : $C2L_CKPT"
echo "Threads    : $THREADS"
echo "Out dir    : $OUTDIR"
echo "Start      : $(date)"
[ "$DEVICE" = "cuda" ] && (nvidia-smi -L || true)

# Run from a neutral cwd so the stale repo-root ./c2l package can't shadow
# c2l_terminal/c2l on sys.path.
cd /tmp

for BK in $BACKENDS; do
    echo "----- infer: backend=$BK -----"
    python "$REPO/analysis/rebuttal/latency_bench.py" \
        --backend "$BK" --device "$DEVICE" --mode infer \
        --threads "$THREADS" \
        --out "$OUTDIR/infer_${BK}_${DEVICE}.json"
done

echo "----- adapt: device=$DEVICE -----"
python "$REPO/analysis/rebuttal/latency_bench.py" \
    --backend hf --device "$DEVICE" --mode adapt \
    --out "$OUTDIR/adapt_${DEVICE}.json"

echo "Done: $(date)"
