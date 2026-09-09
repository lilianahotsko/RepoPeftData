#!/bin/bash
#SBATCH --job-name=bench_c2l_adapt_gpu
#SBATCH --output=slurm_logs/bench_c2l_adapt_gpu_%j.out
#SBATCH --error=slurm_logs/bench_c2l_adapt_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-yuntian_gpu

# GPU (H100) wall-clock benchmark of c2l adapter generation, the GPU counterpart
# to benchmark_c2l_adapt_cpu.sh. Same protocol per repo (fresh cache each repo):
#   1. full     : adapt at HEAD~1  -> complete snapshot + GRU walk
#   2. incr     : adapt at HEAD    -> exactly 1 new commit (1 embed + 1 GRU step)
#   3. cached   : adapt again, unchanged -> pure cache hit
#
# Runs offline against local clones + local HF snapshots; the base LLM is never
# loaded (generation stage only). Report the GPU model with the numbers.

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export C2L_OFFLINE=1
export C2L_DEVICE=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CKPT_SNAP="$HOME/.cache/huggingface/hub/models--code2lora--code2lora-gru/snapshots"
export C2L_CKPT="$(ls -d "$CKPT_SNAP"/*/code2lora_gru.pt | head -1)"

REPOS=(
    "SAML-Toolkits/python-saml"
    "jpsca/jinjax"
    "fastapi/sqlmodel"
)
SRC_ROOT="$SCRATCH/REPO_DATASET/repositories"
BENCH_DIR="${SLURM_TMPDIR:-/tmp}/c2l_bench_gpu"
mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR"   # away from the repo root so the venv's editable c2l wins

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
lscpu | rg "Model name|^CPU\(s\)|MHz" || lscpu | head -20
echo "== checkpoint: $C2L_CKPT =="
python -c "import c2l; print('c2l version', c2l.__version__, c2l.__file__)"
python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

fmt_secs() { printf '%.1f' "$1"; }

echo
printf "%-28s %10s %10s %10s %8s\n" "repo" "full_s" "incr_s" "cached_s" "commits"
SUMMARY=""
for R in "${REPOS[@]}"; do
    NAME="${R//\//_}"
    WORK="$BENCH_DIR/$NAME"
    rm -rf "$WORK"
    cp -r "$SRC_ROOT/$R" "$WORK"
    git -C "$WORK" rev-parse HEAD > /dev/null

    export C2L_ADAPTERS_DIR="$BENCH_DIR/cache_$NAME"
    rm -rf "$C2L_ADAPTERS_DIR"
    OUT="$BENCH_DIR/adapter_$NAME"

    HEAD_SHA=$(git -C "$WORK" rev-parse HEAD)
    git -C "$WORK" checkout -q "HEAD~1"

    T0=$(date +%s.%N)
    c2l adapt "$WORK" --task assert_rhs -o "$OUT" \
        > "$BENCH_DIR/log_${NAME}_full.txt" 2>&1
    T1=$(date +%s.%N)
    FULL=$(python -c "print($T1-$T0)")

    git -C "$WORK" checkout -q "$HEAD_SHA"
    T0=$(date +%s.%N)
    c2l adapt "$WORK" --task assert_rhs -o "$OUT" \
        > "$BENCH_DIR/log_${NAME}_incr.txt" 2>&1
    T1=$(date +%s.%N)
    INCR=$(python -c "print($T1-$T0)")

    T0=$(date +%s.%N)
    c2l adapt "$WORK" --task assert_rhs -o "$OUT" \
        > "$BENCH_DIR/log_${NAME}_cached.txt" 2>&1
    T1=$(date +%s.%N)
    CACHED=$(python -c "print($T1-$T0)")

    NCOMMITS=$(git -C "$WORK" rev-list --first-parent --count HEAD)
    LINE=$(printf "%-28s %10s %10s %10s %8s" "$R" \
        "$(fmt_secs "$FULL")" "$(fmt_secs "$INCR")" "$(fmt_secs "$CACHED")" "$NCOMMITS")
    echo "$LINE"
    SUMMARY+="$LINE"$'\n'
done

echo
echo "== summary (also saved to slurm log) =="
echo "$SUMMARY"
# keep per-repo adapt logs for the paper's provenance
for f in "$BENCH_DIR"/log_*.txt; do
    b=$(basename "$f" .txt)
    cp "$f" "/home/lhotsko/RepoPeftData/slurm_logs/${b}_gpu.txt" || true
done
echo "Done: $(date)"
