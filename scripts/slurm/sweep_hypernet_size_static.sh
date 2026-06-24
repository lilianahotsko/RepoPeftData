#!/bin/bash
# Hypernetwork-size ablation sweep for Code2LoRA-direct / static (base fixed).
#
# Submits ONE training job per hypernetwork size by re-`sbatch`-ing
# `scripts/slurm/train_code2lora_static_v2.sh` with HEAD_HIDDEN overridden.
# The static variant has no GRU, so head trunk width is the only capacity
# axis. The base model (Qwen2.5-Coder-1.5B) is identical across all points.
#
#   head_hidden   ~hypernet params   single-H100-80GB
#   512           ~0.37B             comfortable
#   1024          ~0.75B (default)   comfortable
#   2048          ~1.5B              fine
#   4096          ~3.0B              near ceiling
#
# This is a *submit* launcher: run it on the login node, NOT via sbatch.
#
# Usage:
#   bash scripts/slurm/sweep_hypernet_size_static.sh
#   HEAD_SIZES="1024 2048" bash scripts/slurm/sweep_hypernet_size_static.sh

set -euo pipefail
cd /home/lhotsko/RepoPeftData

# Head trunk widths to sweep (override with HEAD_SIZES="...").
HEAD_SIZES="${HEAD_SIZES:-512 1024 2048 4096}"

# Optional Slurm account override (e.g. ACCOUNT=rrg-yuntian). When set it is
# passed as `sbatch --account=...`, overriding the script's #SBATCH header.
ACCOUNT="${ACCOUNT:-}"
ACCOUNT_ARGS=()
[ -n "$ACCOUNT" ] && ACCOUNT_ARGS+=(--account "$ACCOUNT")

echo "===== Code2LoRA-static hypernet-size sweep ====="
echo "Head widths : $HEAD_SIZES"
echo "Submitting  : $(date)"

for H in $HEAD_SIZES; do
    SUFFIX="h100_v2_static_3ep_h${H}"
    echo "--- HEAD_HIDDEN=$H SUFFIX=$SUFFIX"
    HEAD_HIDDEN="$H" \
    SUFFIX="$SUFFIX" \
        sbatch "${ACCOUNT_ARGS[@]}" \
               --job-name="c2l_static_h${H}" \
               scripts/slurm/train_code2lora_static_v2.sh
done

echo "All Code2LoRA-static size jobs submitted: $(date)"
