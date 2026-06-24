#!/bin/bash
# Hypernetwork-size ablation sweep for Code2LoRA-GRU (base model fixed).
#
# Submits ONE training job per hypernetwork size by re-`sbatch`-ing
# `scripts/slurm/train_code2lora_gru_v2.sh` with HEAD_HIDDEN / GRU_HIDDEN
# overridden. The base model (Qwen2.5-Coder-1.5B) is identical across all
# points -- only the hypernetwork capacity changes.
#
# Capacity is a single joint width multiplier `m`:
#     head_hidden = 1024 * m   ,   gru_hidden = 2048 * m
#
#   m     head/gru      ~hypernet params   single-H100-80GB
#   0.5   512 / 1024    ~0.37B             comfortable
#   1     1024 / 2048   ~0.75B (default)   comfortable
#   2     2048 / 4096   ~1.5B              fine
#   4     4096 / 8192   ~3.3B              near ceiling (see M4 note below)
#
# m=8 (head 8192, ~5.8B) does NOT fit one H100 -- use the multi-GPU trainer
# `hypernetwork/train_code2lora_gru_v2_mgpu.py` if you need to go bigger.
#
# This is a *submit* launcher: run it on the login node, NOT via sbatch.
#
# Usage:
#   bash scripts/slurm/sweep_hypernet_size_gru.sh
#   SIZES="1 2" bash scripts/slurm/sweep_hypernet_size_gru.sh   # subset by m

set -euo pipefail
cd /home/lhotsko/RepoPeftData

# Joint width multipliers to sweep (override with SIZES="...").
SIZES="${SIZES:-0.5 1 2 4}"

# Optional Slurm account override (e.g. ACCOUNT=rrg-yuntian). When set it is
# passed as `sbatch --account=...`, overriding the script's #SBATCH header.
ACCOUNT="${ACCOUNT:-}"
ACCOUNT_ARGS=()
[ -n "$ACCOUNT" ] && ACCOUNT_ARGS+=(--account "$ACCOUNT")

# m=4 (head 4096 / gru 8192) sits near the single-H100 memory ceiling
# (~53 GB optimizer state + 3 GB base + fp32 LM logits). If it OOMs in step 1,
# lower this to 2048 (caveat: that point is then not seq-len-matched to the
# rest of the sweep -- note it in the paper / EXPERIMENT_LOG).
M4_MAX_SEQ_LEN="${M4_MAX_SEQ_LEN:-4096}"

echo "===== Code2LoRA-GRU hypernet-size sweep ====="
echo "Multipliers : $SIZES"
echo "Submitting  : $(date)"

for m in $SIZES; do
    # head = 1024*m, gru = 2048*m -- use awk for the 0.5 case.
    HEAD_HIDDEN=$(awk "BEGIN{printf \"%d\", 1024*$m}")
    GRU_HIDDEN=$(awk "BEGIN{printf \"%d\", 2048*$m}")
    SUFFIX="h100_v2_gru_3ep_h${HEAD_HIDDEN}_g${GRU_HIDDEN}"

    MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
    if [ "$m" = "4" ]; then
        MAX_SEQ_LEN="$M4_MAX_SEQ_LEN"
    fi

    echo "--- m=$m : HEAD_HIDDEN=$HEAD_HIDDEN GRU_HIDDEN=$GRU_HIDDEN "\
"MAX_SEQ_LEN=$MAX_SEQ_LEN SUFFIX=$SUFFIX"
    HEAD_HIDDEN="$HEAD_HIDDEN" \
    GRU_HIDDEN="$GRU_HIDDEN" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    SUFFIX="$SUFFIX" \
        sbatch "${ACCOUNT_ARGS[@]}" \
               --job-name="c2l_gru_h${HEAD_HIDDEN}" \
               scripts/slurm/train_code2lora_gru_v2.sh
done

echo "All Code2LoRA-GRU size jobs submitted: $(date)"
echo "Check: squeue -u $USER --name=c2l_gru_h512,c2l_gru_h1024,c2l_gru_h2048,c2l_gru_h4096"
