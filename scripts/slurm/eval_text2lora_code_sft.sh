#!/bin/bash
# ============================================================
# Evaluate Text2LoRA code-conditioned SFT model on RepoPeftBench
# Runs CR + IR test splits, with and without oracle (DRC) context.
#
# Usage: sbatch scripts/slurm/eval_text2lora_code_sft.sh
# ============================================================
#SBATCH --job-name=t2l-sft-eval
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/text2lora_code_sft_eval_%j.out
#SBATCH --error=logs/text2lora_code_sft_eval_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

echo "==== Text2LoRA Code SFT Evaluation ===="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

export PYTHONPATH="$(pwd)/text2lora/src:$PYTHONPATH"

# Find latest SFT run
SFT_DIR="text2lora/train_outputs/sft/hyper_lora"
if [ ! -d "$SFT_DIR" ]; then
    echo "ERROR: No SFT runs found at $SFT_DIR"
    exit 1
fi

LATEST_RUN=$(ls -td "$SFT_DIR"/code_sft_* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No code_sft_* runs found"
    exit 1
fi
echo "Using run: $LATEST_RUN"

CODE_EMB="$SCRATCH/TEXT2LORA_DATA/code_embeddings.pt"
ORACLE_DIR="$SCRATCH/REPO_DATASET/oracle_cache"
OUT_BASE="$BASELINES_DIR/text2lora_code_sft"
mkdir -p "$OUT_BASE"

for SPLIT in cr_test ir_test; do
    echo ""
    echo "=== $SPLIT (no oracle) ==="
    python baselines/text2lora/evaluate_text2lora_code.py \
        --hypermod-dir "$LATEST_RUN" \
        --code-emb-path "$CODE_EMB" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --output "$OUT_BASE/${SPLIT}.json"

    echo ""
    echo "=== $SPLIT + DRC oracle ==="
    python baselines/text2lora/evaluate_text2lora_code.py \
        --hypermod-dir "$LATEST_RUN" \
        --code-emb-path "$CODE_EMB" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --use-oracle \
        --oracle-cache-dir "$ORACLE_DIR" \
        --output "$OUT_BASE/${SPLIT}_oracle.json"
done

echo ""
echo "==== Done: $(date) ===="
echo "Results in $OUT_BASE/"
