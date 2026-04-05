#!/bin/bash
# ============================================================
# Doc-to-LoRA + DRC v4 evaluation
# D2L internalizes oracle context per repo, then generates assertions
#
# Prerequisites:
#   1. bash scripts/slurm/download_doc2lora.sh
#   2. Oracle context cache v4 built
#
# Submit: sbatch scripts/slurm/eval_doc2lora_drc_v4.sh
# ============================================================
#SBATCH --job-name=d2l-drc4-eval
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/doc2lora_drc_v4_eval_%j.out
#SBATCH --error=logs/doc2lora_drc_v4_eval_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

export PYTHONPATH="$PWD/doc2lora/src:$PYTHONPATH"

CKPT="doc2lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    echo "Run: bash scripts/slurm/download_doc2lora.sh"
    exit 1
fi

echo "==== Doc-to-LoRA + DRC v4 Evaluation ===="
echo "Checkpoint: $CKPT"
echo "Oracle cache: $SCRATCH/ORACLE_CONTEXT_CACHE_V4"
echo "Start: $(date)"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- Evaluating on $SPLIT ---"
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path "$CKPT" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --use-oracle \
        --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
        --max-oracle-tokens 4096 \
        --output "$BASELINES_DIR/doc2lora_drc_v4_${SPLIT}.json" \
        --batch-size 4
done

echo "==== Done: $(date) ===="
