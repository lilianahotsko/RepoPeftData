#!/bin/bash
#SBATCH --job-name=t2l-code-drc4-eval
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/text2lora_code_drc_v4_eval_%j.out
#SBATCH --error=logs/text2lora_code_drc_v4_eval_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

export PYTHONPATH="$PWD/text2lora/src:$PYTHONPATH"

HYPERMOD_DIR="${1:?Usage: sbatch $0 <hypermod_dir>}"
CODE_EMB_PATH="$SCRATCH/TEXT2LORA_DATA/code_embeddings.pt"

echo "==== Text2LoRA Code-Conditioned + DRC v4 Evaluation ===="
echo "Checkpoint: $HYPERMOD_DIR"
echo "Code embeddings: $CODE_EMB_PATH"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- Evaluating on $SPLIT ---"
    python baselines/text2lora/evaluate_text2lora_code.py \
        --hypermod-dir "$HYPERMOD_DIR" \
        --code-emb-path "$CODE_EMB_PATH" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --text2lora-dir text2lora \
        --use-oracle \
        --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
        --max-oracle-tokens 4096 \
        --max-input-tokens 8192 \
        --output "$BASELINES_DIR/text2lora_code_drc_v4_${SPLIT}.json" \
        --batch-size 8
done

echo "==== Done: $(date) ===="
