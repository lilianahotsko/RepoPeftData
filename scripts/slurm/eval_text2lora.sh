#!/bin/bash
# ============================================================
# Evaluate trained Text2LoRA on CR-test and IR-test
# Submit: sbatch scripts/slurm/eval_text2lora.sh <hypermod_dir>
# Example:
#   sbatch scripts/slurm/eval_text2lora.sh \
#       text2lora/train_outputs/recon/hyper_lora/20260326-002600_WtDNtJbn
# ============================================================
#SBATCH --job-name=t2l-eval
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/text2lora_eval_%j.out
#SBATCH --error=logs/text2lora_eval_%j.err

set -euo pipefail
mkdir -p logs

source scripts/slurm/common.sh

export PYTHONPATH="$PWD/text2lora/src:$PYTHONPATH"

HYPERMOD_DIR="${1:?Usage: sbatch $0 <hypermod_dir>}"

echo "==== Text2LoRA Evaluation ===="
echo "Checkpoint: $HYPERMOD_DIR"

for SPLIT in cr_test ir_test; do
    echo -e "\n--- Evaluating on $SPLIT ---"
    python baselines/text2lora/evaluate_text2lora.py \
        --hypermod-dir "$HYPERMOD_DIR" \
        --split "$SPLIT" \
        --splits-dir "$SPLITS_DIR" \
        --text2lora-dir text2lora \
        --output "$SCRATCH/BASELINES/text2lora_text_${SPLIT}.json" \
        --batch-size 8
done

echo "==== Done: $(date) ===="
