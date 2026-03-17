#!/bin/bash
#SBATCH --job-name=retrain_slora16
#SBATCH --output=slurm_logs/retrain_slora16_%j.out
#SBATCH --error=slurm_logs/retrain_slora16_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Retrain: Single LoRA r=16 (improved settings) ====="
echo "Start: $(date)"

python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA_R16" \
    --max-seq-length 2048 \
    --rank 16 --lora-alpha 32 \
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 2e-4 \
    --no-wandb

echo ""
echo "===== Evaluate: Single LoRA r=16 ====="
ADAPTER_PATH="$CKPT_DIR/SINGLE_LORA_R16/adapter"
if [[ -d "$ADAPTER_PATH" ]]; then
    echo "--- cr_test ---"
    python baselines/single_lora/test_single_lora.py \
        --adapter "$ADAPTER_PATH" \
        --splits-dir "$SPLITS_DIR" \
        --split cr_test \
        --output "$BASELINES_DIR/single_lora_no_oracle_cr_test.json"

    echo "--- ir_test ---"
    python baselines/single_lora/test_single_lora.py \
        --adapter "$ADAPTER_PATH" \
        --splits-dir "$SPLITS_DIR" \
        --split ir_test \
        --output "$BASELINES_DIR/single_lora_no_oracle_ir_test.json"
fi

echo "Done: $(date)"
