#!/bin/bash
#SBATCH --job-name=phase3_train
#SBATCH --output=slurm_logs/phase3_train_%j.out
#SBATCH --error=slurm_logs/phase3_train_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 3: Train + eval fine-tuned and single LoRA baselines.
# ~8h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 3: Training baselines ====="
echo "Start: $(date)"

# --- Fine-tuned ---
echo "--- Fine-tuned: Training ---"
python baselines/finetuned/train_finetuned.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/FINETUNED" \
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 2e-5 \
    --no-wandb

echo "--- Fine-tuned: Eval (cr_test) ---"
python baselines/finetuned/test_finetuned.py \
    --model-path "$CKPT_DIR/FINETUNED/final" \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --output "$BASELINES_DIR/finetuned_cr_test.json"

echo "--- Fine-tuned: Eval (ir_test) ---"
python baselines/finetuned/test_finetuned.py \
    --model-path "$CKPT_DIR/FINETUNED/final" \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --output "$BASELINES_DIR/finetuned_ir_test.json"

# --- Single LoRA ---
echo "--- Single LoRA: Training ---"
python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/SINGLE_LORA" \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --no-wandb

echo "--- Single LoRA: Eval (cr_test) ---"
python baselines/single_lora/test_single_lora.py \
    --adapter "$CKPT_DIR/SINGLE_LORA/adapter" \
    --splits-dir "$SPLITS_DIR" --split cr_test \
    --output "$BASELINES_DIR/single_lora_cr_test.json"

echo "--- Single LoRA: Eval (ir_test) ---"
python baselines/single_lora/test_single_lora.py \
    --adapter "$CKPT_DIR/SINGLE_LORA/adapter" \
    --splits-dir "$SPLITS_DIR" --split ir_test \
    --output "$BASELINES_DIR/single_lora_ir_test.json"

echo "Phase 3 complete: $(date)"
