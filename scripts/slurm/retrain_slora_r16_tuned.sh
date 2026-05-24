#!/bin/bash
#SBATCH --job-name=slora16-tuned
#SBATCH --output=slurm_logs/retrain_slora16_tuned_%j.out
#SBATCH --error=slurm_logs/retrain_slora16_tuned_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# v1 sLoRA retrain on the static-snapshot dataset, tuned to escape the
# epoch-1-peak-then-overfit pattern of the original SINGLE_LORA_R16 run.
#
# Original (Table 1):  rank=16 alpha=32 dropout=0 lr=2e-4 epochs=3  ->  eval_loss 0.624 @ ep1, 0.840 @ ep3
#                                                                       EM 45.6 CR / 48.2 IR  (~tied with Pretrained)
#
# Hypothesis tested here: the aggressive peak-LR drives the model into
# an early local minimum that then memorises rather than generalises.
# Slowing the LR + adding LoRA dropout should let the model train for
# more steps before eval loss turns up.

source scripts/slurm/common.sh
mkdir -p slurm_logs

OUT_DIR="$CKPT_DIR/SINGLE_LORA_R16_TUNED"
echo "===== Retrain: Single LoRA r=16 (TUNED) ====="
echo "Start: $(date)"
echo "Out  : $OUT_DIR"

# Lower peak LR (5e-5 vs 2e-4), longer warmup (10% vs 5%), LoRA dropout
# 0.1 vs 0.0, 5 epochs (early-stop on eval_loss keeps the best checkpoint).
python baselines/single_lora/train_single_lora.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUT_DIR" \
    --max-seq-length 2048 \
    --rank 16 --lora-alpha 32 \
    --lora-dropout 0.1 \
    --warmup-ratio 0.10 \
    --epochs 5 --batch-size 4 --grad-accum 8 --lr 5e-5 \
    --no-wandb

echo ""
echo "===== Evaluate: Single LoRA r=16 (tuned) ====="
ADAPTER_PATH="$OUT_DIR/adapter"
if [[ -d "$ADAPTER_PATH" ]]; then
    for split in cr_test ir_test; do
        echo "--- $split ---"
        python baselines/single_lora/test_single_lora.py \
            --adapter "$ADAPTER_PATH" \
            --splits-dir "$SPLITS_DIR" \
            --split "$split" \
            --output "$BASELINES_DIR/single_lora_r16_tuned_${split}.json"
    done
fi

echo "Done: $(date)"
