#!/bin/bash
#SBATCH --job-name=eval_gru
#SBATCH --output=slurm_logs/eval_gru_%j.out
#SBATCH --error=slurm_logs/eval_gru_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

CKPT="${CKPT:-$CKPT_DIR/CODE2LORA_GRU/mamba2_h1024_bptt32_chronological_default}"
MODE="${MODE:-standard}"
SPLIT="${SPLIT:-cr_test}"

echo "===== Eval: Code2LoRA-GRU ====="
echo "Checkpoint: $CKPT"
echo "Mode:       $MODE"
echo "Split:      $SPLIT"
echo "Start: $(date)"

python hypernetwork/eval_code2lora_gru.py \
    --checkpoint "$CKPT" \
    --splits-dir "$SPLITS_DIR" \
    --split "$SPLIT" \
    --mode "$MODE" \
    --commit-steps "0,10,25,50,75,100"

# Also evaluate on IR test if running CR test
if [ "$SPLIT" = "cr_test" ]; then
    echo ""
    echo "===== Also evaluating on ir_test ====="
    python hypernetwork/eval_code2lora_gru.py \
        --checkpoint "$CKPT" \
        --splits-dir "$SPLITS_DIR" \
        --split "ir_test" \
        --mode "$MODE" \
        --commit-steps "0,10,25,50,75,100"
fi

echo "Done: $(date)"
