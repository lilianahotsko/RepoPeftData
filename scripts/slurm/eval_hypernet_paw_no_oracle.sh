#!/bin/bash
#SBATCH --job-name=eval_hpaw
#SBATCH --output=slurm_logs/eval_hpaw_%j.out
#SBATCH --error=slurm_logs/eval_hpaw_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate PAW Hypernetwork (no-oracle) best checkpoint on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: PAW Hypernetwork (no-oracle) ====="
echo "Start: $(date)"

CKPT_PATH="$CKPT_DIR/HYPERNET_PAW/no_oracle"
if [[ ! -f "$CKPT_PATH/lora_mapper_best.pt" ]]; then
    echo "ERROR: lora_mapper_best.pt not found at $CKPT_PATH"
    exit 1
fi

echo "--- CR Test ---"
python hypernetwork/hypernetwork_paw_test.py \
    --checkpoint "$CKPT_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test

echo "--- IR Test ---"
python hypernetwork/hypernetwork_paw_test.py \
    --checkpoint "$CKPT_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --split ir_test

# Copy results to BASELINES for consistency
RESULTS_BASE="${CKPT_PATH}_results"
if [[ -f "$RESULTS_BASE/cr_test/results.json" ]]; then
    cp "$RESULTS_BASE/cr_test/results.json" "$BASELINES_DIR/hypernet_paw_no_oracle_cr_test.json"
    echo "Copied cr_test results to $BASELINES_DIR/"
fi
if [[ -f "$RESULTS_BASE/ir_test/results.json" ]]; then
    cp "$RESULTS_BASE/ir_test/results.json" "$BASELINES_DIR/hypernet_paw_no_oracle_ir_test.json"
    echo "Copied ir_test results to $BASELINES_DIR/"
fi

echo "Done: $(date)"
