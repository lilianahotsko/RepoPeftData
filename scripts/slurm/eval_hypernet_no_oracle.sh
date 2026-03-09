#!/bin/bash
#SBATCH --job-name=eval_hnet
#SBATCH --output=slurm_logs/eval_hnet_%j.out
#SBATCH --error=slurm_logs/eval_hnet_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate Hypernetwork (no-oracle) checkpoint on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: Hypernetwork (no-oracle) ====="
echo "Start: $(date)"

CKPT_PATH="$CKPT_DIR/HYPERNET/no_oracle"
if [[ ! -d "$CKPT_PATH" ]] && [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$CKPT_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --splits cr_test ir_test

# Results go to {checkpoint_dir}_results/{split}/results.json
# Copy to BASELINES for consistency
RESULTS_BASE="${CKPT_PATH}_results"
if [[ -f "$RESULTS_BASE/cr_test/results.json" ]]; then
    cp "$RESULTS_BASE/cr_test/results.json" "$BASELINES_DIR/hypernet_no_oracle_cr_test.json"
    echo "Copied cr_test results to $BASELINES_DIR/hypernet_no_oracle_cr_test.json"
fi
if [[ -f "$RESULTS_BASE/ir_test/results.json" ]]; then
    cp "$RESULTS_BASE/ir_test/results.json" "$BASELINES_DIR/hypernet_no_oracle_ir_test.json"
    echo "Copied ir_test results to $BASELINES_DIR/hypernet_no_oracle_ir_test.json"
fi

echo "Done: $(date)"
