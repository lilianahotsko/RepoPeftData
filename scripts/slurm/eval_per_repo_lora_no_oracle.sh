#!/bin/bash
#SBATCH --job-name=eval_prlora
#SBATCH --output=slurm_logs/eval_prlora_%j.out
#SBATCH --error=slurm_logs/eval_prlora_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Evaluate Per-repo LoRA (no-oracle) checkpoints on ir_test.
# Uses --eval-only to skip training; evaluates each repo's adapter on its ir_test examples.
# Aggregate results saved to $CKPT_DIR/PER_REPO_LORA/aggregate_ir_test.json

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: Per-repo LoRA (no-oracle) ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --eval-only \
    --eval-split ir_test \
    --no-wandb

# Copy aggregate to BASELINES for consistency
AGG="$CKPT_DIR/PER_REPO_LORA/aggregate_ir_test.json"
if [[ -f "$AGG" ]]; then
    cp "$AGG" "$BASELINES_DIR/per_repo_lora_no_oracle_ir_test.json"
    echo "Copied aggregate to $BASELINES_DIR/per_repo_lora_no_oracle_ir_test.json"
fi

echo "Done: $(date)"
