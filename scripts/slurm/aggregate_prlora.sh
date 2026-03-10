#!/bin/bash
#SBATCH --job-name=agg_prlora
#SBATCH --output=slurm_logs/agg_prlora_%j.out
#SBATCH --error=slurm_logs/agg_prlora_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# After all 4 per-repo LoRA chunks complete, run eval-only to aggregate results.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Aggregate Per-repo LoRA results ====="
echo "Start: $(date)"

python baselines/lora_per_repo/run_all_repos.py \
    --splits-dir "$SPLITS_DIR" \
    --output-base "$CKPT_DIR/PER_REPO_LORA" \
    --eval-only \
    --eval-split ir_test \
    --no-wandb

AGG="$CKPT_DIR/PER_REPO_LORA/aggregate_ir_test.json"
if [[ -f "$AGG" ]]; then
    cp "$AGG" "$BASELINES_DIR/per_repo_lora_no_oracle_ir_test.json"
    echo "Copied aggregate to $BASELINES_DIR/"
fi

echo "Done: $(date)"
