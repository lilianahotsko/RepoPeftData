#!/bin/bash
#SBATCH --job-name=eval_hnet_drc8k
#SBATCH --output=slurm_logs/eval_hnet_drc8k_%j.out
#SBATCH --error=slurm_logs/eval_hnet_drc8k_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Evaluate Code2LoRA + DRC 8K checkpoint on cr_test and ir_test.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Eval: Code2LoRA DRC8K ====="
echo "Start: $(date)"

CKPT_PATH="$CKPT_DIR/HYPERNET/drc8k"

python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$CKPT_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --splits cr_test ir_test

RESULTS_BASE="${CKPT_PATH}_results"
for split in cr_test ir_test; do
    if [[ -f "$RESULTS_BASE/$split/results.json" ]]; then
        cp "$RESULTS_BASE/$split/results.json" "$BASELINES_DIR/hypernet_drc8k_${split}.json"
        echo "Copied $split results"
    fi
done

echo "Done: $(date)"
