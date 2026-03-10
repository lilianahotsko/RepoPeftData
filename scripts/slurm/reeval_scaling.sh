#!/bin/bash
#SBATCH --job-name=reeval_scale
#SBATCH --output=slurm_logs/reeval_scaling_%j.out
#SBATCH --error=slurm_logs/reeval_scaling_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Re-evaluate scaling experiments (50/100/200 repos) with fixed metrics.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Re-eval: Scaling experiments (fixed metrics) ====="
echo "Start: $(date)"

for N in 50 100 200; do
    OUT="$CKPT_DIR/HYPERNET/scale_${N}repos"
    echo ""
    echo "========== SCALE: $N repos =========="
    if [[ -d "$OUT" ]]; then
        python hypernetwork/hypernetwork_sampled_test.py \
            --checkpoint "$OUT" \
            --splits-dir "$SPLITS_DIR" \
            --split cr_test

        RESULTS_DIR="${OUT}_results/cr_test"
        if [[ -f "$RESULTS_DIR/results.json" ]]; then
            cp "$RESULTS_DIR/results.json" "$BASELINES_DIR/hypernet_scale_${N}_cr_test.json"
            echo "Copied scale_${N} results"
        fi
    else
        echo "WARN: Checkpoint not found at $OUT"
    fi
done

echo ""
echo "Done: $(date)"
