#!/bin/bash
#SBATCH --job-name=scale_h623
#SBATCH --output=slurm_logs/scale_h623_%j.out
#SBATCH --error=slurm_logs/scale_h623_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

EXPANDED_SPLITS="$SCRATCH/REPO_DATASET/expanded"
OUT="$CKPT_DIR/HYPERNET/scale_623repos"

echo "===== Scaling: Train hypernetwork with ALL 623 repos (expanded splits) ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$EXPANDED_SPLITS" \
    --output-dir "$OUT" \
    --max-seq-len 8192 \
    --epochs 3 --grad-accum 8

echo "--- Eval on CR test ---"
python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$OUT" \
    --splits-dir "$EXPANDED_SPLITS" \
    --split cr_test

RESULTS_DIR="${OUT}_results/cr_test"
if [[ -f "$RESULTS_DIR/results.json" ]]; then
    cp "$RESULTS_DIR/results.json" "$BASELINES_DIR/hypernet_scale_623_cr_test.json"
    echo "Copied results to $BASELINES_DIR/hypernet_scale_623_cr_test.json"
fi

echo "Done: $(date)"
