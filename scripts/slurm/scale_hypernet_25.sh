#!/bin/bash
#SBATCH --job-name=scale_h25
#SBATCH --output=slurm_logs/scale_h25_%j.out
#SBATCH --error=slurm_logs/scale_h25_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

N=25
OUT="$CKPT_DIR/HYPERNET/scale_${N}repos"

echo "===== Scaling: Train hypernetwork with $N repos ====="
echo "Start: $(date)"

python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUT" \
    --limit-train-repos $N \
    --max-seq-len 8192 \
    --epochs 3 --grad-accum 8

echo "--- Eval on CR test ---"
python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$OUT" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test

RESULTS_DIR="${OUT}_results/cr_test"
if [[ -f "$RESULTS_DIR/results.json" ]]; then
    cp "$RESULTS_DIR/results.json" "$BASELINES_DIR/hypernet_scale_${N}_cr_test.json"
    echo "Copied results to $BASELINES_DIR/hypernet_scale_${N}_cr_test.json"
fi

echo "Done: $(date)"
