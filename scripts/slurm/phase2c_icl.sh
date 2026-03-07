#!/bin/bash
#SBATCH --job-name=p2c_icl
#SBATCH --output=slurm_logs/p2c_icl_%j.out
#SBATCH --error=slurm_logs/p2c_icl_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== ICL baseline ====="
echo "Start: $(date)"

for shots in 3 5; do
    echo "--- ICL ${shots}-shot (cr_test) ---"
    python baselines/icl/test_icl.py \
        --splits-dir "$SPLITS_DIR" --split cr_test --n-shots $shots \
        --max-input-tokens 16384 \
        --output "$BASELINES_DIR/icl_${shots}shot_cr_test.json"
done

echo "Done: $(date)"
