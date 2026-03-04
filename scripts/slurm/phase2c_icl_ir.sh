#!/bin/bash
#SBATCH --job-name=p2c_icl_ir
#SBATCH --output=slurm_logs/p2c_icl_ir_%j.out
#SBATCH --error=slurm_logs/p2c_icl_ir_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== ICL baseline (IR test) ====="
echo "Start: $(date)"

for shots in 3 5; do
    echo "--- ICL ${shots}-shot (ir_test_structured) ---"
    python baselines/icl/test_icl.py \
        --splits-dir "$SPLITS_DIR" --split ir_test_structured --n-shots $shots \
        --output "$BASELINES_DIR/icl_${shots}shot_ir_test_structured.json"
done

echo "Done: $(date)"
