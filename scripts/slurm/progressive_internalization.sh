#!/bin/bash
#SBATCH --job-name=prog_intern
#SBATCH --output=slurm_logs/prog_intern_%j.out
#SBATCH --error=slurm_logs/prog_intern_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

CHECKPOINT="$CKPT_DIR/HYPERNET/no_oracle"
OUTPUT="$BASELINES_DIR/progressive_internalization_cr_test.json"

echo "===== Progressive Internalization Experiment ====="
echo "Checkpoint: $CHECKPOINT"
echo "Split: cr_test"
echo "Start: $(date)"

python hypernetwork/eval_progressive.py \
    --checkpoint "$CHECKPOINT" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --mode both \
    --max-files 30 \
    --file-steps "0,1,2,3,5,10,15,20,30" \
    --limit-pairs-per-repo 30 \
    --output "$OUTPUT"

echo "Done: $(date)"
echo "Results: $OUTPUT"
