#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=slurm_logs/analysis_%j.out
#SBATCH --error=slurm_logs/analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

# Run analysis and figure generation. No GPU needed.

source scripts/slurm/common.sh
mkdir -p slurm_logs analysis/figures analysis/output

echo "===== Analysis Pipeline ====="
echo "Start: $(date)"

echo ""
echo "1. Collecting results..."
python analysis/collect_results.py

echo ""
echo "2. Running comprehensive analysis..."
python analysis/analyze_results.py \
    --results-dir "$BASELINES_DIR" \
    --output-dir analysis/output

echo ""
echo "3. Generating figures..."
python analysis/generate_figures.py \
    --results-dir "$BASELINES_DIR" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir analysis/figures

echo ""
echo "Done: $(date)"
