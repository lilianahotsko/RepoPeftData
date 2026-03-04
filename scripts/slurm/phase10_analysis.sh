#!/bin/bash
#SBATCH --job-name=phase10_analysis
#SBATCH --output=slurm_logs/phase10_analysis_%j.out
#SBATCH --error=slurm_logs/phase10_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 10: Analysis, figures, and tables.
# Needs GPU for LoRA visualization (hypernetwork forward passes).
# ~1h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Phase 10: Analysis ====="
echo "Start: $(date)"

# --- Results analysis ---
python analysis/analyze_results.py \
    --results-dir "$BASELINES_DIR" \
    --output-dir analysis/output \
    --hypernet-results "$CKPT_DIR/HYPERNET/full_repos_results/cr_test/results.json"

# --- LoRA visualization ---
python analysis/visualize_loras.py \
    --checkpoint "$CKPT_DIR/HYPERNET/full_repos" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir analysis/figures \
    --limit-repos 50

echo ""
echo "===== Summary of all results ====="
echo ""

# Print all result files
for f in "$BASELINES_DIR"/*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        em=$(python -c "import json; d=json.load(open('$f')); print(f\"{d.get('exact_match_pct',0):.2f}%\")" 2>/dev/null || echo "N/A")
        edit=$(python -c "import json; d=json.load(open('$f')); print(f\"{d.get('edit_similarity',0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $name: EM=$em  EditSim=$edit"
    fi
done

echo ""
echo "Phase 10 complete: $(date)"
echo "Analysis outputs in: analysis/output/"
echo "Figures in: analysis/figures/"
