#!/bin/bash
#SBATCH --job-name=phase9_ablate
#SBATCH --output=slurm_logs/phase9_ablate_%j.out
#SBATCH --error=slurm_logs/phase9_ablate_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 9: Ablation studies.
# Each ablation trains for 1 epoch, evaluates on cr_val.
# ~10h on H100.

source scripts/slurm/common.sh
mkdir -p slurm_logs

ABLATION_DIR="$CKPT_DIR/ABLATIONS"
EVAL_SPLIT="cr_val_structured"

run_ablation() {
    local name=$1
    shift
    local output_dir="$ABLATION_DIR/$name"

    echo "========================================"
    echo "Ablation: $name  $(date)"
    echo "========================================"

    python hypernetwork/hypernetwork_sampled.py \
        --splits-dir "$SPLITS_DIR" \
        --output-dir "$output_dir" \
        --epochs 1 \
        --eval-steps 500 \
        --save-steps 500 \
        "$@"

    python hypernetwork/hypernetwork_sampled_test.py \
        --checkpoint "$output_dir" \
        --splits-dir "$SPLITS_DIR" \
        --split "$EVAL_SPLIT"
}

echo "===== Phase 9: Ablation studies ====="
echo "Start: $(date)"

# --- LoRA Rank ---
for rank in 4 8 16 32; do
    alpha=$((rank * 2))
    run_ablation "rank_${rank}" --rank $rank --alpha $alpha
done

# --- Hypernetwork Width ---
for width in 256 512 1024; do
    run_ablation "width_${width}" --hidden-dim $width
done

# --- Training Data Size ---
for n_repos in 100 200 300 447; do
    run_ablation "data_${n_repos}repos" --limit-train-repos $n_repos
done

echo "Phase 9 complete: $(date)"
