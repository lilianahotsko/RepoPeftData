#!/bin/bash
# Run ablation studies. Each ablation trains the hypernetwork with one
# parameter changed, then evaluates on cr_val for efficiency.
#
# Usage:
#   bash scripts/run_ablations.sh rank        # LoRA rank ablation
#   bash scripts/run_ablations.sh modules     # Target modules ablation
#   bash scripts/run_ablations.sh depth       # Hypernetwork depth ablation
#   bash scripts/run_ablations.sh width       # Hypernetwork width ablation
#   bash scripts/run_ablations.sh data_size   # Training data size ablation
#   bash scripts/run_ablations.sh all         # Run all ablations

set -e

ABLATION=${1:-"all"}
SCRATCH=${SCRATCH:-$HOME/scratch}
SPLITS_DIR="$SCRATCH/REPO_DATASET"
BASE_OUTPUT="$SCRATCH/TRAINING_CHECKPOINTS/ABLATIONS"
EVAL_SPLIT="cr_val_structured"

run_ablation() {
    local name=$1
    shift
    local output_dir="$BASE_OUTPUT/$name"

    echo "========================================"
    echo "Ablation: $name"
    echo "Output: $output_dir"
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

# Rank ablation
run_rank_ablation() {
    for rank in 4 8 16 32; do
        alpha=$((rank * 2))
        run_ablation "rank_${rank}" --rank $rank --alpha $alpha
    done
}

# Hypernetwork depth ablation (1,2,3,4 layers)
# This requires modifying the hypernetwork architecture.
# For now, we vary hidden_dim as a proxy for capacity.
run_width_ablation() {
    for width in 256 512 1024; do
        run_ablation "width_${width}" --hidden-dim $width
    done
}

# Training data size ablation
run_data_size_ablation() {
    for n_repos in 100 200 300 447; do
        run_ablation "data_${n_repos}repos" --limit-train-repos $n_repos
    done
}

case $ABLATION in
    "rank")
        run_rank_ablation
        ;;
    "width")
        run_width_ablation
        ;;
    "data_size")
        run_data_size_ablation
        ;;
    "all")
        run_rank_ablation
        run_width_ablation
        run_data_size_ablation
        ;;
    *)
        echo "Unknown ablation: $ABLATION"
        echo "Options: rank, width, data_size, all"
        exit 1
        ;;
esac

echo "All ablations complete."
