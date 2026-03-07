#!/bin/bash
# Run hypernetwork training with different base model scales.
# The hypernetwork itself is retrained for each base model (cheap: only MLP params).
# Base model stays frozen.
#
# Usage:
#   bash scripts/run_scale_experiment.sh 0.5B   # Qwen2.5-Coder-0.5B
#   bash scripts/run_scale_experiment.sh 3B     # Qwen2.5-Coder-3B

set -e

SIZE=${1:-"0.5B"}
SCRATCH=${SCRATCH:-$HOME/scratch}
SPLITS_DIR="$SCRATCH/REPO_DATASET"

case $SIZE in
    "0.5B")
        MODEL_NAME="Qwen/Qwen2.5-Coder-0.5B"
        OUTPUT_DIR="$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/scale_0.5B"
        ;;
    "1.5B")
        MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B"
        OUTPUT_DIR="$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/scale_1.5B"
        ;;
    "3B")
        MODEL_NAME="Qwen/Qwen2.5-Coder-3B"
        OUTPUT_DIR="$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/scale_3B"
        ;;
    "7B")
        MODEL_NAME="Qwen/Qwen2.5-Coder-7B"
        OUTPUT_DIR="$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/scale_7B"
        ;;
    *)
        echo "Unknown size: $SIZE. Use 0.5B, 1.5B, 3B, or 7B"
        exit 1
        ;;
esac

echo "========================================"
echo "Scale experiment: $SIZE ($MODEL_NAME)"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Train hypernetwork
python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --rank 16 \
    --alpha 32 \
    --hidden-dim 512 \
    --epochs 3 \
    --lr 1e-4 \
    --eval-steps 1000 \
    --save-steps 1000

# Evaluate on cr_test
python hypernetwork/hypernetwork_sampled_test.py \
    --checkpoint "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test

# Evaluate pretrained baseline for this model size
python baselines/pretrained/test_qwen_coder.py \
    --model-name "$MODEL_NAME" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --output "$SCRATCH/BASELINES/pretrained_${SIZE}_cr_test.json"

echo "Scale experiment $SIZE complete."
