#!/bin/bash
# Download pretrained Doc-to-LoRA checkpoint from HuggingFace.
# Run this once before evaluation.
#
# Usage: bash scripts/slurm/download_doc2lora.sh

set -euo pipefail

source scripts/slurm/common.sh

TARGET_DIR="doc2lora/trained_d2l"

echo "==== Downloading Doc-to-LoRA checkpoint ===="
echo "Target: $TARGET_DIR"

huggingface-cli download SakanaAI/doc-to-lora \
    --local-dir "$TARGET_DIR" \
    --include "*/"

echo "==== Done ===="
ls -lh "$TARGET_DIR/gemma_demo/checkpoint-80000/pytorch_model.bin" 2>/dev/null || \
    echo "WARNING: checkpoint not found at expected path"
