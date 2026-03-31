#!/bin/bash
# Create oracle_lora_drc_v4 symlinks for Text2LoRA training.
# Run this after per-repo LoRA + DRC v4 training completes.
#
# Usage: bash scripts/slurm/setup_oracle_lora_drc_v4.sh

set -euo pipefail

SRC_BASE="$SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA_DRC_V4_8K"
DST_BASE="text2lora/train_outputs/sft/oracle_lora_drc_v4"

mkdir -p "$DST_BASE"

count=0
for repo_dir in "$SRC_BASE"/*/; do
    # repo_dir is e.g. .../PER_REPO_LORA_DRC_V4_8K/author/
    author=$(basename "$repo_dir")
    # Skip non-directory entries
    [ -d "$repo_dir" ] || continue

    for name_dir in "$repo_dir"/*/; do
        name=$(basename "$name_dir")
        adapter_dir="$name_dir/adapter"
        [ -d "$adapter_dir" ] || continue
        [ -f "$adapter_dir/adapter_model.safetensors" ] || continue

        slug="${author}__${name}"
        dst="$DST_BASE/$slug"
        mkdir -p "$dst"

        # Symlink adapter files
        ln -sf "$adapter_dir/adapter_config.json" "$dst/adapter_config.json"
        ln -sf "$adapter_dir/adapter_model.safetensors" "$dst/adapter_model.safetensors"

        # Write args.yaml
        cat > "$dst/args.yaml" << EOF
model_dir: Qwen/Qwen2.5-Coder-1.5B
train_ds_names:
- ${slug}
EOF
        count=$((count + 1))
    done
done

echo "Created $count oracle LoRA symlinks in $DST_BASE"
