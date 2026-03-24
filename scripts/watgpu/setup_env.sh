#!/bin/bash
# One-time setup: create venv and download data from HuggingFace.
#
# Usage:
#   bash scripts/watgpu/setup_env.sh
#   bash scripts/watgpu/setup_env.sh --hf-repo nanigock/RepoPeft-data
set -euo pipefail

HF_REPO="${1:-nanigock/RepoPeft-data}"
SCRATCH="${SCRATCH:-$HOME/scratch}"
VENV_DIR="$SCRATCH/venvs/repopeft"
DATA_ROOT="$SCRATCH/repopeft_data"

echo "=========================================="
echo "RepoPeft watgpu setup"
echo "  SCRATCH:   $SCRATCH"
echo "  VENV:      $VENV_DIR"
echo "  DATA:      $DATA_ROOT"
echo "  HF_REPO:   $HF_REPO"
echo "=========================================="

# ── 1. Create venv ──
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n>>> Creating venv..."
    module purge
    module load python/3.12 cuda/12.6 2>/dev/null || true
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch transformers peft datasets accelerate \
                bitsandbytes wandb huggingface_hub codebleu \
                sentencepiece protobuf
    echo "Venv created."
else
    echo -e "\n>>> Venv already exists, activating..."
    source "$VENV_DIR/bin/activate"
fi

# ── 2. Download data from HuggingFace ──
echo -e "\n>>> Downloading dataset from $HF_REPO..."
mkdir -p "$DATA_ROOT"

python - "$HF_REPO" "$DATA_ROOT" <<'PYEOF'
import sys, os
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
data_root = sys.argv[2]

print(f"Downloading {repo_id} -> {data_root}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=data_root,
    allow_patterns=["splits/main/*", "oracle_context_cache/*"],
)
print("Done.")
PYEOF

# ── 3. Verify ──
echo -e "\n>>> Verifying..."
for f in train.json cr_val.json cr_test.json ir_val.json ir_test.json; do
    if [ -f "$DATA_ROOT/splits/main/$f" ]; then
        echo "  OK: splits/main/$f"
    else
        echo "  MISSING: splits/main/$f"
    fi
done

if [ -d "$DATA_ROOT/oracle_context_cache" ]; then
    n=$(ls "$DATA_ROOT/oracle_context_cache/" | wc -l)
    echo "  OK: oracle_context_cache/ ($n files)"
else
    echo "  MISSING: oracle_context_cache/"
fi

echo -e "\n=========================================="
echo "Setup complete. You can now submit training jobs:"
echo "  sbatch scripts/watgpu/train_fft_8k.sh"
echo "  sbatch scripts/watgpu/train_slora_8k.sh"
echo "=========================================="
