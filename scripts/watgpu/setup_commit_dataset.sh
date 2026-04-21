#!/bin/bash
# One-time fetch of the commit-level Parquet dataset from HuggingFace.
#
# Usage:
#   bash scripts/watgpu/setup_commit_dataset.sh
#   bash scripts/watgpu/setup_commit_dataset.sh nanigock/code2lora-gru-commits
#
# After this completes you can submit:
#   sbatch scripts/watgpu/train_code2lora_gru_commits.sh
set -euo pipefail

HF_REPO="${1:-nanigock/code2lora-gru-commits}"
SCRATCH="${SCRATCH:-$HOME/scratch}"
VENV_DIR="$SCRATCH/venvs/repopeft"
DATA_ROOT="$SCRATCH/repopeft_data"
COMMIT_DATA_DIR="$DATA_ROOT/commit_parquet_hf"

echo "=========================================="
echo "RepoPeft commit-level dataset setup"
echo "  SCRATCH:      $SCRATCH"
echo "  VENV:         $VENV_DIR"
echo "  COMMIT_DATA:  $COMMIT_DATA_DIR"
echo "  HF_REPO:      $HF_REPO"
echo "=========================================="

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run: scripts/watgpu/setup_env.sh first"
    exit 1
fi

module purge
module load python/3.12 cuda/12.6 2>/dev/null || true
source "$VENV_DIR/bin/activate"

# pyarrow is needed by the loader; install if missing (idempotent).
python -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

mkdir -p "$COMMIT_DATA_DIR"

python - "$HF_REPO" "$COMMIT_DATA_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]

print(f"Downloading {repo_id} -> {local_dir}")
# commits/* and qna/* are the two directories the loader actually reads.
# splits/* are the cross-repo repo-id JSONs; README.md is the dataset card.
# shards/* is skipped (only needed if you want per-repo reproducibility).
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "commits/*",
        "qna/*",
        "splits/*",
        "README.md",
    ],
)
print("Done.")
PYEOF

echo -e "\n>>> Verifying..."
for sub in commits qna; do
    if [ -d "$COMMIT_DATA_DIR/$sub" ]; then
        n=$(ls "$COMMIT_DATA_DIR/$sub"/*.parquet 2>/dev/null | wc -l)
        echo "  OK:      $sub/ ($n parquet files)"
    else
        echo "  MISSING: $sub/"
    fi
done
for f in train.json cr_val.json cr_test.json; do
    if [ -f "$COMMIT_DATA_DIR/splits/$f" ]; then
        echo "  OK:      splits/$f"
    else
        echo "  note:    splits/$f not present (fine, parquet carries splits)"
    fi
done

echo -e "\n=========================================="
echo "Commit dataset ready at:"
echo "  $COMMIT_DATA_DIR"
echo "Submit training with:"
echo "  sbatch scripts/watgpu/train_code2lora_gru_commits.sh"
echo "=========================================="
