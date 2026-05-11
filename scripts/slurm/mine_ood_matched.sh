#!/bin/bash
# Mine a matched-distribution OOD set: same star / size / license / pytest
# filters as the training corpus, only post-cutoff in created_at. This fixes
# the population mismatch that made the existing OOD eval incomparable to IR.
#
# Outputs JSONL to $SCRATCH/REPO_DATASET/ood_repos_matched.jsonl
#
# Required env: GITHUB_TOKEN (rate-limited search API; 5000 req/h)
# Optional env: TARGET (default 200; "all-we-can-find" semantics if you set it
#                       high, the miner just stops when GitHub returns no more
#                       candidates).
#               CREATED_AFTER (YYYY-MM-DD; if unset, infer from training set)

#SBATCH --job-name=mine_ood_matched
#SBATCH --output=slurm_logs/mine_ood_matched_%j.out
#SBATCH --error=slurm_logs/mine_ood_matched_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN before sbatch (export GITHUB_TOKEN=ghp_...)}"
export GITHUB_TOKEN

OUT_PATH="${OUT_PATH:-$SCRATCH/REPO_DATASET/ood_repos_matched.jsonl}"
INFER_CACHE="${INFER_CACHE:-$SCRATCH/REPO_DATASET/ood_matched.baseline_created_cache.json}"
TARGET="${TARGET:-200}"

# Training-matched filters (from repos_collection/mine_repos*.py).
# Star slices cover the full training distribution; we'll let GitHub's `created:>`
# trim the result set to the post-cutoff window.
STAR_SLICES="${STAR_SLICES:-50..100,101..200,201..299,300..500,501..1000,1001..2000}"
SIZE_RANGE="${SIZE_RANGE:-3000..15000}"          # KB; same as training
PUSHED_AFTER="${PUSHED_AFTER:-2025-01-01}"        # any post-2025 activity

# Baselines: training + cr_val + cr_test (so the OOD set excludes them and
# any forks-of-them). Use the GRU-format files since they include metadata.
BASELINE_FLAGS=(
  --baseline-json "$SCRATCH/REPO_DATASET/gru_train.json"
  --baseline-json "$SCRATCH/REPO_DATASET/cr_val.json"
  --baseline-json "$SCRATCH/REPO_DATASET/cr_test.json"
)

# Cutoff date: fresh inference unless caller passed CREATED_AFTER.
CREATED_FLAGS=()
if [ -n "${CREATED_AFTER:-}" ]; then
    CREATED_FLAGS+=(--created-after "$CREATED_AFTER")
    echo "Using explicit CREATED_AFTER=$CREATED_AFTER"
else
    # Force a fresh cache file (rename old one if present) so the cutoff is
    # re-inferred from the current baseline.
    if [ -f "$INFER_CACHE" ]; then
        mv "$INFER_CACHE" "${INFER_CACHE}.old.$(date +%s)"
    fi
    CREATED_FLAGS+=(--infer-cache "$INFER_CACHE")
    echo "Re-inferring cutoff from baseline (cache=$INFER_CACHE)"
fi

echo "Out path     : $OUT_PATH"
echo "Target       : $TARGET"
echo "Star slices  : $STAR_SLICES"
echo "Size range   : $SIZE_RANGE"
echo "Pushed after : $PUSHED_AFTER"

python repos_collection/mine_ood_repos.py mine \
    "${BASELINE_FLAGS[@]}" \
    --target "$TARGET" \
    --out "$OUT_PATH" \
    --pushed-after "$PUSHED_AFTER" \
    --size-range "$SIZE_RANGE" \
    --star-slice "$STAR_SLICES" \
    --max-pages 10 \
    --append-out \
    "${CREATED_FLAGS[@]}"

echo
echo "=== Summary ==="
wc -l "$OUT_PATH" 2>/dev/null || echo "(empty)"
python3 - "$OUT_PATH" <<'PY'
import json, sys
from collections import Counter
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
if not rows:
    print("(no rows)"); sys.exit(0)
stars = np.array([r["stars"] for r in rows])
size = np.array([r["size_kb"] for r in rows])
lic = Counter(r.get("license","") for r in rows)
created_mo = Counter(r["created_at"][:7] for r in rows if r.get("created_at"))
print(f"n_repos: {len(rows)}")
print(f"stars  : p10={int(np.percentile(stars,10))} p50={int(np.percentile(stars,50))} p90={int(np.percentile(stars,90))} max={int(stars.max())}")
print(f"size_kb: p10={int(np.percentile(size,10))} p50={int(np.percentile(size,50))} p90={int(np.percentile(size,90))} max={int(size.max())}")
print(f"licenses: {dict(lic)}")
print(f"created months (top 6): {dict(created_mo.most_common(6))}")
PY

echo "Done: $(date)"
