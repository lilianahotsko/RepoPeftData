#!/bin/bash
# Embed every (repo, commit) snapshot from the static-commit manifest.
# Canonical Qwen3-Embedding-0.6B recipe; per-blob caching gives ~10x speedup.
#
# Empirically ~10 blob embeds/s on H100 -> ~6h for 207k blobs total.
# Use START/STOP env vars to shard across multiple GPUs (cuts wall to ~1.5h on 4 GPUs).
#
# Inputs:
#   $SCRATCH/REPO_DATASET/static_commit/manifest.tsv
#   $SCRATCH/REPO_DATASET/{repositories,repositories_ood}/<author>/<repo>/.git
# Outputs:
#   $SCRATCH/REPO_DATASET/static_commit/cache/<author>__<repo>/{blob_*,snapshot_*}
#   $SCRATCH/REPO_DATASET/static_commit/snapshot_embeddings.json
#       (built incrementally; final merge happens at the end of every shard run)

#SBATCH --job-name=build_static_commit_embed
#SBATCH --output=slurm_logs/build_static_commit_embed_%j.out
#SBATCH --error=slurm_logs/build_static_commit_embed_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

MANIFEST="${MANIFEST:-$SCRATCH/REPO_DATASET/static_commit/manifest.tsv}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/static_commit}"

EXTRA_ARGS=()
if [ -n "${START:-}" ]; then EXTRA_ARGS+=(--start "$START"); fi
if [ -n "${STOP:-}" ]; then EXTRA_ARGS+=(--stop "$STOP"); fi
if [ -n "${LIMIT_REPOS:-}" ]; then EXTRA_ARGS+=(--limit-repos "$LIMIT_REPOS"); fi

echo "Manifest : $MANIFEST"
echo "Out dir  : $OUT_DIR"
echo "Shard    : start=${START:-0} stop=${STOP:-0} limit=${LIMIT_REPOS:-0}"

python create_dataset/build_static_commit_embeddings.py \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "Done: $(date)"
