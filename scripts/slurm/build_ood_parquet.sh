#!/bin/bash
# Build the OOD commit-parquet from the mined OOD repos JSONL.
# This is the strong-review-ready OOD test suite for Table 1.
#
# Output: $OOD_OUT_DIR/commits.parquet + qna_pairs.parquet, plus a sidecar
# $SCRATCH/REPO_DATASET/splits/ood_test.json wired into the cross_repo_split
# vocabulary so the parquet loader can yield ``ood_test`` rows.

#SBATCH --job-name=build_ood_parquet
#SBATCH --output=slurm_logs/build_ood_parquet_%j.out
#SBATCH --error=slurm_logs/build_ood_parquet_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

OOD_JSONL="${OOD_JSONL:-$SCRATCH/REPO_DATASET/ood_repos_100.jsonl}"
OOD_REPOS_ROOT="${OOD_REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories_ood}"
OOD_WORK_DIR="${OOD_WORK_DIR:-$SCRATCH/REPO_DATASET/ood_bundle}"
OOD_OUT_DIR="${OOD_OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood}"

mkdir -p "$OOD_REPOS_ROOT" "$OOD_WORK_DIR" "$OOD_OUT_DIR"

if [ ! -f "$OOD_JSONL" ]; then
  echo "ERROR: OOD JSONL not found at $OOD_JSONL"
  echo "Run repos_collection/mine_ood_repos.py first (see its docstring)."
  exit 1
fi

echo "===== Build OOD commit-parquet ====="
echo "JSONL:        $OOD_JSONL ($(wc -l < "$OOD_JSONL") repos)"
echo "Repos root:   $OOD_REPOS_ROOT"
echo "Work dir:     $OOD_WORK_DIR"
echo "Out dir:      $OOD_OUT_DIR"
echo "Start:        $(date)"

python create_dataset/build_ood_parquet_from_mined_jsonl.py \
  --mined-jsonl  "$OOD_JSONL" \
  --repos-root   "$OOD_REPOS_ROOT" \
  --work-dir     "$OOD_WORK_DIR" \
  --out-dir      "$OOD_OUT_DIR" \
  --clone-jobs 8 --parquet-workers 8

echo "Done: $(date)"
