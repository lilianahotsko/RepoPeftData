#!/bin/bash
# Clone matched-distribution OOD repos and build commit-parquet using the
# SAME pipeline as the training corpus (build_commit_parquet_db.py).
#
# Inputs:  $SCRATCH/REPO_DATASET/ood_repos_matched.jsonl  (from mine_ood_matched.sh)
# Outputs: $SCRATCH/REPO_DATASET/repositories_ood_matched/<owner>/<repo>/
#          $SCRATCH/REPO_DATASET/commit_parquet_ood_matched/{commits,qna_pairs}.parquet
#          $SCRATCH/REPO_DATASET/ood_matched_bundle/  (intermediate splits + sharded parquets)

#SBATCH --job-name=build_ood_matched_parquet
#SBATCH --output=slurm_logs/build_ood_matched_parquet_%j.out
#SBATCH --error=slurm_logs/build_ood_matched_parquet_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

MINED_JSONL="${MINED_JSONL:-$SCRATCH/REPO_DATASET/ood_repos_matched.jsonl}"
REPOS_ROOT="${REPOS_ROOT:-$SCRATCH/REPO_DATASET/repositories_ood_matched}"
WORK_DIR="${WORK_DIR:-$SCRATCH/REPO_DATASET/ood_matched_bundle}"
OUT_DIR="${OUT_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood_matched}"
CLONE_JOBS="${CLONE_JOBS:-8}"
PARQUET_WORKERS="${PARQUET_WORKERS:-16}"

echo "Mined JSONL    : $MINED_JSONL"
echo "Repos root     : $REPOS_ROOT"
echo "Work dir       : $WORK_DIR"
echo "Out dir        : $OUT_DIR"
echo "Clone jobs     : $CLONE_JOBS"
echo "Parquet workers: $PARQUET_WORKERS"

python create_dataset/build_ood_parquet_from_mined_jsonl.py \
    --mined-jsonl   "$MINED_JSONL" \
    --repos-root    "$REPOS_ROOT" \
    --work-dir      "$WORK_DIR" \
    --out-dir       "$OUT_DIR" \
    --clone-jobs    "$CLONE_JOBS" \
    --parquet-workers "$PARQUET_WORKERS"

echo "Done: $(date)"
ls -lah "$OUT_DIR"/ 2>&1 | head -10
