#!/bin/bash
# Orchestrate the full harrier-oss-v1-27b embedding -> dataset pipeline by
# submitting the four stages with Slurm dependencies:
#
#   1. diff embeddings      (GPU array)
#   2. repo-state embeddings(GPU array)
#   3. merge shards         (CPU, afterok 1 & 2)
#   4. build snapshots ds   (CPU, afterok 3)
#
# Everything writes to harrier-specific paths so the Qwen3-0.6B (2048-d)
# dataset is untouched:
#   shards : commit_parquet_hf_v2_harrier27b_shards/{diff,repo_state}
#   merged : commit_parquet_hf_v2_harrier27b
#   snaps  : code2lora_snapshots_harrier27b_hf
#
# This is a *submit* launcher: run it on the login node, NOT via sbatch.
#
# Usage:
#   ACCOUNT=rrg-yuntian bash scripts/slurm/run_harrier27b_pipeline.sh
#   ACCOUNT=rrg-yuntian SHARD_TOTAL=16 bash scripts/slurm/run_harrier27b_pipeline.sh

set -euo pipefail
cd /home/lhotsko/RepoPeftData
source scripts/slurm/common.sh

ACCOUNT="${ACCOUNT:-}"
ACCT_ARGS=()
[ -n "$ACCOUNT" ] && ACCT_ARGS+=(--account "$ACCOUNT")

SHARD_TOTAL="${SHARD_TOTAL:-16}"
ARRAY_MAX=$(( SHARD_TOTAL - 1 ))

SHARDS_BASE="$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_shards"
MERGED_DIR="$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b"
SNAP_DIR="$SCRATCH/REPO_DATASET/code2lora_snapshots_harrier27b_hf"

echo "===== harrier-27b pipeline ====="
echo "Account     : ${ACCOUNT:-<script default>}"
echo "Shard total : $SHARD_TOTAL (array 0-$ARRAY_MAX)"
echo "Shards base : $SHARDS_BASE"
echo "Merged      : $MERGED_DIR"
echo "Snapshots   : $SNAP_DIR"

# 1) diff embeddings
DIFF_JID=$(SHARD_TOTAL="$SHARD_TOTAL" \
    sbatch "${ACCT_ARGS[@]}" --parsable --array=0-"$ARRAY_MAX" \
        scripts/slurm/build_diff_embeddings_harrier27b.sh)
echo "[1] diff embeddings      : job $DIFF_JID"

# 2) repo-state embeddings
REPO_JID=$(SHARD_TOTAL="$SHARD_TOTAL" \
    sbatch "${ACCT_ARGS[@]}" --parsable --array=0-"$ARRAY_MAX" \
        scripts/slurm/build_repo_state_embeddings_harrier27b.sh)
echo "[2] repo-state embeddings: job $REPO_JID"

# 3) merge (after both arrays finish)
MERGE_JID=$(COMMITS_DIR="$SCRATCH/REPO_DATASET/commit_parquet_hf/commits" \
    SHARDS_BASE="$SHARDS_BASE" OUT_DIR_V2="$MERGED_DIR" \
    sbatch "${ACCT_ARGS[@]}" --parsable \
        --dependency="afterok:${DIFF_JID}:${REPO_JID}" \
        scripts/slurm/merge_gru_v2_embeddings.sh \
        --model-name microsoft/harrier-oss-v1-27b --pooling lasttoken)
echo "[3] merge shards         : job $MERGE_JID (afterok $DIFF_JID,$REPO_JID)"

# 4) build snapshots dataset (after merge)
SNAP_JID=$(V2_COMMITS_DIR="$MERGED_DIR/commits" SNAP_OUT_DIR="$SNAP_DIR" \
    sbatch "${ACCT_ARGS[@]}" --parsable \
        --dependency="afterok:${MERGE_JID}" \
        scripts/slurm/build_code2lora_snapshots.sh)
echo "[4] build snapshots ds   : job $SNAP_JID (afterok $MERGE_JID)"

echo "Submitted harrier-27b pipeline: $DIFF_JID -> $REPO_JID -> $MERGE_JID -> $SNAP_JID"
echo "Merged dataset will be at: $MERGED_DIR"
