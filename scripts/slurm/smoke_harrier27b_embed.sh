#!/bin/bash
# Smoke test for the harrier-oss-v1-27b embedding pipeline: embed a tiny subset
# (a handful of diffs + the repos in one of many shards) and assert the output
# vectors are 5376-d (harrier hidden size, last-token + L2 recipe). Validates
# the full code path on GPU before committing to the multi-hour full run.
#
# Submit:
#   ACCOUNT=rrg-yuntian sbatch scripts/slurm/smoke_harrier27b_embed.sh
#
# Outputs go to *_smoke paths and can be deleted afterward.

#SBATCH --job-name=smoke_harrier27b
#SBATCH --output=slurm_logs/smoke_harrier27b_%j.out
#SBATCH --error=slurm_logs/smoke_harrier27b_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-microsoft/harrier-oss-v1-27b}"
POOLING="${POOLING:-lasttoken}"
DTYPE="${DTYPE:-bfloat16}"
# Huge shard-total so shard 0 contains only ~1-2 repos (fast).
SMOKE_SHARD_TOTAL="${SMOKE_SHARD_TOTAL:-400}"
EXPECT_DIM="${EXPECT_DIM:-5376}"

INPUT_DIR="$SCRATCH/REPO_DATASET/commit_parquet_hf/commits"
DIFF_OUT="$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_smoke/diff"
REPO_OUT="$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_harrier27b_smoke/repo_state"
CACHE_ROOT="$SCRATCH/REPO_DATASET/static_commit/cache_harrier27b_smoke"

echo "===== SMOKE: harrier-27b embeddings ====="
echo "Model   : $MODEL_NAME (pooling=$POOLING dtype=$DTYPE)"
echo "Expect dim = $EXPECT_DIM"
echo "Start   : $(date)"
nvidia-smi -L || true

echo "--- diff (limit 32) ---"
python create_dataset/build_diff_embeddings_shard.py \
    --input-dir "$INPUT_DIR" --out-dir "$DIFF_OUT" \
    --splits cr_val --shard-index 0 --shard-total "$SMOKE_SHARD_TOTAL" \
    --model-name "$MODEL_NAME" --pooling "$POOLING" --dtype "$DTYPE" \
    --batch-size 8 --diff-batch 32 --device cuda --limit 32

echo "--- repo-state (shard 0 of $SMOKE_SHARD_TOTAL) ---"
python create_dataset/build_repo_state_embeddings_shard.py \
    --input-dir "$INPUT_DIR" --out-dir "$REPO_OUT" --cache-root "$CACHE_ROOT" \
    --splits cr_val --shard-index 0 --shard-total "$SMOKE_SHARD_TOTAL" \
    --model-name "$MODEL_NAME" --pooling "$POOLING" --dtype "$DTYPE" \
    --blob-batch 2 --device cuda

echo "--- validate output dims ---"
python - "$DIFF_OUT" "$REPO_OUT" "$EXPECT_DIM" <<'PY'
import sys, glob
import pyarrow.parquet as pq
diff_out, repo_out, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = True
for label, base, col in [("diff", diff_out, "diff_embedding"),
                         ("repo_state", repo_out, "repo_state_embedding")]:
    files = glob.glob(f"{base}/**/shard_*.parquet", recursive=True)
    files = [f for f in files if pq.read_metadata(f).num_rows > 0]
    if not files:
        print(f"[{label}] NO non-empty shard parquet found under {base}")
        ok = False
        continue
    t = pq.read_table(files[0])
    dim = len(t.column(col)[0].as_py())
    status = "OK" if dim == expect else "MISMATCH"
    print(f"[{label}] {files[0]}: rows={t.num_rows} dim={dim} (expect {expect}) -> {status}")
    ok = ok and (dim == expect)
print("SMOKE RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo "Done: $(date)"
