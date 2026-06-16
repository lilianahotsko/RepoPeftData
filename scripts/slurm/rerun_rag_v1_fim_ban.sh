#!/bin/bash
#SBATCH --job-name=rerun_rag_v1
#SBATCH --output=slurm_logs/rerun_rag_v1_%A_%a.out
#SBATCH --error=slurm_logs/rerun_rag_v1_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Re-run V1 RAG (Table 1 dataset: $SCRATCH/REPO_DATASET/{cr,ir}_test.json,
# one-cut-per-repo QnAs, with chunk indices under $SCRATCH/RAG_CHUNK_CACHE)
# with the FIM-ban patch (bad_words_ids on Qwen FIM/file-sep/repo-name).
#
# Array task 0 -> cr_test
# Array task 1 -> ir_test
#
#   OUT_DIR=$SCRATCH/RAG_V1_FIM_BAN \
#     sbatch --array=0-1 scripts/slurm/rerun_rag_v1_fim_ban.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="${OUT_DIR:-$SCRATCH/RAG_V1_FIM_BAN}"
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE}"
TOP_K="${TOP_K:-3}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$OUT_DIR"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SPLIT=cr_test ;;
    1) SPLIT=ir_test ;;
    *) echo "[error] bad SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

OUT_JSON="$OUT_DIR/rag_top${TOP_K}_${SPLIT}_fim_ban.json"

echo "===== Re-run V1 RAG (FIM-ban)  split=${SPLIT}  k=${TOP_K} ====="
echo "Splits dir : $SPLITS_DIR"
echo "Cache dir  : $CACHE_DIR"
echo "Out json   : $OUT_JSON"
echo "Start      : $(date)"
nvidia-smi -L || true

python baselines/rag/test_rag.py \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$CACHE_DIR" \
    --split "$SPLIT" \
    --top-k "$TOP_K" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$OUT_JSON"

echo "Done: $(date)"
