#!/bin/bash
#SBATCH --job-name=rerun_fft_rag
#SBATCH --output=slurm_logs/rerun_fft_rag_%A_%a.out
#SBATCH --error=slurm_logs/rerun_fft_rag_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

# Re-run the Table 1 "FFT + RAG" row with the same context calibration applied
# to the standalone RAG row (FIM-token ban at decode time + un-truncated prefix,
# and optionally Qwen2.5-Coder's native repo-completion layout).
#
# The published row (53.9 CR / 56.8 IR) came from
# baselines/finetuned/test_finetuned.py, which has neither the ban nor the
# repo_fim layout. baselines/rag/test_rag.py has both and loads its generation
# model from --model-name, so pointing that at the FFT checkpoint reuses the
# exact calibrated code path as the RAG row -- no new inference code.
#
# Retrieval config is held at the published FFT+RAG setting
# (RAG_CHUNK_CACHE_256, k=5) so that calibration is the only thing that changes.
#
# Array task 0 -> cr_test, plain     2 -> ir_test, plain
#            1 -> cr_test, repo_fim  3 -> ir_test, repo_fim
#
#   sbatch --array=0-3 scripts/slurm/rerun_fft_rag_fim_ban.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="${OUT_DIR:-$SCRATCH/FFT_RAG_FIM_BAN}"
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/REPO_DATASET}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE_256}"
MODEL_PATH="${MODEL_PATH:-$CKPT_DIR/FFT/final}"
TOP_K="${TOP_K:-5}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$OUT_DIR"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SPLIT=cr_test; PROMPT_FORMAT=plain ;;
    1) SPLIT=cr_test; PROMPT_FORMAT=repo_fim ;;
    2) SPLIT=ir_test; PROMPT_FORMAT=plain ;;
    3) SPLIT=ir_test; PROMPT_FORMAT=repo_fim ;;
    *) echo "[error] bad SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

FMT_TAG=""
if [ "$PROMPT_FORMAT" = "repo_fim" ]; then FMT_TAG="_repofim"; fi
OUT_JSON="$OUT_DIR/fft_rag256_k${TOP_K}${FMT_TAG}_${SPLIT}_fim_ban.json"

echo "===== Re-run FFT+RAG (FIM-ban)  split=${SPLIT}  fmt=${PROMPT_FORMAT}  k=${TOP_K} ====="
echo "Model      : $MODEL_PATH"
echo "Splits dir : $SPLITS_DIR"
echo "Cache dir  : $CACHE_DIR"
echo "Out json   : $OUT_JSON"
echo "Start      : $(date)"
nvidia-smi -L || true

python baselines/rag/test_rag.py \
    --model-name "$MODEL_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --cache-dir "$CACHE_DIR" \
    --split "$SPLIT" \
    --top-k "$TOP_K" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-format "$PROMPT_FORMAT" \
    --output "$OUT_JSON"

echo "Done: $(date)"
