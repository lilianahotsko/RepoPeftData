#!/bin/bash
#SBATCH --job-name=rerun_fim_ban
#SBATCH --output=slurm_logs/rerun_fim_ban_%A_%a.out
#SBATCH --error=slurm_logs/rerun_fim_ban_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Re-run RAG / DRC / Pretrained / Text2LoRA on cr_test + ir_test under
# two prefix-budget regimes (matches Table 1 = static 256-token cap and
# Table 2 = native commit-derived prefix). The patch to _generate_batch
# bans the Qwen2.5-Coder FIM / file-sep / repo-name special tokens at
# decode time. Per-QnA predictions are dumped to JSONL alongside the
# usual score JSON.
#
# Array layout (16 tasks):
#   task = (method_i * 4) + (budget_i * 2) + suite_i
#   method_i: 0=rag, 1=drc, 2=pretrained, 3=text2lora
#   budget_i: 0=prefix256 (static),  1=full (commit-derived)
#   suite_i : 0=cr_test, 1=ir_test
#
# Submit:
#   OUT_ROOT=$SCRATCH/BASELINES_V2_FIM_BAN \
#     sbatch --array=0-15 scripts/slurm/rerun_baselines_fim_ban.sh

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_ROOT="${OUT_ROOT:-$SCRATCH/BASELINES_V2_FIM_BAN}"
QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
RAG_CACHE_DIR="${RAG_CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS}"
DRC_CACHE_DIR="${DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS}"
TEXT2LORA_HYPERMOD_DIR="${TEXT2LORA_HYPERMOD_DIR:-text2lora/train_outputs/sft/hyper_lora/code_sft_v2_full7_14789268}"
TEXT2LORA_CODE_EMB_PATH="${TEXT2LORA_CODE_EMB_PATH:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt}"
TEXT2LORA_DIR="${TEXT2LORA_DIR:-$(pwd)/text2lora}"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-8}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
METHOD_I=$(( IDX / 4 ))
REM=$(( IDX % 4 ))
BUDGET_I=$(( REM / 2 ))
SUITE_I=$(( REM % 2 ))

case "$METHOD_I" in
    0) METHOD=rag ;;
    1) METHOD=drc ;;
    2) METHOD=pretrained ;;
    3) METHOD=text2lora ;;
    *) echo "[error] bad method_i=$METHOD_I" >&2; exit 1 ;;
esac
case "$BUDGET_I" in
    0) BUDGET_TAG=prefix256;   PREFIX_MAX_TOKENS=256 ;;
    1) BUDGET_TAG=full;        PREFIX_MAX_TOKENS=0   ;;
    *) echo "[error] bad budget_i=$BUDGET_I" >&2; exit 1 ;;
esac
case "$SUITE_I" in
    0) SUITE=cr_test ;;
    1) SUITE=ir_test ;;
    *) echo "[error] bad suite_i=$SUITE_I" >&2; exit 1 ;;
esac

OUT_DIR="$OUT_ROOT/${METHOD}__${BUDGET_TAG}"
mkdir -p "$OUT_DIR"
PRED_OUT="$OUT_DIR/${METHOD}_${BUDGET_TAG}_${SUITE}.jsonl"
rm -f "$PRED_OUT"

EXTRA=()
if [ "$PREFIX_MAX_TOKENS" != "0" ]; then
    EXTRA+=(--prefix-max-tokens "$PREFIX_MAX_TOKENS")
fi
if [ "$METHOD" = "rag" ]; then
    EXTRA+=(--rag-cache-dir "$RAG_CACHE_DIR" --rag-top-k 3
            --rag-max-context-tokens 1536)
fi
if [ "$METHOD" = "drc" ]; then
    EXTRA+=(--drc-cache-dir "$DRC_CACHE_DIR" --drc-max-tokens 4096)
fi
if [ "$METHOD" = "text2lora" ]; then
    export PYTHONPATH="$(pwd)/text2lora/src:${PYTHONPATH:-}"
    EXTRA+=(--text2lora-hypermod-dir "$TEXT2LORA_HYPERMOD_DIR"
            --text2lora-code-emb-path "$TEXT2LORA_CODE_EMB_PATH"
            --text2lora-dir "$TEXT2LORA_DIR")
fi

echo "===== rerun_fim_ban  task ${IDX}  method=${METHOD}  budget=${BUDGET_TAG}  suite=${SUITE} ====="
echo "Output dir : $OUT_DIR"
echo "Pred jsonl : $PRED_OUT"
echo "Start      : $(date)"
nvidia-smi -L || true

python evaluation/run_baselines_v2.py \
    --method "$METHOD" \
    --qna-dir "$QNA_DIR" \
    --suites "$SUITE" \
    --output-dir "$OUT_DIR" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
    --bootstrap "$BOOTSTRAP" \
    --predictions-out "$PRED_OUT" \
    "${EXTRA[@]}"

echo "Done: $(date)"
echo "Wrote $(wc -l < "$PRED_OUT" 2>/dev/null || echo 0) prediction rows to $PRED_OUT"
