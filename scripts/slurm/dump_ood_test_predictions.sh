#!/bin/bash
#SBATCH --job-name=dump_preds_ood_test
#SBATCH --output=slurm_logs/dump_preds_ood_test_%A_%a.out
#SBATCH --error=slurm_logs/dump_preds_ood_test_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Dump per-QnA predictions (and the augmented RAG / DRC prompt for the
# context-injection baselines) for a sampled subset of the ood_test suite.
# The subset is driven by a {repo_id, commit_sha} JSONL produced by
# `visualize_gru_ood_test_qnas.py --write-selected-keys`.
#
# Array layout (one job per method):
#   0 -> code2lora        (static)
#   1 -> code2lora_gru
#   2 -> rag
#   3 -> drc
#   4 -> text2lora
#
# Submit:
#   # 1. produce the keys file (no GPU):
#   python visualize_gru_ood_test_qnas.py \
#       --write-selected-keys $SCRATCH/OOD_TEST_PRED_DUMP/selected_keys.jsonl
#
#   # 2. submit all 5 dump jobs:
#   KEYS_FILE=$SCRATCH/OOD_TEST_PRED_DUMP/selected_keys.jsonl \
#       OUT_DIR=$SCRATCH/OOD_TEST_PRED_DUMP \
#       sbatch --array=0-4 scripts/slurm/dump_ood_test_predictions.sh
#
#   # 3. rebuild the report with prediction text & contexts attached:
#   python visualize_gru_ood_test_qnas.py \
#       --predictions-dir $SCRATCH/OOD_TEST_PRED_DUMP \
#       --output report_gru_ood_test_qnas.html

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

KEYS_FILE="${KEYS_FILE:?KEYS_FILE must point to the selected_keys.jsonl produced by visualize_gru_ood_test_qnas.py --write-selected-keys}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set (one .jsonl per method goes here)}"
# OOD uses the unified parquet (qna_pairs.parquet) plus the OOD commit shards.
QNA_PARQUET="${QNA_PARQUET:-$SCRATCH/REPO_DATASET/commit_parquet_ood/qna_pairs.parquet}"
QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood_qna_dir}"
COMMITS_DIR="${COMMITS_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_ood}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf_ood}"
QNAS_DIR_ROOT="${QNAS_DIR_ROOT:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf_ood}"

C2L_STATIC_CKPT="${C2L_STATIC_CKPT:-$CKPT_DIR/CODE2LORA_STATIC/h100_v2_ood_run5/head.ep2.pt}"
C2L_GRU_CKPT="${C2L_GRU_CKPT:-$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep_ood/gru_head.best.pt}"
RAG_CACHE_DIR="${RAG_CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE_AST_OOD}"
DRC_CACHE_DIR="${DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_OOD}"
TEXT2LORA_HYPERMOD_DIR="${TEXT2LORA_HYPERMOD_DIR:-text2lora/train_outputs/sft/hyper_lora/code_sft_v2_full7_14789268}"
TEXT2LORA_CODE_EMB_PATH="${TEXT2LORA_CODE_EMB_PATH:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2_ood.pt}"
TEXT2LORA_DIR="${TEXT2LORA_DIR:-$(pwd)/text2lora}"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-8}"

# The v2 baseline driver expects --qna-dir to point at a folder containing
# `<suite>.parquet`. For OOD we point at the unified qna_pairs.parquet through
# a temporary symlinked dir created by this script so we can keep using the
# upstream loader unmodified.
if [[ ! -d "$QNA_DIR" ]]; then
    mkdir -p "$QNA_DIR"
fi
if [[ ! -e "$QNA_DIR/ood_test.parquet" ]]; then
    ln -sf "$QNA_PARQUET" "$QNA_DIR/ood_test.parquet"
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
    0) METHOD="code2lora" ;;
    1) METHOD="code2lora_gru" ;;
    2) METHOD="rag" ;;
    3) METHOD="drc" ;;
    4) METHOD="text2lora" ;;
    *) echo "[error] unknown SLURM_ARRAY_TASK_ID=$IDX (expected 0..4)" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
PRED_OUT="$OUT_DIR/${METHOD}.jsonl"
SHARD_DIR="$OUT_DIR/_eval_${METHOD}"
mkdir -p "$SHARD_DIR"

# Start from a clean predictions file so the JSONL is consistent with the
# shard JSON we are about to (re)write.
rm -f "$PRED_OUT"
rm -f "$SHARD_DIR"/*.json 2>/dev/null || true

echo "===== Dump OOD predictions [task ${IDX}] ====="
echo "Method        : $METHOD"
echo "Keys          : $KEYS_FILE  ($(wc -l < "$KEYS_FILE") commits)"
echo "QnA parquet   : $QNA_PARQUET"
echo "Predictions   : $PRED_OUT"
echo "Shard JSON dir: $SHARD_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

case "$METHOD" in
    code2lora)
        python evaluation/run_code2lora_static_v2_eval.py \
            --checkpoint "$C2L_STATIC_CKPT" \
            --snapshots-dir "$SNAPSHOTS_DIR" \
            --suite ood_test \
            --output-dir "$SHARD_DIR" \
            --max-input-tokens "$MAX_INPUT_TOKENS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
            --bootstrap 0 \
            --predictions-out "$PRED_OUT" \
            --restrict-keys "$KEYS_FILE" \
            --method-label "code2lora"
        ;;
    code2lora_gru)
        python evaluation/run_code2lora_gru_v2_eval.py \
            --checkpoint "$C2L_GRU_CKPT" \
            --commits-dir "$COMMITS_DIR" \
            --qnas-dir "$QNAS_DIR_ROOT" \
            --suite ood_test \
            --output-dir "$SHARD_DIR" \
            --max-input-tokens "$MAX_INPUT_TOKENS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
            --bootstrap 0 \
            --predictions-out "$PRED_OUT" \
            --restrict-keys "$KEYS_FILE" \
            --method-label "code2lora_gru"
        ;;
    rag)
        python evaluation/run_baselines_v2.py \
            --method rag \
            --qna-dir "$QNA_DIR" \
            --suites ood_test \
            --output-dir "$SHARD_DIR" \
            --rag-cache-dir "$RAG_CACHE_DIR" \
            --rag-top-k "${RAG_TOP_K:-3}" \
            --rag-max-context-tokens "${RAG_MAX_CONTEXT_TOKENS:-1536}" \
            --max-input-tokens "$MAX_INPUT_TOKENS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
            --bootstrap 0 \
            --predictions-out "$PRED_OUT" \
            --restrict-keys "$KEYS_FILE"
        ;;
    drc)
        python evaluation/run_baselines_v2.py \
            --method drc \
            --qna-dir "$QNA_DIR" \
            --suites ood_test \
            --output-dir "$SHARD_DIR" \
            --drc-cache-dir "$DRC_CACHE_DIR" \
            --drc-max-tokens "${DRC_MAX_TOKENS:-4096}" \
            --max-input-tokens "$MAX_INPUT_TOKENS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
            --bootstrap 0 \
            --predictions-out "$PRED_OUT" \
            --restrict-keys "$KEYS_FILE"
        ;;
    text2lora)
        export PYTHONPATH="$(pwd)/text2lora/src:${PYTHONPATH:-}"
        python evaluation/run_baselines_v2.py \
            --method text2lora \
            --qna-dir "$QNA_DIR" \
            --suites ood_test \
            --output-dir "$SHARD_DIR" \
            --text2lora-hypermod-dir "$TEXT2LORA_HYPERMOD_DIR" \
            --text2lora-code-emb-path "$TEXT2LORA_CODE_EMB_PATH" \
            --text2lora-dir "$TEXT2LORA_DIR" \
            --max-input-tokens "$MAX_INPUT_TOKENS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT" \
            --bootstrap 0 \
            --predictions-out "$PRED_OUT" \
            --restrict-keys "$KEYS_FILE"
        ;;
esac

echo "Done: $(date)"
echo "Wrote $(wc -l < "$PRED_OUT") prediction rows to $PRED_OUT"
