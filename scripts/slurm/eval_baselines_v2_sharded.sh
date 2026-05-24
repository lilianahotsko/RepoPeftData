#!/bin/bash
#SBATCH --job-name=eval_baseline_v2_sh
#SBATCH --output=slurm_logs/eval_baseline_v2_sh_%A_%a.out
#SBATCH --error=slurm_logs/eval_baseline_v2_sh_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Sharded v2 baseline (pretrained / fft / slora / rag / drc / text2lora / doc2lora) evaluation.
#
# Uses a SLURM array. Each array task evaluates ONE suite x ONE shard of the
# repos in that suite. The Python driver writes a JSON file after EVERY
# (repo, commit) group, so a wall-time kill never loses more than the one
# commit currently in flight. Re-launching the same array picks up where
# each shard left off (resume is automatic).
#
# Usage examples
# --------------
#   # Pretrained on all 4 suites with 4 repo shards each -> 16 array tasks:
#   METHOD=pretrained NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # FFT with the LATEST checkpoint, 4 suites x 4 shards:
#   METHOD=fft  CKPT=$CKPT_DIR/FFT_V2/h100_v2/checkpoint-44000 \
#     NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # SLoRA on the latest adapter:
#   METHOD=slora CKPT=$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000 \
#     NUM_SHARDS=4 SUITES="ir_val ir_test cr_val cr_test" \
#     sbatch --array=0-15 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # RAG (top-k=3) against the per-commit chunk cache:
#   METHOD=rag RAG_CACHE_DIR=$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS RAG_TOP_K=3 \
#     NUM_SHARDS=4 SUITES="ir_test cr_val" \
#     sbatch --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # DRC (adaptive 4K-token budget) against the per-commit DRC cache:
#   METHOD=drc DRC_CACHE_DIR=$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS DRC_MAX_TOKENS=4096 \
#     NUM_SHARDS=4 SUITES="ir_test cr_val" \
#     sbatch --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # Text2LoRA (Code-SFT v2) using a trained hypermod run:
#   METHOD=text2lora \
#     TEXT2LORA_HYPERMOD_DIR=text2lora/train_outputs/sft/hyper_lora/code_sft_v2_<run> \
#     TEXT2LORA_CODE_EMB_PATH=$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt \
#     NUM_SHARDS=4 SUITES="ir_test cr_val" \
#     sbatch --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
#
#   # Doc-to-LoRA (v2) using a trained D2L checkpoint:
#   METHOD=doc2lora \
#     DOC2LORA_CKPT=$CKPT_DIR/DOC2LORA_V2/<jobid>/checkpoint-20000/pytorch_model.bin \
#     DOC2LORA_DRC_CACHE_DIR=$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS \
#     NUM_SHARDS=4 SUITES="ir_test cr_val" \
#     sbatch --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
#
# Array-index decoding: idx = suite_index * NUM_SHARDS + shard_i.
# After all shards finish, merge with:
#   python evaluation/merge_eval_shards.py --auto-detect --input-dir $OUT_DIR

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
# Force offline mode: the model files are already in HF_HOME from prior runs,
# so all from_pretrained() calls should read from disk and never hit
# huggingface.co/api/* (which triggers 429s under sharded parallelism).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD="${METHOD:-pretrained}"
CKPT="${CKPT:-}"
QNA_DIR="${QNA_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna}"
SUITES_STR="${SUITES:-ir_val ir_test cr_val cr_test}"
read -r -a SUITES <<< "$SUITES_STR"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUFFIX="${SUFFIX:-h100_v2_sharded}"
OUT_DIR="$CKPT_DIR/BASELINES_V2/${METHOD}_${SUFFIX}"
mkdir -p "$OUT_DIR"

MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QNAS_PER_COMMIT_LIMIT="${QNAS_PER_COMMIT_LIMIT:-8}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
# Prefix-budget ablation: when >0, left-truncate each QnA's prefix to its last
# N tokens BEFORE RAG query / DRC compression / prompt assembly. Use this to
# reproduce the static-snapshot prefix budget (median ~224 tok) on the
# commit-derived suites so the static-vs-streaming "DRC helps / DRC hurts"
# inversion can be attributed to prefix length rather than DRC content.
PREFIX_MAX_TOKENS="${PREFIX_MAX_TOKENS:-0}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_SUITES=${#SUITES[@]}
N_TASKS=$(( N_SUITES * NUM_SHARDS ))
SUITE_I=$(( IDX / NUM_SHARDS ))
SHARD_I=$(( IDX % NUM_SHARDS ))
if [ "$SUITE_I" -ge "$N_SUITES" ]; then
    echo "[skip] array index $IDX out of range (n_tasks=$N_TASKS)"
    exit 0
fi
SUITE="${SUITES[$SUITE_I]}"

EXTRA_ARGS=()
if [ -n "$CKPT" ]; then
    EXTRA_ARGS+=(--ckpt "$CKPT")
fi
if [ "$QNAS_PER_COMMIT_LIMIT" != "0" ]; then
    EXTRA_ARGS+=(--qnas-per-commit-limit "$QNAS_PER_COMMIT_LIMIT")
fi
if [ "$PREFIX_MAX_TOKENS" != "0" ]; then
    EXTRA_ARGS+=(--prefix-max-tokens "$PREFIX_MAX_TOKENS")
fi

# Context-injection methods (rag / drc).
if [ "$METHOD" = "rag" ]; then
    RAG_CACHE_DIR="${RAG_CACHE_DIR:-$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS}"
    RAG_TOP_K="${RAG_TOP_K:-3}"
    RAG_EMBED_MODEL="${RAG_EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
    RAG_QUERY_CHARS="${RAG_QUERY_CHARS:-2000}"
    RAG_MAX_CONTEXT_TOKENS="${RAG_MAX_CONTEXT_TOKENS:-1536}"
    RAG_HYBRID="${RAG_HYBRID:-1}"
    if [ ! -d "$RAG_CACHE_DIR" ]; then
        echo "[error] RAG_CACHE_DIR not found: $RAG_CACHE_DIR" >&2
        echo "        Build it with scripts/slurm/build_rag_cache_per_commit.sh" >&2
        exit 1
    fi
    EXTRA_ARGS+=(
        --rag-cache-dir "$RAG_CACHE_DIR"
        --rag-top-k "$RAG_TOP_K"
        --rag-embed-model-name "$RAG_EMBED_MODEL"
        --rag-query-chars "$RAG_QUERY_CHARS"
        --rag-max-context-tokens "$RAG_MAX_CONTEXT_TOKENS"
    )
    if [ "$RAG_HYBRID" = "1" ]; then
        EXTRA_ARGS+=(--rag-hybrid)
    else
        EXTRA_ARGS+=(--no-rag-hybrid)
    fi
fi
if [ "$METHOD" = "drc" ]; then
    DRC_CACHE_DIR="${DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS}"
    DRC_MAX_TOKENS="${DRC_MAX_TOKENS:-4096}"
    if [ ! -d "$DRC_CACHE_DIR" ]; then
        echo "[error] DRC_CACHE_DIR not found: $DRC_CACHE_DIR" >&2
        echo "        Build it with scripts/slurm/build_drc_cache_per_commit.sh" >&2
        exit 1
    fi
    EXTRA_ARGS+=(
        --drc-cache-dir "$DRC_CACHE_DIR"
        --drc-max-tokens "$DRC_MAX_TOKENS"
    )
fi
if [ "$METHOD" = "text2lora" ]; then
    TEXT2LORA_HYPERMOD_DIR="${TEXT2LORA_HYPERMOD_DIR:-}"
    TEXT2LORA_CODE_EMB_PATH="${TEXT2LORA_CODE_EMB_PATH:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt}"
    TEXT2LORA_DIR="${TEXT2LORA_DIR:-$(pwd)/text2lora}"
    if [ -z "$TEXT2LORA_HYPERMOD_DIR" ]; then
        echo "[error] TEXT2LORA_HYPERMOD_DIR is required for METHOD=text2lora" >&2
        echo "        Train with scripts/slurm/train_text2lora_code_sft_v2.sh" >&2
        exit 1
    fi
    if [ ! -d "$TEXT2LORA_HYPERMOD_DIR" ]; then
        echo "[error] TEXT2LORA_HYPERMOD_DIR not found: $TEXT2LORA_HYPERMOD_DIR" >&2
        exit 1
    fi
    if [ ! -f "$TEXT2LORA_HYPERMOD_DIR/hypermod.pt" ]; then
        echo "[error] hypermod.pt missing under: $TEXT2LORA_HYPERMOD_DIR" >&2
        exit 1
    fi
    if [ ! -f "$TEXT2LORA_CODE_EMB_PATH" ]; then
        echo "[error] code_embeddings_v2.pt missing: $TEXT2LORA_CODE_EMB_PATH" >&2
        echo "        Build with scripts/slurm/extract_code_embeddings_v2.sh" >&2
        exit 1
    fi
    export PYTHONPATH="$(pwd)/text2lora/src:${PYTHONPATH:-}"
    EXTRA_ARGS+=(
        --text2lora-hypermod-dir "$TEXT2LORA_HYPERMOD_DIR"
        --text2lora-code-emb-path "$TEXT2LORA_CODE_EMB_PATH"
        --text2lora-dir "$TEXT2LORA_DIR"
    )
fi
if [ "$METHOD" = "doc2lora" ]; then
    DOC2LORA_CKPT="${DOC2LORA_CKPT:-}"
    DOC2LORA_DRC_CACHE_DIR="${DOC2LORA_DRC_CACHE_DIR:-$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS}"
    DOC2LORA_MAX_CTX_TOKENS="${DOC2LORA_MAX_CTX_TOKENS:-4096}"
    if [ -z "$DOC2LORA_CKPT" ]; then
        echo "[error] DOC2LORA_CKPT is required for METHOD=doc2lora" >&2
        echo "        Train with scripts/slurm/train_doc2lora_v2.sh" >&2
        exit 1
    fi
    if [ ! -e "$DOC2LORA_CKPT" ]; then
        echo "[error] DOC2LORA_CKPT not found: $DOC2LORA_CKPT" >&2
        exit 1
    fi
    if [ ! -d "$DOC2LORA_DRC_CACHE_DIR" ]; then
        echo "[error] DOC2LORA_DRC_CACHE_DIR not found: $DOC2LORA_DRC_CACHE_DIR" >&2
        echo "        Build with scripts/slurm/build_drc_cache_per_commit.sh" >&2
        exit 1
    fi
    export PYTHONPATH="$(pwd)/doc2lora/src:${PYTHONPATH:-}"
    EXTRA_ARGS+=(
        --doc2lora-ckpt "$DOC2LORA_CKPT"
        --doc2lora-drc-cache-dir "$DOC2LORA_DRC_CACHE_DIR"
        --doc2lora-max-ctx-tokens "$DOC2LORA_MAX_CTX_TOKENS"
    )
fi

echo "===== Eval baseline v2 SHARDED  task ${IDX}/${N_TASKS} ====="
echo "Method        : $METHOD"
echo "Checkpoint    : ${CKPT:-<none>}"
echo "Suite         : $SUITE (suite_i=$SUITE_I)"
echo "Shard         : $SHARD_I of $NUM_SHARDS"
echo "QnA dir       : $QNA_DIR"
echo "Output dir    : $OUT_DIR"
echo "Start         : $(date)"
nvidia-smi -L || true

python evaluation/run_baselines_v2.py \
    --method "$METHOD" \
    --qna-dir "$QNA_DIR" \
    --suites "$SUITE" \
    --output-dir "$OUT_DIR" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --bootstrap "$BOOTSTRAP" \
    --shard-i "$SHARD_I" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
