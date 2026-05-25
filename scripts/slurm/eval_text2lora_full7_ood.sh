#!/bin/bash
# Extract OOD repo embeddings, merge with v2 pool, then submit GPU eval on
# rrg-yuntian_gpu (RRG account; prep requests a GPU so the job can run on RRG).
#
# Submit from repo root:
#   sbatch scripts/slurm/eval_text2lora_full7_ood.sh
#
# Outputs:
#   $CKPT_DIR/BASELINES_V2/text2lora_h100_v2_full7_ood_sharded/

#SBATCH --job-name=t2l-full7-ood-prep
#SBATCH --output=slurm_logs/t2l_full7_ood_prep_%j.out
#SBATCH --error=slurm_logs/t2l_full7_ood_prep_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=rrg-yuntian_gpu

set -euo pipefail
source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
RRG_ACCOUNT="${RRG_ACCOUNT:-rrg-yuntian_gpu}"

SCRATCH="${SCRATCH:-/scratch/lhotsko}"
# OOD commits with repo_state_embedding live under the snapshots build;
# QnAs come from the v2 OOD bundle.
OOD_COMMITS_DIR="${OOD_COMMITS_DIR:-$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits}"
OOD_QNA_DIR="${OOD_QNA_DIR:-$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_ood/qna}"
BASE_EMB="${BASE_EMB:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt}"
OOD_EMB="${OOD_EMB:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2_ood.pt}"
MERGED_EMB="${MERGED_EMB:-$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2_with_ood.pt}"
HYPERMOD_DIR="${HYPERMOD_DIR:-$SCRATCH/text2lora_train_outputs/sft/hyper_lora/code_sft_v2_full7_14789268}"

echo "===== Text2LoRA full7 OOD: extract + merge embeddings ====="
echo "OOD commits : $OOD_COMMITS_DIR"
echo "OOD QnA     : $OOD_QNA_DIR"
echo "Hypermod    : $HYPERMOD_DIR"
echo "RRG account : $RRG_ACCOUNT"
echo "Start       : $(date)"

if [ ! -f "$OOD_COMMITS_DIR/ood_test.parquet" ]; then
    echo "[error] missing $OOD_COMMITS_DIR/ood_test.parquet" >&2
    exit 1
fi
if [ ! -f "$OOD_QNA_DIR/ood_test.parquet" ]; then
    echo "[error] missing $OOD_QNA_DIR/ood_test.parquet" >&2
    exit 1
fi
if [ ! -f "$HYPERMOD_DIR/hypermod.pt" ]; then
    echo "[error] missing $HYPERMOD_DIR/hypermod.pt" >&2
    exit 1
fi

python baselines/text2lora/extract_code_embeddings_v2.py \
    --commits-dir "$OOD_COMMITS_DIR" \
    --include-suites ood_test \
    --output "$OOD_EMB"

python scripts/merge_code_embeddings.py \
    --output "$MERGED_EMB" \
    "$BASE_EMB" \
    "$OOD_EMB"

EVAL_ID=$(METHOD=text2lora \
    TEXT2LORA_HYPERMOD_DIR="$HYPERMOD_DIR" \
    TEXT2LORA_CODE_EMB_PATH="$MERGED_EMB" \
    QNA_DIR="$OOD_QNA_DIR" \
    SUFFIX=h100_v2_full7_ood_sharded \
    NUM_SHARDS=4 \
    SUITES="ood_test" \
    sbatch --parsable \
        --account="$RRG_ACCOUNT" \
        --gres=gpu:h100:1 \
        --array=0-3 \
        scripts/slurm/eval_baselines_v2_sharded.sh)

echo "Submitted OOD eval array on $RRG_ACCOUNT: $EVAL_ID"
echo "$EVAL_ID" > .last_t2l_full7_ood_eval_jobid
echo "Merged embeddings: $MERGED_EMB"
echo "Done            : $(date)"
