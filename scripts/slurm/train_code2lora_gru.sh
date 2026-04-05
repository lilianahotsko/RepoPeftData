#!/bin/bash
#SBATCH --job-name=train_gru
#SBATCH --output=slurm_logs/train_gru_%j.out
#SBATCH --error=slurm_logs/train_gru_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

INIT_TYPE="${INIT_TYPE:-mamba2}"
GRU_HIDDEN="${GRU_HIDDEN:-1024}"
BPTT_WINDOW="${BPTT_WINDOW:-32}"
FILE_ORDER="${FILE_ORDER:-chronological}"
SUFFIX="${SUFFIX:-default}"

OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/${INIT_TYPE}_h${GRU_HIDDEN}_bptt${BPTT_WINDOW}_${FILE_ORDER}_${SUFFIX}"

echo "===== Train: Code2LoRA-GRU ====="
echo "Init type:    $INIT_TYPE"
echo "GRU hidden:   $GRU_HIDDEN"
echo "BPTT window:  $BPTT_WINDOW"
echo "File order:   $FILE_ORDER"
echo "Output dir:   $OUT_DIR"
echo "Start: $(date)"

python hypernetwork/train_code2lora_gru.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUT_DIR" \
    --init-type "$INIT_TYPE" \
    --gru-hidden-dim "$GRU_HIDDEN" \
    --bptt-window "$BPTT_WINDOW" \
    --file-order "$FILE_ORDER" \
    --max-seq-len 8192 \
    --num-bases 16 \
    --trunk-depth 2 \
    --epochs 3 \
    --grad-accum 8 \
    --lr 1e-4

echo "Done: $(date)"
