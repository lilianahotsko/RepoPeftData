#!/bin/bash
# SLURM array job that runs the camera-ready Code2LoRA-GRU ablations.
# Each task trains one ablation row of the file-level GRU on the full
# RepoPeftBench split and writes results to $CKPT_DIR/CODE2LORA_GRU_ABLATIONS/.
#
# Submit with::
#
#   sbatch --array=0-29 scripts/slurm/ablations_code2lora_gru.sh
#
# Indices map to:
#   0..3   LoRA rank: 4, 8, 16, 32
#   4..6   GRU hidden: 256, 512, 1024
#   7..9   GRU init type: mamba2, meanpool, zeros
#   10..13 BPTT window: 16, 32, 64, full (=10000)
#   14..17 File order: chronological, reverse, alphabetical, random
#   18..20 PAW bases: 8, 16, 32
#   21..22 Embedding components: mean (1024), max (1024)
#                                (mean+max=2048 is the headline run)
#   23..26 Direct-projection \method{} probes:
#          rank 4 / 8 / 16 / 32   (separate launcher, see scripts/slurm/train_paw_hypernet.sh)
#   27..29 Three random seeds for the headline GRU_file run.

#SBATCH --job-name=gru_ablations
#SBATCH --output=slurm_logs/gru_ablation_%A_%a.out
#SBATCH --error=slurm_logs/gru_ablation_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

ABLATION_DIR="$CKPT_DIR/CODE2LORA_GRU_ABLATIONS"
mkdir -p "$ABLATION_DIR"

# Defaults shared by every ablation; the ablation-specific overrides clobber
# whichever variables they touch.
INIT_TYPE=mamba2
GRU_HIDDEN=1024
BPTT_WINDOW=32
FILE_ORDER=chronological
RANK=16
NUM_BASES=16
EMBED_COMPONENTS=mean_max
SEED=3407
EPOCHS=5

case "${SLURM_ARRAY_TASK_ID:-0}" in
  # ---- LoRA rank ----------------------------------------------------------
  0)  RANK=4   ; SUFFIX=rank4 ;;
  1)  RANK=8   ; SUFFIX=rank8 ;;
  2)  RANK=16  ; SUFFIX=rank16 ;;
  3)  RANK=32  ; SUFFIX=rank32 ;;

  # ---- GRU hidden dim -----------------------------------------------------
  4)  GRU_HIDDEN=256  ; SUFFIX=h256 ;;
  5)  GRU_HIDDEN=512  ; SUFFIX=h512 ;;
  6)  GRU_HIDDEN=1024 ; SUFFIX=h1024 ;;

  # ---- GRU init type ------------------------------------------------------
  7)  INIT_TYPE=mamba2   ; SUFFIX=init_mamba2 ;;
  8)  INIT_TYPE=meanpool ; SUFFIX=init_meanpool ;;
  9)  INIT_TYPE=zeros    ; SUFFIX=init_zeros ;;

  # ---- BPTT window --------------------------------------------------------
  10) BPTT_WINDOW=16    ; SUFFIX=bptt16 ;;
  11) BPTT_WINDOW=32    ; SUFFIX=bptt32 ;;
  12) BPTT_WINDOW=64    ; SUFFIX=bptt64 ;;
  13) BPTT_WINDOW=10000 ; SUFFIX=bptt_full ;;

  # ---- File order ---------------------------------------------------------
  14) FILE_ORDER=chronological ; SUFFIX=order_chrono ;;
  15) FILE_ORDER=reverse       ; SUFFIX=order_reverse ;;
  16) FILE_ORDER=alphabetical  ; SUFFIX=order_alpha ;;
  17) FILE_ORDER=random        ; SUFFIX=order_random ;;

  # ---- PAW bases ----------------------------------------------------------
  18) NUM_BASES=8  ; SUFFIX=bases8 ;;
  19) NUM_BASES=16 ; SUFFIX=bases16 ;;
  20) NUM_BASES=32 ; SUFFIX=bases32 ;;

  # ---- Embedding components (mean / max only vs. mean+max) ----------------
  # NOTE: pooling is fixed at dataset-construction time (embed_repos/), so
  # changing it requires re-emitting the splits with a different pool. We
  # keep these slots reserved but not run by default; see Section 6 in the
  # paper for the discussion.
  21) echo "[skip] emb_mean ablation requires re-pooled splits"; exit 0 ;;
  22) echo "[skip] emb_max  ablation requires re-pooled splits"; exit 0 ;;

  # ---- Slots 23-26 reserved for direct-projection rank scan (different script).
  23|24|25|26) echo "[skip] direct-projection rank ablations live in train_paw_hypernet.sh"; exit 0 ;;

  # ---- Three seeds for the headline GRU_file row --------------------------
  27) SEED=3407 ; SUFFIX=seed3407 ;;
  28) SEED=42   ; SUFFIX=seed42 ;;
  29) SEED=2026 ; SUFFIX=seed2026 ;;

  *)
    echo "ERROR: unhandled SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-?}" >&2
    exit 2
    ;;
esac

OUT_DIR="$ABLATION_DIR/$(printf 'a%02d' ${SLURM_ARRAY_TASK_ID:-0})_${SUFFIX}"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
case "$EMBED_COMPONENTS" in
  mean)     EXTRA_ARGS+=(--embed-components mean) ;;
  max)      EXTRA_ARGS+=(--embed-components max) ;;
  mean_max) ;;  # default
esac

echo "===== Ablation [${SLURM_ARRAY_TASK_ID}] $SUFFIX ====="
echo "  init=$INIT_TYPE hidden=$GRU_HIDDEN bptt=$BPTT_WINDOW order=$FILE_ORDER"
echo "  rank=$RANK bases=$NUM_BASES embed=$EMBED_COMPONENTS seed=$SEED epochs=$EPOCHS"
echo "  -> $OUT_DIR"

python hypernetwork/train_code2lora_gru.py \
  --splits-dir "$SPLITS_DIR" \
  --output-dir "$OUT_DIR" \
  --init-type "$INIT_TYPE" \
  --gru-hidden-dim "$GRU_HIDDEN" \
  --bptt-window "$BPTT_WINDOW" \
  --file-order "$FILE_ORDER" \
  --rank "$RANK" \
  --num-bases "$NUM_BASES" \
  --max-seq-len 8192 \
  --trunk-depth 2 \
  --epochs "$EPOCHS" \
  --grad-accum 8 \
  --lr 1e-4 \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

# Eval the resulting checkpoint on the canonical bench with bootstrap CIs.
CKPT="$OUT_DIR/code2lora_gru_best.pt"
if [ -f "$CKPT" ]; then
  python hypernetwork/eval_code2lora_gru.py \
    --checkpoint "$CKPT" \
    --splits-dir "$SPLITS_DIR" \
    --split cr_test \
    --bootstrap 5000 \
    --seed 3407 \
    --output "$OUT_DIR/eval_cr_test_ci.json"
  python hypernetwork/eval_code2lora_gru.py \
    --checkpoint "$CKPT" \
    --splits-dir "$SPLITS_DIR" \
    --split ir_test \
    --bootstrap 5000 \
    --seed 3407 \
    --output "$OUT_DIR/eval_ir_test_ci.json"
fi

echo "Done: $(date)"
