#!/bin/bash
# Evaluation launcher for the hypernetwork-size ablation.
#
# For each trained size (GRU and static), submits the existing *sharded* eval
# array against that size's checkpoint, writing to a size-encoded eval OUT_DIR.
# It is SAFE to run repeatedly: a size whose training checkpoint does not yet
# exist is skipped, so you can re-run this as jobs finish (or just run it once
# after all 8 training jobs complete).
#
# Checkpoints expected (produced by the training sweeps):
#   GRU    : $CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep_h{H}_g{G}/gru_head.best.pt
#   static : $CKPT_DIR/CODE2LORA_STATIC/h100_v2_static_3ep_h{H}/head.latest.pt
#            (static slurm runs with SKIP_EVAL=1, so there is no best.pt --
#             head.latest.pt is the final-epoch checkpoint.)
#
# Eval outputs (one dir per size; merge shards there afterwards):
#   GRU    : $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/gru_h{H}_g{G}/
#   static : $CKPT_DIR/CODE2LORA_STATIC_EVAL_V2/static_h{H}/
#
# This is a *submit* launcher: run it on the login node, NOT via sbatch.
#
# Usage:
#   bash scripts/slurm/sweep_hypernet_size_eval.sh
#   SUITES="ir_test cr_test" SIZES="0.5 1 2 4" bash scripts/slurm/sweep_hypernet_size_eval.sh
#   VARIANTS="gru" bash scripts/slurm/sweep_hypernet_size_eval.sh   # GRU only
#
# After all arrays finish, merge each eval dir:
#   for d in $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/gru_h* \
#            $CKPT_DIR/CODE2LORA_STATIC_EVAL_V2/static_h*; do
#       python evaluation/merge_eval_shards.py --auto-detect --input-dir "$d"
#   done

set -euo pipefail
cd /home/lhotsko/RepoPeftData
source scripts/slurm/common.sh

SIZES="${SIZES:-0.5 1 2 4}"
VARIANTS="${VARIANTS:-gru static}"
# Ablation reporting uses the held-out test suites; override to add val suites.
export SUITES="${SUITES:-ir_test cr_test}"
export NUM_SHARDS="${NUM_SHARDS:-4}"

# Optional Slurm account override (e.g. ACCOUNT=rrg-yuntian).
ACCOUNT="${ACCOUNT:-}"
ACCOUNT_ARGS=()
[ -n "$ACCOUNT" ] && ACCOUNT_ARGS+=(--account "$ACCOUNT")

# array size = n_suites * num_shards - 1
N_SUITES=$(echo "$SUITES" | wc -w)
ARRAY_MAX=$(( N_SUITES * NUM_SHARDS - 1 ))

echo "===== Hypernet-size ablation EVAL launcher ====="
echo "Variants : $VARIANTS"
echo "Sizes    : $SIZES"
echo "Suites   : $SUITES  (array 0-$ARRAY_MAX, shards=$NUM_SHARDS)"
echo "Start    : $(date)"

for m in $SIZES; do
    H=$(awk "BEGIN{printf \"%d\", 1024*$m}")
    G=$(awk "BEGIN{printf \"%d\", 2048*$m}")

    for variant in $VARIANTS; do
        case "$variant" in
            gru)
                CKPT="$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep_h${H}_g${G}/gru_head.best.pt"
                SUFFIX="gru_h${H}_g${G}"
                EVAL_SCRIPT="scripts/slurm/eval_code2lora_gru_v2_sharded.sh"
                ;;
            static)
                CKPT="$CKPT_DIR/CODE2LORA_STATIC/h100_v2_static_3ep_h${H}/head.latest.pt"
                SUFFIX="static_h${H}"
                EVAL_SCRIPT="scripts/slurm/eval_code2lora_static_v2_sharded.sh"
                ;;
            *)
                echo "  [skip] unknown variant '$variant'"; continue ;;
        esac

        if [ ! -f "$CKPT" ]; then
            echo "  [skip] $variant m=$m : checkpoint not found yet ($CKPT)"
            continue
        fi
        echo "  [submit] $variant m=$m : CKPT=$CKPT SUFFIX=$SUFFIX"
        CKPT="$CKPT" SUFFIX="$SUFFIX" SUITES="$SUITES" NUM_SHARDS="$NUM_SHARDS" \
            sbatch "${ACCOUNT_ARGS[@]}" \
                   --array=0-"$ARRAY_MAX" \
                   --job-name="eval_${SUFFIX}" \
                   "$EVAL_SCRIPT"
    done
done

echo "Eval launcher done: $(date)"
