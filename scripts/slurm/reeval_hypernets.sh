#!/bin/bash
#SBATCH --job-name=reeval_hnet
#SBATCH --output=slurm_logs/reeval_hypernets_%j.out
#SBATCH --error=slurm_logs/reeval_hypernets_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Re-evaluate Code2LoRA Direct + PAW with fixed postprocess_prediction.

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Re-eval: Hypernetworks (fixed metrics) ====="
echo "Start: $(date)"

echo ""
echo "========== CODE2LORA (DIRECT) =========="
CKPT_PATH="$CKPT_DIR/HYPERNET/no_oracle"
if [[ -d "$CKPT_PATH" ]]; then
    python hypernetwork/hypernetwork_sampled_test.py \
        --checkpoint "$CKPT_PATH" \
        --splits-dir "$SPLITS_DIR" \
        --splits cr_test ir_test

    RESULTS_BASE="${CKPT_PATH}_results"
    for split in cr_test ir_test; do
        if [[ -f "$RESULTS_BASE/$split/results.json" ]]; then
            cp "$RESULTS_BASE/$split/results.json" "$BASELINES_DIR/hypernet_no_oracle_${split}.json"
            echo "Copied $split results"
        fi
    done
else
    echo "WARN: Hypernet checkpoint not found at $CKPT_PATH"
fi

echo ""
echo "========== CODE2LORA (PAW) =========="
CKPT_PATH="$CKPT_DIR/HYPERNET_PAW/no_oracle"
if [[ -f "$CKPT_PATH/lora_mapper_best.pt" ]]; then
    for split in cr_test ir_test; do
        echo "--- $split ---"
        python hypernetwork/hypernetwork_paw_test.py \
            --checkpoint "$CKPT_PATH" \
            --splits-dir "$SPLITS_DIR" \
            --split $split
    done

    RESULTS_BASE="${CKPT_PATH}_results"
    for split in cr_test ir_test; do
        if [[ -f "$RESULTS_BASE/$split/results.json" ]]; then
            cp "$RESULTS_BASE/$split/results.json" "$BASELINES_DIR/hypernet_paw_no_oracle_${split}.json"
            echo "Copied $split results"
        fi
    done
else
    echo "WARN: PAW checkpoint not found at $CKPT_PATH"
fi

echo ""
echo "Done: $(date)"
