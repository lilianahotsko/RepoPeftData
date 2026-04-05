#!/bin/bash
# Submit missing baseline evaluations needed for the complete results table.
# Run from repo root: bash scripts/slurm/run_missing_baselines.sh
#
# Missing baselines:
#   1. Text2LoRA -- eval scripts exist, need results in consolidated table
#   2. Doc2LoRA -- eval scripts exist, need to run
#   3. Per-repo LoRA on CR test split (only IR results exist)
#   4. Code2LoRA Composable -- implemented but no results

set -e
mkdir -p slurm_logs logs

echo "========================================="
echo "Submit Missing Baselines"
echo "========================================="

# --- 1. Doc2LoRA ---
echo ""
echo "[1] Doc2LoRA evaluation"
if [ -f "doc2lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin" ]; then
    sbatch scripts/slurm/eval_doc2lora.sh
    echo "  -> Submitted"
else
    echo "  -> SKIPPED: Doc2LoRA checkpoint not found. Run: bash scripts/slurm/download_doc2lora.sh"
fi

# --- 2. Text2LoRA ---
echo ""
echo "[2] Text2LoRA code-conditioned evaluation"
if ls "$SCRATCH/TRAINING_CHECKPOINTS/TEXT2LORA/"* >/dev/null 2>&1; then
    for SPLIT in cr_test ir_test; do
        sbatch --job-name=t2l_${SPLIT} \
            --output=slurm_logs/text2lora_eval_${SPLIT}_%j.out \
            --error=slurm_logs/text2lora_eval_${SPLIT}_%j.err \
            --time=06:00:00 --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G \
            --account=def-yuntian \
            --wrap="source scripts/slurm/common.sh && \
                    python baselines/text2lora/evaluate_text2lora_code.py \
                        --split $SPLIT \
                        --splits-dir $SPLITS_DIR \
                        --output $BASELINES_DIR/text2lora_code_${SPLIT}.json"
    done
    echo "  -> Submitted"
else
    echo "  -> SKIPPED: Text2LoRA checkpoints not found"
fi

# --- 3. Per-repo LoRA on CR test ---
echo ""
echo "[3] Per-repo LoRA on cr_test"
sbatch --job-name=prl_cr \
    --output=slurm_logs/prl_cr_%j.out \
    --error=slurm_logs/prl_cr_%j.err \
    --time=24:00:00 --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G \
    --account=def-yuntian \
    --wrap="source scripts/slurm/common.sh && \
            python baselines/lora_per_repo/test_lora.py \
                --split cr_test \
                --splits-dir $SPLITS_DIR \
                --checkpoints-dir $CKPT_DIR/LORA_PER_REPO \
                --output $BASELINES_DIR/lora_per_repo_cr_test.json"
echo "  -> Submitted"

echo ""
echo "All missing baseline jobs submitted."
echo "Check with: squeue -u \$USER"
