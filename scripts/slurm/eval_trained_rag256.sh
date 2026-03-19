#!/bin/bash
#SBATCH --job-name=eval_trained_rag256
#SBATCH --output=slurm_logs/eval_trained_rag256_%j.out
#SBATCH --error=slurm_logs/eval_trained_rag256_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

CHUNK_CACHE="$SCRATCH/RAG_CHUNK_CACHE_256"

echo "===== Eval: FFT + RAG-256 (k=5), sLoRA + RAG-256 (k=5) ====="
echo "Start: $(date)"

for split in cr_test ir_test; do
    echo ""
    echo "--- FFT + RAG-256 k=5, split=$split ---"
    python baselines/finetuned/test_finetuned.py \
        --model-path "$CKPT_DIR/FFT/final" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --rag-cache-dir "$CHUNK_CACHE" \
        --top-k 5 \
        --output "$BASELINES_DIR/fft_rag256_k5_${split}.json"

    echo ""
    echo "--- sLoRA + RAG-256 k=5, split=$split ---"
    python baselines/single_lora/test_single_lora.py \
        --adapter "$CKPT_DIR/SINGLE_LORA/adapter" \
        --splits-dir "$SPLITS_DIR" \
        --split "$split" \
        --rag-cache-dir "$CHUNK_CACHE" \
        --top-k 5 \
        --output "$BASELINES_DIR/slora_rag256_k5_${split}.json"
done

echo ""
echo "Done: $(date)"
