#!/bin/bash
# ============================================================
# Generate teacher logprobs for Doc2LoRA context distillation
# Teacher: Qwen2.5-Coder-1.5B with DRC context in the prompt
# Output: D2L-compatible parquet files with top-16 logprobs
#
# Submit: sbatch scripts/slurm/generate_d2l_logprobs.sh
# ============================================================
#SBATCH --job-name=d2l-logprobs
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/d2l_logprobs_%j.out
#SBATCH --error=slurm_logs/d2l_logprobs_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

OUTPUT_DIR="doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train"

echo "==== Generating D2L Teacher Logprobs ===="
echo "Model: Qwen/Qwen2.5-Coder-1.5B"
echo "Oracle cache: $SCRATCH/ORACLE_CONTEXT_CACHE_V4"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

python baselines/doc2lora/generate_teacher_logprobs.py \
    --model Qwen/Qwen2.5-Coder-1.5B \
    --split train \
    --splits-dir "$SPLITS_DIR" \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V4" \
    --output-dir "$OUTPUT_DIR" \
    --max-ctx-tokens 4096 \
    --max-teacher-tokens 8192 \
    --max-input-tokens 2048 \
    --shard-size 50

echo "==== Done: $(date) ===="
echo "Parquet files written to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "(no parquet files found)"
