#!/bin/bash
# ============================================================
# Train Doc2LoRA on Qwen2.5-Coder-1.5B with RepoPeftBench data
# Context distillation: KL loss with pre-generated teacher logprobs
#
# Prerequisites:
#   sbatch scripts/slurm/generate_d2l_logprobs.sh  (must complete first)
#
# Submit: sbatch scripts/slurm/train_doc2lora_qwen.sh
# ============================================================
#SBATCH --job-name=d2l-train
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/d2l_train_%j.out
#SBATCH --error=slurm_logs/d2l_train_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

OUTPUT_DIR="$CKPT_DIR/DOC2LORA_QWEN"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train"

# Verify teacher logprobs exist
if ! ls "$DATA_DIR"/*.parquet 1>/dev/null 2>&1; then
    echo "ERROR: No parquet files found in $DATA_DIR"
    echo "Run: sbatch scripts/slurm/generate_d2l_logprobs.sh"
    exit 1
fi

echo "==== Training Doc2LoRA (Qwen2.5-Coder-1.5B) ===="
echo "Config: doc2lora/configs/repopeft/qwen25_coder_15b_kl.yaml"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

cd doc2lora

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export WANDB_PROJECT="doc2lora_repopeft"
export WANDB_DIR="$SCRATCH/.wandb"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Single-GPU accelerate config (override the default multi-GPU config)
cat > /tmp/accelerate_single_gpu.yaml <<'ACCEL'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
ACCEL

accelerate launch --config_file /tmp/accelerate_single_gpu.yaml \
    train.py \
    configs/repopeft/qwen25_coder_15b_kl.yaml \
    --model_name_or_path=Qwen/Qwen2.5-Coder-1.5B \
    --output_dir="$OUTPUT_DIR" \
    --target_modules=down_proj \
    --lora_r=8 \
    --eval_strategy=no \
    --max_steps=20000 \
    --gradient_accumulation_steps=8 \
    --save_steps=2000 \
    --save_total_limit=3 \
    --logging_steps=50 \
    --learning_rate=4e-5 \
    --warmup_steps=200 \
    --use_kl_loss=True \
    --per_rank_gen=True \
    --per_layer_processing=True \
    --gen_lora_l1_reg_coef=0.1 \
    --use_per_ctx_average_loss=True \
    --max_packed_inp_len=4096 \
    --max_packed_ctx_len=4096

cd ..
echo "==== Done: $(date) ===="
echo "Checkpoints in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/ 2>/dev/null
