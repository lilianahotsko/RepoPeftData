#!/bin/bash
# ============================================================
# Train Doc2LoRA on Qwen2.5-Coder-1.5B against the v2 commit-derived
# RepoPeftBench dataset. Context distillation with KL loss against
# pre-generated teacher logprobs that condition on per-commit DRC.
#
# Output directory is pinned to $CKPT_DIR/DOC2LORA_V2/<SLURM_JOB_ID> so
# a queued eval array can target the artifact path before training has
# even started (override with DOC2LORA_OUTPUT_DIR=...).
#
# Prerequisites:
#   1. NUM_SHARDS=4 SUITES=train sbatch --array=0-3 \
#        scripts/slurm/build_drc_cache_per_commit.sh    # ~30 min CPU
#   2. NUM_SHARDS=4 sbatch --array=0-3 \
#        scripts/slurm/generate_d2l_logprobs_v2.sh      # ~3-4 h H100
#
# Submit:
#   sbatch scripts/slurm/train_doc2lora_v2.sh
# ============================================================
#SBATCH --job-name=d2l-train-v2
#SBATCH --account=rrg-yuntian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/d2l_train_v2_%j.out
#SBATCH --error=slurm_logs/d2l_train_v2_%j.err

set -euo pipefail
mkdir -p slurm_logs

source scripts/slurm/common.sh

# Pin the run-dir from the SLURM job id (override via DOC2LORA_OUTPUT_DIR)
# so a queued eval array can target this path before training starts.
DOC2LORA_OUTPUT_DIR="${DOC2LORA_OUTPUT_DIR:-$CKPT_DIR/DOC2LORA_V2/${SLURM_JOB_ID}}"
mkdir -p "$DOC2LORA_OUTPUT_DIR"

DATA_DIR="doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train_v2"
CONFIG="configs/repopeft/qwen25_coder_15b_kl_v2.yaml"

# Verify teacher logprobs exist (the trainer's data loader globs
# <DATA_DIR>/<split>/*.parquet -- see ctx_to_lora.data.processing:
# get_ds_kwargs, line ~128).
if ! ls "$DATA_DIR"/train/*.parquet 1>/dev/null 2>&1; then
    echo "ERROR: No parquet files found in $DATA_DIR/train/"
    echo "  Generate them with:"
    echo "    NUM_SHARDS=4 sbatch --array=0-3 scripts/slurm/generate_d2l_logprobs_v2.sh"
    exit 1
fi

echo "==== Training Doc2LoRA v2 (Qwen2.5-Coder-1.5B) ===="
echo "Config       : doc2lora/$CONFIG"
echo "Data         : $DATA_DIR"
echo "Output dir   : $DOC2LORA_OUTPUT_DIR"
echo "Start        : $(date)"
nvidia-smi -L || true

cd doc2lora

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export WANDB_PROJECT="doc2lora_repopeft_v2"
export WANDB_DIR="$SCRATCH/.wandb"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

# Single-GPU accelerate config (override the default multi-GPU config).
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
    "$CONFIG" \
    --model_name_or_path=Qwen/Qwen2.5-Coder-1.5B \
    --output_dir="$DOC2LORA_OUTPUT_DIR" \
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
echo "Checkpoints in: $DOC2LORA_OUTPUT_DIR"
ls -la "$DOC2LORA_OUTPUT_DIR"/ 2>/dev/null
