#!/bin/bash
#SBATCH --job-name=train_gru_diff
#SBATCH --output=slurm_logs/train_gru_diff_%j.out
#SBATCH --error=slurm_logs/train_gru_diff_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#
# Code2LoRA-GRU commit-level training (diff-based, max 4 assertions per commit).
# watgpu cluster — same hyperparameters as scripts/slurm/train_code2lora_gru_commits_4max.sh
#
# Prerequisites:
#   - bash scripts/watgpu/setup_env.sh   (venv + base data)
#   - commits_assertions.db under SPLITS_DIR (from HF snapshot if published, or build:
#       python create_dataset/build_commit_assertion_db.py --splits-dir "$SPLITS_DIR" ...
#     with REPOS_ROOT pointing at cloned repos; see scripts/slurm/build_commit_assertion_db.sh)
#
# Submit from repo root:  sbatch scripts/watgpu/train_code2lora_gru_commits_4max.sh

source scripts/watgpu/common.sh

DB_PATH="${DB_PATH:-$SPLITS_DIR/commits_assertions.db}"
SUFFIX="${SUFFIX:-default}"
# Cap training repos (first N from train split). Unset or empty = use all.
LIMIT_TRAIN_REPOS=10
OUT_DIR="$CKPT_DIR/CODE2LORA_GRU/commit_level_${SUFFIX}"

echo "===== Train: Code2LoRA-GRU (commit-level, diff-based) [watgpu] ====="
echo "DB path:       $DB_PATH"
echo "Splits dir:    $SPLITS_DIR"
echo "Output dir:    $OUT_DIR"
echo "Train repos:   ${LIMIT_TRAIN_REPOS:-all}"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"

python hypernetwork/train_code2lora_gru_commits.py \
    --db-path "$DB_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUT_DIR" \
    --model-name "Qwen/Qwen2.5-Coder-1.5B" \
    --embed-model "Qwen/Qwen3-Embedding-0.6B" \
    --init-type zeros \
    --gru-hidden-dim 1024 \
    --gru-num-layers 1 \
    --rank 16 \
    --alpha 32 \
    --num-bases 16 \
    --trunk-depth 2 \
    --lora-hidden-dim 512 \
    --chunk-tokens 512 \
    --chunk-overlap 64 \
    --max-seq-len 8192 \
    --max-assertions-per-commit 4 \
    --grad-accum 4 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --max-grad-norm 5.0 \
    --epochs 3 \
    --eval-steps 200 \
    --save-steps 500 \
    --logging-steps 10 \
    --seed 3407 \
    --no-initial-eval \
    ${LIMIT_TRAIN_REPOS:+--limit-train-repos "$LIMIT_TRAIN_REPOS"}

echo "Done: $(date)"
