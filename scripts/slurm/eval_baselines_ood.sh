#!/bin/bash
# ============================================================================
# Run OOD-test evaluation for the baselines that work *out of the box* with
# the static ``ood_test.json`` produced by ``evaluation/build_ood_bench_json.py``.
#
# Methods evaluated (no per-repo preprocessing required):
#   * pretrained        (Qwen2.5-Coder-1.5B, no fine-tuning)
#   * fft_no_oracle     (full fine-tuning)
#   * single_lora       (single LoRA shared across repos)
#
# Methods that REQUIRE additional preprocessing on the OOD repos before they
# can be evaluated and are *therefore skipped here*:
#   * RAG / RAG-256        -> needs a chunk-cache per OOD repo
#   * DRC / oracle_context -> needs a resolved-dependency cache per OOD repo
#   * ICL                  -> needs per-repo Qwen3 embeddings (or a code change
#                              to draw random examples from train_repos)
#   * Code2LoRA-direct     -> needs per-repo Qwen3 embedding (mean+max pool)
#   * Code2LoRA-GRU-file   -> needs per-file Qwen3 embeddings
#   * Text2LoRA-code       -> needs per-repo Qwen3 embedding
#   * Text2LoRA-text       -> can run if you provide a textual description
#
# Usage:
#   bash scripts/slurm/eval_baselines_ood.sh
#
# Submits one SLURM job per method.
# ============================================================================
set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

SPLIT="${SPLIT:-ood_test}"
JSON="${SPLITS_DIR:-$SCRATCH/REPO_DATASET}/${SPLIT}.json"
if [[ ! -f "$JSON" ]]; then
    echo "ERROR: $JSON not found. Run evaluation/build_ood_bench_json.py first."
    exit 1
fi

n_pairs=$(python -c "import json; d=json.load(open('$JSON')); print(sum(len(r.get('qna_pairs',[])) for r in d.get('repositories',{}).values()))")
n_repos=$(python -c "import json; d=json.load(open('$JSON')); print(len(d.get('repositories',{})))")
echo "Submitting OOD evals on $JSON: $n_repos repos, $n_pairs pairs"

# ------------------------------------------------------------------
# 1) Pretrained
# ------------------------------------------------------------------
JOB_PRETRAINED=$(sbatch --parsable \
    --job-name=ood_pretrained \
    --account=def-yuntian --time=05:00:00 \
    --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G \
    --output=slurm_logs/ood_pretrained_%j.out \
    --error=slurm_logs/ood_pretrained_%j.err \
    --wrap="set -euo pipefail; source scripts/slurm/common.sh; \
        echo '== Pretrained on $SPLIT =='; \
        python baselines/pretrained/test_qwen_coder.py \
            --splits-dir \"\$SPLITS_DIR\" \
            --split $SPLIT \
            --output \"\$BASELINES_DIR/pretrained_${SPLIT}.json\"")
echo "  pretrained        -> job $JOB_PRETRAINED"

# ------------------------------------------------------------------
# 2) FFT (no-oracle)
# ------------------------------------------------------------------
FFT_PATH="$CKPT_DIR/FFT/final"
if [[ -d "$FFT_PATH" ]]; then
    JOB_FFT=$(sbatch --parsable \
        --job-name=ood_fft \
        --account=def-yuntian --time=05:00:00 \
        --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G \
        --output=slurm_logs/ood_fft_%j.out \
        --error=slurm_logs/ood_fft_%j.err \
        --wrap="set -euo pipefail; source scripts/slurm/common.sh; \
            echo '== FFT (no-oracle) on $SPLIT =='; \
            python baselines/finetuned/test_finetuned.py \
                --model-path \"$FFT_PATH\" \
                --splits-dir \"\$SPLITS_DIR\" \
                --split $SPLIT \
                --output \"\$BASELINES_DIR/fft_no_oracle_${SPLIT}.json\"")
    echo "  fft_no_oracle     -> job $JOB_FFT"
else
    echo "  fft_no_oracle     [skip: $FFT_PATH not found]"
fi

# ------------------------------------------------------------------
# 3) Single LoRA (no-oracle)
# ------------------------------------------------------------------
SLORA_PATH="$CKPT_DIR/SINGLE_LORA/adapter"
if [[ -d "$SLORA_PATH" ]]; then
    JOB_SLORA=$(sbatch --parsable \
        --job-name=ood_slora \
        --account=def-yuntian --time=05:00:00 \
        --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G \
        --output=slurm_logs/ood_slora_%j.out \
        --error=slurm_logs/ood_slora_%j.err \
        --wrap="set -euo pipefail; source scripts/slurm/common.sh; \
            echo '== Single LoRA (no-oracle) on $SPLIT =='; \
            python baselines/single_lora/test_single_lora.py \
                --adapter \"$SLORA_PATH\" \
                --splits-dir \"\$SPLITS_DIR\" \
                --split $SPLIT \
                --output \"\$BASELINES_DIR/single_lora_no_oracle_${SPLIT}.json\"")
    echo "  single_lora       -> job $JOB_SLORA"
else
    echo "  single_lora       [skip: $SLORA_PATH not found]"
fi

echo
echo "Submitted. Watch progress with: squeue -u \$USER"
echo "Outputs land at \$BASELINES_DIR/<method>_${SPLIT}.json"
