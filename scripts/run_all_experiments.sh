#!/bin/bash
# Master script to run all experiments for the EMNLP submission.
# Execute on a machine with H100 GPU.
#
# Usage:
#   bash scripts/run_all_experiments.sh         # Run everything
#   bash scripts/run_all_experiments.sh baselines  # Only baselines
#   bash scripts/run_all_experiments.sh composable # Only composable experiments

set -e

STAGE=${1:-"all"}
SCRATCH=${SCRATCH:-$HOME/scratch}
SPLITS_DIR="$SCRATCH/REPO_DATASET"
REPOS_ROOT="$SCRATCH/REPO_DATASET/repositories"

echo "============================================"
echo "EMNLP Experiment Runner"
echo "SCRATCH: $SCRATCH"
echo "SPLITS_DIR: $SPLITS_DIR"
echo "Stage: $STAGE"
echo "============================================"

# ---- Step 0: Install dependencies ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "setup" ]; then
    echo "Installing codebleu..."
    pip install codebleu scikit-learn matplotlib 2>/dev/null || true
fi

# ---- Step 1: Re-embed repos with file-level embeddings ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "embed" ]; then
    echo ""
    echo "===== Step 1: Re-embed repos with file-level embeddings ====="
    python create_dataset/embed_repos.py \
        --repos-root "$REPOS_ROOT" \
        --overwrite

    echo "Recreating splits with file embeddings..."
    python create_dataset/create_splits.py \
        --repos-root "$REPOS_ROOT" \
        --out-dir "$SPLITS_DIR"
fi

# ---- Step 2: Baselines ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baselines" ]; then
    echo ""
    echo "===== Step 2: Running baselines ====="

    # 2a. Pretrained (on both CR and IR)
    echo "--- Pretrained baseline ---"
    python baselines/pretrained/test_qwen_coder.py \
        --splits-dir "$SPLITS_DIR" --split cr_test_structured \
        --output "$SCRATCH/BASELINES/pretrained_cr_test_structured.json"
    python baselines/pretrained/test_qwen_coder.py \
        --splits-dir "$SPLITS_DIR" --split ir_test_structured \
        --output "$SCRATCH/BASELINES/pretrained_ir_test_structured.json"

    # 2b. RAG baseline
    echo "--- RAG baseline ---"
    for k in 3 5 10; do
        python baselines/rag/test_rag.py \
            --splits-dir "$SPLITS_DIR" --repos-root "$REPOS_ROOT" \
            --split cr_test_structured --top-k $k \
            --output "$SCRATCH/BASELINES/rag_top${k}_cr_test_structured.json"
    done

    # 2c. ICL baseline
    echo "--- ICL baseline ---"
    for shots in 3 5; do
        python baselines/icl/test_icl.py \
            --splits-dir "$SPLITS_DIR" --split cr_test_structured --n-shots $shots \
            --output "$SCRATCH/BASELINES/icl_${shots}shot_cr_test_structured.json"
    done

    # 2d. Fine-tuned baseline
    echo "--- Fine-tuned baseline ---"
    python baselines/finetuned/train_finetuned.py \
        --splits-dir "$SPLITS_DIR" --no-wandb
    python baselines/finetuned/test_finetuned.py \
        --model-path "$SCRATCH/TRAINING_CHECKPOINTS/FINETUNED/final" \
        --splits-dir "$SPLITS_DIR" --split cr_test_structured \
        --output "$SCRATCH/BASELINES/finetuned_cr_test_structured.json"

    # 2e. Single LoRA baseline
    echo "--- Single LoRA baseline ---"
    python baselines/single_lora/train_single_lora.py \
        --splits-dir "$SPLITS_DIR" --no-wandb
    python baselines/single_lora/test_single_lora.py \
        --adapter "$SCRATCH/TRAINING_CHECKPOINTS/SINGLE_LORA/adapter" \
        --splits-dir "$SPLITS_DIR" --split cr_test_structured \
        --output "$SCRATCH/BASELINES/single_lora_cr_test_structured.json"

    # 2f. Per-repo LoRA (sample of repos)
    echo "--- Per-repo LoRA baseline ---"
    python baselines/lora_per_repo/run_all_repos.py \
        --splits-dir "$SPLITS_DIR" --limit-repos 20 --no-wandb
fi

# ---- Step 3: Hypernetwork (main method) ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "hypernet" ]; then
    echo ""
    echo "===== Step 3: Hypernetwork training + eval ====="
    python hypernetwork/hypernetwork_sampled.py \
        --splits-dir "$SPLITS_DIR" \
        --output-dir "$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos"

    python hypernetwork/hypernetwork_sampled_test.py \
        --checkpoint "$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos" \
        --splits-dir "$SPLITS_DIR" \
        --splits cr_test_structured ir_test_structured
fi

# ---- Step 4: Composable hypernetwork ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "composable" ]; then
    echo ""
    echo "===== Step 4: Composable hypernetwork ====="
    for comp in additive weighted gated; do
        echo "--- Composition: $comp ---"
        python hypernetwork/hypernetwork_composable.py \
            --splits-dir "$SPLITS_DIR" \
            --composition $comp \
            --output-dir "$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET_COMPOSABLE"
    done

    # Incremental adaptation experiment
    echo "--- Incremental adaptation ---"
    python hypernetwork/eval_incremental.py \
        --checkpoint "$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET_COMPOSABLE_weighted" \
        --splits-dir "$SPLITS_DIR" --split cr_test_structured --limit-repos 20
fi

# ---- Step 5: Scale experiments ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "scale" ]; then
    echo ""
    echo "===== Step 5: Scale experiments ====="
    bash scripts/run_scale_experiment.sh 0.5B
    bash scripts/run_scale_experiment.sh 3B
fi

# ---- Step 6: Ablations ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "ablations" ]; then
    echo ""
    echo "===== Step 6: Ablation studies ====="
    bash scripts/run_ablations.sh all
fi

# ---- Step 7: Analysis ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "analysis" ]; then
    echo ""
    echo "===== Step 7: Analysis ====="
    python analysis/analyze_results.py \
        --results-dir "$SCRATCH/BASELINES" \
        --output-dir analysis/output

    python analysis/visualize_loras.py \
        --checkpoint "$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos" \
        --splits-dir "$SPLITS_DIR" \
        --output-dir analysis/figures
fi

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results in: $SCRATCH/BASELINES/"
echo "Analysis in: analysis/output/"
echo "Paper in: paper/"
echo "============================================"
