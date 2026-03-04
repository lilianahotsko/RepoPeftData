#!/bin/bash

#SBATCH --job-name=analyze_preprocessed_data
#SBATCH --output=gated_hypernetwork.out
#SBATCH --error=gated_hypernetwork.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --account=def-yuntian
#SBATCH --mail-user=lhotsko@uwaterloo.ca

# Load modules

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

# python embed_repos/4_construct_embeddings.py \
#     --repos-root /home/lhotsko/scratch/repos_without_tests \
#     --out-dir /home/lhotsko/scratch/repo_embeddings \
#     --device cuda \
#     --chunk-tokens 2048 \
#     --chunk-overlap 256 \
#     --batch-size 32
    # --model-name Qwen/Qwen3-Embedding-0.6B
# cd /home/lhotsko/RepoPeftData

# python embed_repos/6_construct_embeddings.py
# salloc --time=5:00:00 --gres=gpu:h100:1 --cpus-per-task=8 --mem=80G --account=def-yuntian
# salloc --time=05:00:00  --mem=32G --account=def-yuntian # allocate cpu only


# python baselines/pretrained/test_qwen_coder.py \
#     --output $SCRATCH/BASELINES/qwen_full.json \
#     --split cr_test

python baselines/rag/build_indices.py

# cd /home/lhotsko/RepoPeftData
# python hypernetwork/hypernetwork_sampled_test.py \
#     --checkpoint $SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos \
#     --splits cr_test

# python hypernetwork/hypernetwork.py \
#     --train-json /scratch/lhotsko/overfit_split/train_qna_pairs/test_next_block.jsonl \
#     --val-json /scratch/lhotsko/overfit_split/val_qna_pairs/test_next_block.jsonl \
#     --emb-dir /scratch/lhotsko/repo_embeddings

# python hypernetwork/hypernetwork_base.py \
#       --rank 8 --alpha 8 \
#   --lr 1e-4 \
#   --hidden-dim 512 \
#   --grad-accum 8 \
#   --epochs 2