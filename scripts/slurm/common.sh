#!/bin/bash
# Common setup sourced by all SLURM scripts.
# Do not submit this directly.

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

cd /home/lhotsko/RepoPeftData

export SPLITS_DIR="$SCRATCH/REPO_DATASET"
export REPOS_ROOT="$SCRATCH/REPO_DATASET/repositories"
export BASELINES_DIR="$SCRATCH/BASELINES"
export CKPT_DIR="$SCRATCH/TRAINING_CHECKPOINTS"

mkdir -p "$BASELINES_DIR"
