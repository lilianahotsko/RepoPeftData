#!/bin/bash
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

# salloc --time=05:00:00  --mem=32G --account=def-yuntian
# cd $SCRATCH

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

cd /home/lhotsko/RepoPeftData
python repos_collection/clean_repos.py