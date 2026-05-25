#!/bin/bash
#SBATCH --job-name=plot_dataset_construction
#SBATCH --output=slurm_logs/plot_dataset_construction_%j.out
#SBATCH --error=slurm_logs/plot_dataset_construction_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --account=def-yuntian_cpu

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

python RepoPeft_Paper/scripts/plot_dataset_construction.py \
  --output RepoPeft_Paper/figures/dataset_construction.pdf
