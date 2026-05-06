#!/bin/bash
# Tiny SLURM job that runs the Text2LoRA injection diagnostic against
# a known-trained checkpoint. The output (printed to stderr/stdout)
# tells you whether the LoRA is being injected / non-trivial.
#
# Submit with::
#   HYPERMOD_DIR=text2lora/train_outputs/recon/hyper_lora/<run_name> \
#       sbatch scripts/slurm/diagnose_t2l.sh
#
# Defaults to the current Text2LoRA(text) run if HYPERMOD_DIR is unset.

#SBATCH --job-name=t2l_diag
#SBATCH --output=slurm_logs/t2l_diag_%j.out
#SBATCH --error=slurm_logs/t2l_diag_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

HYPERMOD_DIR="${HYPERMOD_DIR:-text2lora/train_outputs/recon/hyper_lora/20260325-232815_R9TsaGJT}"

if [ ! -f "$HYPERMOD_DIR/hypermod.pt" ]; then
  echo "ERROR: $HYPERMOD_DIR/hypermod.pt not found" >&2
  exit 1
fi

echo "===== T2L injection diagnostic ====="
echo "Checkpoint: $HYPERMOD_DIR"

python baselines/text2lora/diagnose_t2l_injection.py \
  --hypermod-dir "$HYPERMOD_DIR"
