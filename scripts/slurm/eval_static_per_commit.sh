#!/bin/bash
# Per-commit decay evaluation for Code2LoRA-direct (static) on all suites.
# Output JSON has the same schema as the GRU per-commit eval, so the unified
# plotter can overlay both models on a single decay figure.
#
# Inputs:
#   CHECKPOINT (env)  = path to hypernetwork_best.pt
#   SPLITS_DIR (env)  = dir containing ir_val.json / ir_test.json / cr_val.json /
#                       cr_test.json / ood_test.json
# Output:
#   $OUT_JSON

#SBATCH --job-name=eval_static_timeline
#SBATCH --output=slurm_logs/eval_static_timeline_%j.out
#SBATCH --error=slurm_logs/eval_static_timeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=def-yuntian

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/REPO_DATASET/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

: "${CHECKPOINT:?must set CHECKPOINT=/path/to/hypernetwork_best.pt}"
: "${SPLITS_DIR:?must set SPLITS_DIR=/path/to/splits}"
OUT_JSON="${OUT_JSON:-$(dirname "$CHECKPOINT")/bench_results/static_per_commit_timeline.json}"

SUITES="${SUITES:-ir_val ir_test cr_val cr_test ood_test}"
MAX_PC="${MAX_PC:-64}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

mkdir -p "$(dirname "$OUT_JSON")"

echo "Checkpoint : $CHECKPOINT"
echo "Splits     : $SPLITS_DIR"
echo "Suites     : $SUITES"
echo "Output     : $OUT_JSON"

# shellcheck disable=SC2086
python evaluation/eval_code2lora_static_per_commit.py \
    --checkpoint "$CHECKPOINT" \
    --splits-dir "$SPLITS_DIR" \
    --suites $SUITES \
    --max-assertions-per-commit "$MAX_PC" \
    --bootstrap "$BOOTSTRAP" \
    --output-json "$OUT_JSON" \
    "$@"

echo "Done: $(date)"
