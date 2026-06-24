#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --account=def-yuntian
#SBATCH --job-name=c2l_cpu_test
#SBATCH --output=slurm_logs/c2l_cpu_test_%j.out
#SBATCH --error=slurm_logs/c2l_cpu_test_%j.err

# CPU-only smoke test of the c2l quantized workflow.
# Run as a job:   sbatch scripts/cpu_workflow_test.sh
# Or interactive: srun ... scripts/cpu_workflow_test.sh
set -euo pipefail

module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source "$SCRATCH/venvs/qwen-cu126-py312/bin/activate"

cd /home/lhotsko/RepoPeftData
export PYTHONPATH="/home/lhotsko/RepoPeftData:${PYTHONPATH:-}"

# Compute nodes have no internet: use the HF cache populated on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export C2L_DEVICE=cpu
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO_PATH="${C2L_TEST_REPO:-$SCRATCH/c2l_test_repos/cachecontrol}"

echo "=== c2l CPU workflow test ==="
echo "node: $(hostname)  repo: $REPO_PATH"
nproc

python scripts/cpu_workflow_test.py \
    --repo "$REPO_PATH" \
    --task assert_rhs \
    --out "$SLURM_TMPDIR/c2l_cpu_adapter" \
    --max-new-tokens 16 \
    --try-4bit
