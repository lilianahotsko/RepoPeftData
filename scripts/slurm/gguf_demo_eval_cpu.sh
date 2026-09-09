#!/bin/bash
#SBATCH --job-name=gguf_demo_eval_cpu
#SBATCH --output=slurm_logs/gguf_demo_eval_cpu_%j.out
#SBATCH --error=slurm_logs/gguf_demo_eval_cpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=def-yuntian_cpu

# GGUF (Q4_K_M, llama.cpp) evaluation + CPU latency for the demo repo
# (0xricksanchez/like-dbg) at both demo commits: every assertion (assert_rhs,
# 16 tokens) and every QA pair (qa, 64 tokens), served as a quantized base
# GGUF + runtime LoRA GGUF via `llama-completion -no-cnv` -- exactly the demo's
# CPU serving path. Reports assertion EM / edit-sim, QA ROUGE-L / token-F1, and
# per-query llama.cpp latency (prefill + decode t/s, ms/token) on this CPU node.

set -euo pipefail
cd /home/lhotsko/RepoPeftData
mkdir -p slurm_logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "== hardware =="
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|MHz" || lscpu | head -20
echo "threads = ${SLURM_CPUS_PER_TASK:-8}"
echo "start   = $(date)"

python3 analysis/rebuttal/gguf_demo_eval.py \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --out /scratch/lhotsko/demo_repos/gguf_eval_likedbg.json

echo "done    = $(date)"
