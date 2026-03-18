#!/bin/bash
#SBATCH --job-name=train_hnet_drc8k
#SBATCH --output=slurm_logs/train_hnet_drc8k_%j.out
#SBATCH --error=slurm_logs/train_hnet_drc8k_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=rrg-yuntian

source scripts/slurm/common.sh
mkdir -p slurm_logs

echo "===== Train: Hypernetwork (Code2LoRA) + DRC v2, max_seq=8192 ====="
echo "Start: $(date)"

# Saves to HYPERNET/drc8k — does NOT overwrite HYPERNET/oracle (v1 cache)
python hypernetwork/hypernetwork_sampled.py \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$CKPT_DIR/HYPERNET/drc8k" \
    --use-oracle \
    --oracle-cache-dir "$SCRATCH/ORACLE_CONTEXT_CACHE_V2" \
    --max-seq-len 8192 \
    --epochs 3 --grad-accum 8

echo "Done: $(date)"
