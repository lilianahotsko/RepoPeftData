#!/bin/bash
# Submit all Code2LoRA-GRU ablation experiments.
# Run from the repo root: bash scripts/slurm/run_gru_ablations.sh
#
# Prerequisites:
#   1. Run extract_commit_history.sh first (creates gru_*.json splits)
#   2. Ensure base model + embeddings are available on scratch

set -e
mkdir -p slurm_logs

echo "========================================="
echo "Code2LoRA-GRU Ablation Suite"
echo "========================================="

# --- 1. Main experiment (default config) ---
echo ""
echo "[1/8] Main: mamba2 init, h=1024, bptt=32, chronological"
INIT_TYPE=mamba2 GRU_HIDDEN=1024 BPTT_WINDOW=32 FILE_ORDER=chronological SUFFIX=main \
    sbatch scripts/slurm/train_code2lora_gru.sh

# --- 2. File ordering ablation ---
echo ""
echo "[2/8] File ordering ablation"
for ORDER in reverse random alphabetical; do
    INIT_TYPE=mamba2 GRU_HIDDEN=1024 BPTT_WINDOW=32 FILE_ORDER=$ORDER SUFFIX=order_${ORDER} \
        sbatch scripts/slurm/train_code2lora_gru.sh
done

# --- 3. h_0 initializer ablation ---
echo ""
echo "[3/8] Initializer ablation"
for INIT in zeros meanpool; do
    INIT_TYPE=$INIT GRU_HIDDEN=1024 BPTT_WINDOW=32 FILE_ORDER=chronological SUFFIX=init_${INIT} \
        sbatch scripts/slurm/train_code2lora_gru.sh
done

# --- 4. GRU hidden dim ablation ---
echo ""
echo "[4/8] Hidden dim ablation"
for HDIM in 512 2048; do
    INIT_TYPE=mamba2 GRU_HIDDEN=$HDIM BPTT_WINDOW=32 FILE_ORDER=chronological SUFFIX=hdim_${HDIM} \
        sbatch scripts/slurm/train_code2lora_gru.sh
done

# --- 5. BPTT window ablation ---
echo ""
echo "[5/8] BPTT window ablation"
for BPTT in 8 16 64; do
    INIT_TYPE=mamba2 GRU_HIDDEN=1024 BPTT_WINDOW=$BPTT FILE_ORDER=chronological SUFFIX=bptt_${BPTT} \
        sbatch scripts/slurm/train_code2lora_gru.sh
done

# --- 6. No truncated BPTT (full backprop) ---
echo ""
echo "[6/8] Full BPTT (no truncation)"
INIT_TYPE=mamba2 GRU_HIDDEN=1024 BPTT_WINDOW=9999 FILE_ORDER=chronological SUFFIX=fullbptt \
    sbatch scripts/slurm/train_code2lora_gru.sh

echo ""
echo "[7/8] Skipping architecture comparison (LSTM/Transformer) -- requires code changes"
echo "       See code2lora_gru.py for future architecture variants."

echo ""
echo "[8/8] Skipping preamble fraction ablation -- requires dataset regeneration"
echo "       Re-run extract_commit_history.py with --preamble-frac 0.05/0.20"

echo ""
echo "All ablation jobs submitted. Check with: squeue -u \$USER"
echo "Total jobs submitted: ~10"
