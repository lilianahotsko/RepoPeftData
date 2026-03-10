#!/bin/bash
# Quick status check: jobs, recent logs, latest results.

echo "===== JOB QUEUE ====="
squeue -u $USER --format="%.10i %.14j %.8T %.10M %.15R" --sort=i 2>/dev/null

echo ""
echo "===== RECENT LOG TAILS ====="
for f in slurm_logs/reeval_*.out slurm_logs/prlora_chunk*.out slurm_logs/train_*oracle*.out; do
    if [[ -f "$f" ]]; then
        echo ""
        echo "--- $f ---"
        tail -3 "$f"
    fi
done

echo ""
echo "===== RESULTS SUMMARY ====="
cd /home/lhotsko/RepoPeftData
python3 analysis/collect_results.py 2>/dev/null || echo "(collect_results.py failed)"
