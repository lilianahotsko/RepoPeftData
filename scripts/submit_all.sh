#!/bin/bash
# Submit all experiment phases with SLURM dependency chaining.
# Phases that can run in parallel are submitted concurrently.
#
# Usage:
#   bash scripts/submit_all.sh          # Submit everything
#   bash scripts/submit_all.sh --dry    # Print commands without submitting
#   bash scripts/submit_all.sh --from 5 # Start from phase 5

set -e

DRY_RUN=false
START_FROM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry) DRY_RUN=true; shift ;;
        --from) START_FROM=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p slurm_logs

submit() {
    local script=$1
    shift
    if $DRY_RUN; then
        echo "[DRY] sbatch $@ $script"
        echo "DRY_JOB_$(basename $script .sh)"
    else
        sbatch --parsable "$@" "$script"
    fi
}

echo "============================================"
echo "Submitting EMNLP experiment pipeline"
echo "Dry run: $DRY_RUN"
echo "Start from phase: $START_FROM"
echo "============================================"

# Phase 1: Embed (prerequisite for everything else)
if [ $START_FROM -le 1 ]; then
    JOB1=$(submit scripts/slurm/phase1_embed.sh)
    echo "Phase 1 (embed): $JOB1"
    DEP1="--dependency=afterok:$JOB1"
else
    DEP1=""
fi

# Phases 2-5 can run in parallel after Phase 1
if [ $START_FROM -le 2 ]; then
    JOB2=$(submit scripts/slurm/phase2_inference_baselines.sh $DEP1)
    echo "Phase 2 (inference baselines): $JOB2"
fi

if [ $START_FROM -le 3 ]; then
    JOB3=$(submit scripts/slurm/phase3_training_baselines.sh $DEP1)
    echo "Phase 3 (training baselines): $JOB3"
fi

if [ $START_FROM -le 4 ]; then
    JOB4=$(submit scripts/slurm/phase4_lora_per_repo.sh $DEP1)
    echo "Phase 4 (per-repo LoRA): $JOB4"
fi

if [ $START_FROM -le 5 ]; then
    JOB5=$(submit scripts/slurm/phase5_hypernet_eval.sh $DEP1)
    echo "Phase 5 (hypernet eval): $JOB5"
fi

# Phase 6: Composable (after Phase 1 for file embeddings)
if [ $START_FROM -le 6 ]; then
    JOB6=$(submit scripts/slurm/phase6_composable.sh $DEP1)
    echo "Phase 6 (composable weighted): $JOB6"
fi

# Phase 7: Composable variants (after Phase 6)
if [ $START_FROM -le 7 ]; then
    DEP6=""
    [ -n "${JOB6:-}" ] && DEP6="--dependency=afterok:$JOB6"
    JOB7=$(submit scripts/slurm/phase7_composable_variants.sh $DEP6)
    echo "Phase 7 (composable variants): $JOB7"
fi

# Phase 8: Scale (after Phase 1)
if [ $START_FROM -le 8 ]; then
    JOB8=$(submit scripts/slurm/phase8_scale.sh $DEP1)
    echo "Phase 8 (scale): $JOB8"
fi

# Phase 9: Ablations (after Phase 1)
if [ $START_FROM -le 9 ]; then
    JOB9=$(submit scripts/slurm/phase9_ablations.sh $DEP1)
    echo "Phase 9 (ablations): $JOB9"
fi

# Phase 10: Analysis (after everything)
if [ $START_FROM -le 10 ]; then
    ALL_DEPS=""
    for jid in ${JOB2:-} ${JOB3:-} ${JOB4:-} ${JOB5:-} ${JOB6:-} ${JOB7:-} ${JOB8:-} ${JOB9:-}; do
        if [ -n "$jid" ]; then
            if [ -z "$ALL_DEPS" ]; then
                ALL_DEPS="--dependency=afterok:$jid"
            else
                ALL_DEPS="$ALL_DEPS:$jid"
            fi
        fi
    done
    JOB10=$(submit scripts/slurm/phase10_analysis.sh $ALL_DEPS)
    echo "Phase 10 (analysis): $JOB10"
fi

echo ""
echo "============================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: slurm_logs/"
echo "============================================"
