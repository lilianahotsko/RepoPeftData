#!/bin/bash
# OOD eval of the released Code2LoRA-GRU checkpoint on the *published* OOD set
# (code2lora/code2lora-data-ood: commits.parquet + qna.parquet, raw diffs that
# are embedded on the fly). Same per-commit protocol as training (h_k walk),
# headline number = LoRA frozen at h_T scored over every held-out assertion.
#
# Env knobs:
#   CHECKPOINT   path to code2lora_gru.pt   (default: cached HF snapshot)
#   COMMITS_PQ   commits.parquet            (default: cached HF snapshot)
#   QNA_PQ       qna.parquet                (default: cached HF snapshot)
#   SMOKE=1      quick validation (2 repos, capped assertions, no bootstrap)
#   LIMIT_REPOS  cap repos (overrides SMOKE default)

#SBATCH --job-name=eval_gru_ood_pub
#SBATCH --output=slurm_logs/eval_gru_ood_pub_%j.out
#SBATCH --error=slurm_logs/eval_gru_ood_pub_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-yuntian_gpu

set -euo pipefail

source scripts/slurm/common.sh
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

HUB="$HOME/.cache/huggingface/hub"
CKPT_SNAP="$HUB/models--code2lora--code2lora-gru/snapshots/26460cba5f0de9b708277488ae0b1a826435961e"
OOD_SNAP="$HUB/datasets--code2lora--code2lora-data-ood/snapshots/d8f19d7ed7fba66f7bd634ae8c4b5566c48e432f"

CHECKPOINT="${CHECKPOINT:-$CKPT_SNAP/code2lora_gru.pt}"
COMMITS_PQ="${COMMITS_PQ:-$OOD_SNAP/commits.parquet}"
QNA_PQ="${QNA_PQ:-$OOD_SNAP/qna.parquet}"

OUT_DIR="${OUT_DIR:-/home/lhotsko/RepoPeftData/evaluation/results_ood_published}"
mkdir -p "$OUT_DIR"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/bench_ood_published.json}"

# Smoke vs full configuration
if [[ "${SMOKE:-0}" == "1" ]]; then
    LIMIT_REPOS="${LIMIT_REPOS:-2}"
    MAX_FINAL="${MAX_FINAL:-40}"
    BOOTSTRAP="${BOOTSTRAP:-0}"
    TIMELINE_MODE="${TIMELINE_MODE:-percentiles}"
    OUTPUT_JSON="${OUT_DIR}/bench_ood_published_smoke.json"
else
    LIMIT_REPOS="${LIMIT_REPOS:-0}"     # 0 => all repos
    MAX_FINAL="${MAX_FINAL:-0}"         # 0 => all held-out assertions
    BOOTSTRAP="${BOOTSTRAP:-5000}"
    TIMELINE_MODE="${TIMELINE_MODE:-percentiles}"
fi

echo "Checkpoint : $CHECKPOINT"
echo "Commits    : $COMMITS_PQ"
echo "QnA        : $QNA_PQ"
echo "Output     : $OUTPUT_JSON"
echo "Smoke      : ${SMOKE:-0}  limit_repos=$LIMIT_REPOS  max_final=$MAX_FINAL  bootstrap=$BOOTSTRAP"

LIMIT_ARGS=()
if [[ "$LIMIT_REPOS" != "0" ]]; then
    LIMIT_ARGS=(--limit-repos "$LIMIT_REPOS")
fi

python hypernetwork/eval_code2lora_gru_commits_metrics.py \
    --checkpoint        "$CHECKPOINT" \
    --commits-parquet   "$COMMITS_PQ" \
    --qna-parquet       "$QNA_PQ" \
    --suites            cross_repo_ood_test \
    --assertion-mode    new \
    --final-mode        last_lora_all_assertions \
    --timeline-mode     "$TIMELINE_MODE" \
    --max-assertions-final "$MAX_FINAL" \
    --bootstrap         "$BOOTSTRAP" \
    --output-json       "$OUTPUT_JSON" \
    "${LIMIT_ARGS[@]}" \
    "$@"

echo "Done: $(date)"
echo
echo "Summary:"
python3 - "$OUTPUT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("cross_repo_ood_test") or {}
fin = s.get("final") or {}
print(f"  final  EM        = {fin.get('em_pct','?')}%")
print(f"  final  EditSim   = {fin.get('edit_similarity','?')}")
print(f"  final  CodeBLEU  = {fin.get('code_bleu','?')}")
print(f"  n_repos_scored   = {fin.get('n_repos_scored','?')}")
print(f"  n_assertions     = {fin.get('n','?')}")
PY
