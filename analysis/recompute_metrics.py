#!/usr/bin/env python3
"""
Recompute metrics on existing result files using the fixed postprocess_prediction.
Reads entries from $SCRATCH/BASELINES/*.json and prints updated metrics.
No GPU needed -- just reads stored predictions and re-evaluates.

Usage:
    python analysis/recompute_metrics.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments


def recompute(result_path: Path) -> dict | None:
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    entries = data.get("entries", [])
    if not entries:
        return None

    em_count = 0
    edit_sum = 0.0
    bleu_sum = 0.0
    n = len(entries)

    for e in entries:
        pred_raw = e.get("got_raw", e.get("got", ""))
        target = e.get("expected", "")

        pred_clean = postprocess_prediction(pred_raw, target)
        target_clean = strip_comments(target)

        em = exact_match(pred_clean, target_clean)
        es = edit_similarity(pred_clean, target_clean)
        bl = code_bleu_score(pred_clean, target_clean)

        em_count += int(em)
        edit_sum += es
        bleu_sum += bl

    return {
        "file": result_path.name,
        "n": n,
        "exact_match_pct": 100.0 * em_count / n,
        "edit_similarity": edit_sum / n,
        "code_bleu": bleu_sum / n,
        "old_em": data.get("exact_match_pct", 0),
        "old_es": data.get("edit_similarity", 0),
        "old_cb": data.get("code_bleu", 0),
    }


def main():
    results_dir = Path(os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))) / "BASELINES"
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    files = sorted(results_dir.glob("*.json"))
    print(f"Found {len(files)} result files in {results_dir}\n")

    header = f"{'File':<45} | {'Old EM':>7} | {'New EM':>7} | {'Δ EM':>6} | {'Old ES':>7} | {'New ES':>7} | {'Δ ES':>6}"
    print(header)
    print("-" * len(header))

    for f in files:
        r = recompute(f)
        if r is None:
            continue
        d_em = r["exact_match_pct"] - r["old_em"]
        d_es = r["edit_similarity"] - r["old_es"]
        print(f"{r['file']:<45} | {r['old_em']:>6.2f}% | {r['exact_match_pct']:>6.2f}% | {d_em:>+5.2f} | {r['old_es']:>7.4f} | {r['edit_similarity']:>7.4f} | {d_es:>+6.4f}")


if __name__ == "__main__":
    main()
