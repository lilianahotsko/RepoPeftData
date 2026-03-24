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


def recompute(result_path: Path, save: bool = False) -> dict | None:
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

        # Update per-entry metrics
        e["exact_match"] = em
        e["edit_similarity"] = es
        e["code_bleu"] = bl
        e["got"] = pred_clean

    result = {
        "file": result_path.name,
        "n": n,
        "exact_match_pct": 100.0 * em_count / n,
        "edit_similarity": edit_sum / n,
        "code_bleu": bleu_sum / n,
        "old_em": data.get("exact_match_pct", 0),
        "old_es": data.get("edit_similarity", 0),
        "old_cb": data.get("code_bleu", 0),
    }

    if save:
        data["exact_match_pct"] = result["exact_match_pct"]
        data["exact_match_count"] = em_count
        data["edit_similarity"] = result["edit_similarity"]
        data["code_bleu"] = result["code_bleu"]
        data["entries"] = entries
        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Write updated metrics back to JSON files")
    args = ap.parse_args()

    results_dir = Path(os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))) / "BASELINES"
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    files = sorted(results_dir.glob("*.json"))
    print(f"Found {len(files)} result files in {results_dir}")
    if args.save:
        print("  ** --save enabled: JSON files will be updated in-place **")
    print()

    header = f"{'File':<45} | {'Old CB':>7} | {'New CB':>7} | {'Δ CB':>7}"
    print(header)
    print("-" * len(header))

    for f in files:
        r = recompute(f, save=args.save)
        if r is None:
            continue
        d_cb = r["code_bleu"] - r["old_cb"]
        print(f"{r['file']:<45} | {r['old_cb']:>7.4f} | {r['code_bleu']:>7.4f} | {d_cb:>+7.4f}")


if __name__ == "__main__":
    main()
