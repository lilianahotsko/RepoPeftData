#!/usr/bin/env python3
"""Print a paper-ready summary of GRU<sub>commit</sub> smart-cap bench results.

Reads ``bench_<method>_<suite>.json`` files written by
``evaluation/run_repopeft_bench.py`` and prints one row per suite with EM /
EditSim / CodeBLEU and 95% bootstrap CIs. Missing suites are reported as
"pending" so it's safe to run while SLURM jobs are still in flight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


SUITE_FILENAMES = {
    "CR-test": "bench_gru_commit_cr_test.json",
    "IR-test": "bench_gru_commit_ir_test.json",
    "OOD-test": "bench_gru_commit_ood_test.json",
}


def fmt_ci(mean: float, ci: Any, pct: bool = False) -> str:
    if isinstance(ci, dict):
        lo = ci.get("low", mean)
        hi = ci.get("high", mean)
    elif ci is not None:
        lo, hi = ci[0], ci[1]
    else:
        lo, hi = mean, mean
    if pct:
        return f"{mean*100:5.2f} [{lo*100:5.2f}, {hi*100:5.2f}]"
    return f"{mean:5.4f} [{lo:5.4f}, {hi:5.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, required=True)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for suite, fname in SUITE_FILENAMES.items():
        p = args.bench_dir / fname
        if not p.exists():
            rows.append({"suite": suite, "status": "pending", "path": str(p)})
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"suite": suite, "status": f"error: {e}", "path": str(p)})
            continue
        rows.append({
            "suite": suite,
            "status": "done",
            "n": data.get("n", 0),
            "n_repos_scored": data.get("n_repos_scored", 0),
            "n_repos_skipped": data.get("n_repos_skipped", 0),
            "em_pct": data.get("exact_match_pct"),
            "edit_sim": data.get("edit_similarity"),
            "code_bleu": data.get("code_bleu"),
            "ci95": data.get("ci95", {}),
        })

    print(f"\nBench dir: {args.bench_dir}\n")
    hdr = (f"{'Suite':<10} {'EM %':>22} {'EditSim':>26} {'CodeBLEU':>26} "
           f"{'n_pairs':>10} {'repos':>8} {'skip':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] != "done":
            print(f"{r['suite']:<10} [{r['status']}]  {r.get('path', '')}")
            continue
        em = r["em_pct"] / 100.0 if r["em_pct"] is not None else 0.0
        ci = r["ci95"] or {}
        print(f"{r['suite']:<10} "
              f"{fmt_ci(em, ci.get('exact_match'), pct=True):>22} "
              f"{fmt_ci(r['edit_sim'] or 0.0, ci.get('edit_similarity')):>26} "
              f"{fmt_ci(r['code_bleu'] or 0.0, ci.get('code_bleu')):>26} "
              f"{r['n']:>10,} {r['n_repos_scored']:>8} {r['n_repos_skipped']:>5}")
    print()


if __name__ == "__main__":
    main()
