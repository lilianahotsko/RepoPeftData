#!/usr/bin/env python3
"""Plot per-commit accuracy decay over repo history.

Reads the output JSON from ``eval_code2lora_gru_commits_metrics.py`` (which
includes a ``per_commit_timeline`` per repo when ``--timeline-mode=all``) and
produces two plots:

  1. **Absolute** decay: x = ``n_commits_after_first_kept_commit``
     (i.e. the in-repo commit index). Aggregated across repos by bucket.
  2. **Normalized** decay: x = ``position`` ∈ [0, 1] (= commit_idx / max_idx_per_repo)
     so short and long repos are comparable.

Also writes the bucketed data as a JSON sidecar so the LaTeX / paper figure
script can ingest it directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _percentiles(arr: List[float], qs=(10, 50, 90)) -> Dict[str, float]:
    if not arr:
        return {f"p{q}": 0.0 for q in qs}
    a = np.asarray(arr, dtype=np.float64)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def aggregate(timeline_rows: List[dict],
              x_field: str, x_buckets: List[float]
              ) -> List[dict]:
    """Bucket timeline rows by *x_field* and compute mean EM per bucket."""
    by_bucket: Dict[int, List[dict]] = {i: [] for i in range(len(x_buckets) - 1)}
    for row in timeline_rows:
        x = row[x_field]
        for i in range(len(x_buckets) - 1):
            lo, hi = x_buckets[i], x_buckets[i + 1]
            if lo <= x < hi or (i == len(x_buckets) - 2 and x == hi):
                by_bucket[i].append(row); break
    out: List[dict] = []
    for i in range(len(x_buckets) - 1):
        lo, hi = x_buckets[i], x_buckets[i + 1]
        rows = by_bucket[i]
        if not rows:
            out.append({"bucket": (lo, hi), "n_rows": 0,
                        "em_pct": None, "edit_similarity": None,
                        "code_bleu": None, "n_assertions": 0,
                        "n_repos": 0})
            continue
        n_assert = sum(int(r.get("n", 0)) for r in rows)
        em_w = sum(float(r.get("em_pct", 0.0)) * int(r.get("n", 0)) for r in rows)
        edit_w = sum(float(r.get("edit_similarity", 0.0)) * int(r.get("n", 0)) for r in rows)
        bleu_w = sum(float(r.get("code_bleu", 0.0)) * int(r.get("n", 0)) for r in rows)
        out.append({
            "bucket": (lo, hi),
            "n_rows": len(rows),
            "em_pct": em_w / max(n_assert, 1),
            "edit_similarity": edit_w / max(n_assert, 1),
            "code_bleu": bleu_w / max(n_assert, 1),
            "n_assertions": n_assert,
            "n_repos": len({r["repo_id"] for r in rows}),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-result", type=Path, required=True,
                    help="Output JSON from eval_code2lora_gru_commits_metrics.py.")
    ap.add_argument("--suite", default="cross_repo_ood_test",
                    help="Which suite key to read from the bench result.")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Plots written to <prefix>_absolute.{png,json} and <prefix>_normalized.{png,json}.")
    args = ap.parse_args()

    res = json.loads(args.bench_result.read_text(encoding="utf-8"))
    suite = res.get(args.suite)
    if not suite:
        print(f"No suite '{args.suite}' in {args.bench_result}; available: {[k for k in res if isinstance(res[k], dict)]}")
        return

    per_repo = suite.get("per_repo") or {}
    rows: List[dict] = []
    for repo_id, rinfo in per_repo.items():
        timeline = rinfo.get("per_commit_timeline") or rinfo.get("timeline") or []
        if not timeline:
            continue
        # Determine repo's commit-index range so we can normalize.
        idxs = [int(t.get("commit_index", t.get("commit_idx", 0))) for t in timeline]
        if not idxs:
            continue
        idx_min, idx_max = min(idxs), max(idxs)
        span = max(idx_max - idx_min, 1)
        for t in timeline:
            ci = int(t.get("commit_index", t.get("commit_idx", 0)))
            if int(t.get("n", 0)) == 0:
                continue  # skip commits with 0 scored assertions
            rows.append({
                "repo_id": repo_id,
                "commit_index": ci,
                "n_after_first": ci - idx_min,
                "position": (ci - idx_min) / span,
                "em_pct": float(t.get("em_pct", t.get("em", 0.0)) or 0.0),
                "edit_similarity": float(t.get("edit_similarity", 0.0) or 0.0),
                "code_bleu": float(t.get("code_bleu", 0.0) or 0.0),
                "n": int(t.get("n", 0)),
            })

    if not rows:
        print(f"No per-commit timeline rows found in suite '{args.suite}'.")
        return

    print(f"Loaded {len(rows):,} per-commit rows across {len({r['repo_id'] for r in rows})} repos")

    # --- 1. Absolute decay (commits after first kept) ---
    max_abs = max(r["n_after_first"] for r in rows)
    abs_buckets = [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, max(2000, max_abs + 1)]
    abs_agg = aggregate(rows, "n_after_first", abs_buckets)

    # --- 2. Normalized decay (relative position 0..1) ---
    pos_buckets = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
    pos_agg = aggregate(rows, "position", pos_buckets)

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (Path(str(out_prefix) + "_absolute.json")).write_text(
        json.dumps(abs_agg, indent=2), encoding="utf-8")
    (Path(str(out_prefix) + "_normalized.json")).write_text(
        json.dumps(pos_agg, indent=2), encoding="utf-8")

    # Pretty-print
    for name, agg in [("ABSOLUTE  (n_commits_after_first)", abs_agg),
                       ("NORMALIZED (position in [0,1])", pos_agg)]:
        print(f"\n=== {name} ===")
        print(f"{'bucket':<22} {'EM %':>7} {'EditSim':>9} {'CodeBLEU':>9} "
              f"{'n_assert':>10} {'n_rows':>8} {'n_repos':>8}")
        print("-" * 80)
        for r in agg:
            lo, hi = r["bucket"]
            blab = f"[{lo}, {hi})"
            if r["em_pct"] is None:
                print(f"{blab:<22} {'-':>7} {'-':>9} {'-':>9} {0:>10} {0:>8} {0:>8}"); continue
            print(f"{blab:<22} {r['em_pct']:>7.2f} {r['edit_similarity']:>9.4f} "
                  f"{r['code_bleu']:>9.4f} {r['n_assertions']:>10,} "
                  f"{r['n_rows']:>8,} {r['n_repos']:>8,}")

    # --- Optional matplotlib plots (skip silently if matplotlib missing) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available; skipping PNG plots)")
        return

    for name, agg, xlabel in [
        ("absolute", abs_agg, "n_commits_after_first_kept_commit"),
        ("normalized", pos_agg, "relative position in repo timeline"),
    ]:
        xs, ys, ws = [], [], []
        for r in agg:
            if r["em_pct"] is None: continue
            lo, hi = r["bucket"]
            xs.append((lo + hi) / 2)
            ys.append(r["em_pct"])
            ws.append(r["n_assertions"])
        if not xs: continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, "-o", color="#1f77b4")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Exact-Match %")
        ax.set_title(f"GRU per-commit EM decay ({args.suite}, {name})")
        ax.grid(True, alpha=0.3)
        out_png = Path(str(out_prefix) + f"_{name}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"  -> {out_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
