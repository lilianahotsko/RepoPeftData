#!/usr/bin/env python3
"""Plot per-commit accuracy decay overlay for multiple suites.

Reads one (or several) JSON outputs of ``eval_code2lora_gru_commits_metrics.py``
and produces two overlay plots that compare every available suite:

  1. Absolute decay  : x = n_commits_after_first_kept_commit
  2. Normalized decay: x = position in [0, 1] (commit_idx / max_idx_per_repo)

Each plot shows three panels (EM %, EditSim, CodeBLEU) with one curve per suite.
Bucketed aggregates are also dumped as JSON sidecars for paper figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SUITE_ORDER = [
    "in_repo_val",
    "in_repo_test",
    "cross_repo_cr_val",
    "cross_repo_cr_test",
    "cross_repo_ood_test",
]
SUITE_LABEL = {
    "in_repo_val":        "IR-val",
    "in_repo_test":       "IR-test",
    "cross_repo_cr_val":  "CR-val",
    "cross_repo_cr_test": "CR-test",
    "cross_repo_ood_test": "OOD",
}
SUITE_COLOR = {
    "in_repo_val":        "#999999",
    "in_repo_test":       "#1f77b4",
    "cross_repo_cr_val":  "#2ca02c",
    "cross_repo_cr_test": "#d62728",
    "cross_repo_ood_test": "#9467bd",
}


def extract_rows(suite_obj: dict) -> List[dict]:
    """Pull per-commit timeline rows out of one suite dict and normalize position."""
    per_repo = (suite_obj or {}).get("per_repo") or {}
    rows: List[dict] = []
    for repo_id, rinfo in per_repo.items():
        tl = (rinfo or {}).get("per_commit_timeline") or (rinfo or {}).get("timeline") or []
        if not tl:
            continue
        idxs = [int(t.get("commit_index", t.get("commit_idx", 0))) for t in tl]
        if not idxs:
            continue
        idx_min, idx_max = min(idxs), max(idxs)
        span = max(idx_max - idx_min, 1)
        for t in tl:
            n = int(t.get("n", 0) or 0)
            if n == 0:
                continue
            ci = int(t.get("commit_index", t.get("commit_idx", 0)))
            rows.append({
                "repo_id": repo_id,
                "commit_index": ci,
                "n_after_first": ci - idx_min,
                "position": (ci - idx_min) / span,
                "em_pct": float(t.get("em_pct", t.get("exact_match", 0.0)) or 0.0),
                "edit_similarity": float(t.get("edit_similarity", 0.0) or 0.0),
                "code_bleu": float(t.get("code_bleu", 0.0) or 0.0),
                "n": n,
            })
    # Some eval outputs report EM as a fraction in [0,1] (exact_match) while
    # others use percentage (em_pct). Normalize to percentage.
    if rows and max(r["em_pct"] for r in rows) <= 1.0:
        for r in rows:
            r["em_pct"] = r["em_pct"] * 100.0
    return rows


def bucket_rows(rows: List[dict], x_field: str, buckets: List[float]) -> List[dict]:
    by: Dict[int, List[dict]] = {i: [] for i in range(len(buckets) - 1)}
    for r in rows:
        x = r[x_field]
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            if lo <= x < hi or (i == len(buckets) - 2 and x == hi):
                by[i].append(r); break
    out: List[dict] = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        rs = by[i]
        if not rs:
            out.append({"bucket": [lo, hi], "n_rows": 0, "n_assertions": 0,
                        "n_repos": 0, "em_pct": None, "edit_similarity": None,
                        "code_bleu": None})
            continue
        na = sum(r["n"] for r in rs)
        em = sum(r["em_pct"] * r["n"] for r in rs) / max(na, 1)
        es = sum(r["edit_similarity"] * r["n"] for r in rs) / max(na, 1)
        cb = sum(r["code_bleu"] * r["n"] for r in rs) / max(na, 1)
        out.append({"bucket": [lo, hi], "n_rows": len(rs), "n_assertions": na,
                    "n_repos": len({r["repo_id"] for r in rs}),
                    "em_pct": em, "edit_similarity": es, "code_bleu": cb})
    return out


def _midpoints(agg: List[dict]) -> Tuple[List[float], Dict[str, List[float]], List[int]]:
    xs, ws = [], []
    ys = {"em_pct": [], "edit_similarity": [], "code_bleu": []}
    for r in agg:
        if r["em_pct"] is None:
            continue
        lo, hi = r["bucket"]
        xs.append((lo + hi) / 2)
        ws.append(r["n_assertions"])
        for k in ys:
            ys[k].append(r[k])
    return xs, ys, ws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-result", type=Path, action="append", required=True,
                    help="Path to bench timeline JSON. Pass multiple times to merge "
                         "(e.g. one file with the four canonical suites + one with OOD).")
    ap.add_argument("--label", action="append", default=None,
                    help="Optional label per --bench-result (e.g. 'GRU', 'Static'). "
                         "When given, suite curves are prefixed with the label and "
                         "shown as separate series for static-vs-GRU comparison.")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Outputs written to <prefix>_{absolute,normalized}.{png,json}.")
    ap.add_argument("--suites", nargs="+", default=None,
                    help="Restrict to these suite names (default: all found).")
    args = ap.parse_args()

    labels = args.label or []
    if labels and len(labels) != len(args.bench_result):
        ap.error("--label count must match --bench-result count")

    # 1. Gather rows per suite, merged across input files.
    # When labels are given, we keep curves per (label, suite) so we can plot
    # static vs GRU side-by-side. Otherwise we merge all files by suite.
    rows_by_suite: Dict[str, List[dict]] = {}
    sources: Dict[str, str] = {}
    for bi, bp in enumerate(args.bench_result):
        label = labels[bi] if labels else None
        d = json.loads(bp.read_text(encoding="utf-8"))
        suite_keys = d.get("suites") or [k for k, v in d.items() if isinstance(v, dict) and (v.get("per_repo") or v.get("final"))]
        for sk in suite_keys:
            if args.suites and sk not in args.suites:
                continue
            so = d.get(sk)
            if not isinstance(so, dict):
                continue
            rs = extract_rows(so)
            if not rs:
                continue
            key = f"{label}::{sk}" if label else sk
            rows_by_suite.setdefault(key, []).extend(rs)
            sources[key] = str(bp)

    if not rows_by_suite:
        print("No timeline rows found in any provided bench file.")
        return

    # 2. Buckets shared across suites so curves align.
    max_abs = max(r["n_after_first"] for rs in rows_by_suite.values() for r in rs)
    abs_buckets = [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, max(2000, max_abs + 1)]
    pos_buckets = [round(0.1 * i, 1) for i in range(11)]

    # 3. Bucket each suite.
    out_abs: Dict[str, List[dict]] = {}
    out_pos: Dict[str, List[dict]] = {}
    print(f"\n{'Suite':<22}{'rows':>10}{'commits':>10}{'repos':>8}{'EM(all)':>10}")
    print("-" * 60)
    for sk in [s for s in SUITE_ORDER if s in rows_by_suite] + \
              [s for s in rows_by_suite if s not in SUITE_ORDER]:
        rs = rows_by_suite[sk]
        out_abs[sk] = bucket_rows(rs, "n_after_first", abs_buckets)
        out_pos[sk] = bucket_rows(rs, "position", pos_buckets)
        na = sum(r["n"] for r in rs)
        em = sum(r["em_pct"] * r["n"] for r in rs) / max(na, 1)
        print(f"{sk:<22}{na:>10,}{len(rs):>10,}"
              f"{len({r['repo_id'] for r in rs}):>8,}{em:>10.2f}")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (Path(str(args.out_prefix) + "_absolute.json")).write_text(
        json.dumps({"buckets": abs_buckets, "by_suite": out_abs, "sources": sources}, indent=2))
    (Path(str(args.out_prefix) + "_normalized.json")).write_text(
        json.dumps({"buckets": pos_buckets, "by_suite": out_pos, "sources": sources}, indent=2))

    # 4. Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib unavailable; only JSON written.")
        return

    metrics = [("em_pct", "Exact-Match %"),
               ("edit_similarity", "EditSim"),
               ("code_bleu", "CodeBLEU")]
    for kind, by_suite, xlabel, xscale in [
        ("absolute", out_abs, "n_commits_after_first_kept_commit", "symlog"),
        ("normalized", out_pos, "position in repo timeline (0=oldest, 1=newest)", "linear"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
        for ax, (mkey, mlabel) in zip(axes, metrics):
            for sk, agg in by_suite.items():
                xs, ys, ws = _midpoints(agg)
                if not xs:
                    continue
                ax.plot(xs, ys[mkey], "-o",
                        color=SUITE_COLOR.get(sk, None),
                        label=SUITE_LABEL.get(sk, sk),
                        linewidth=2, markersize=5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(mlabel)
            ax.grid(True, alpha=0.3)
            if kind == "absolute":
                ax.set_xscale(xscale)
        axes[0].legend(loc="best", fontsize=9, frameon=True)
        fig.suptitle(f"Code2LoRA-GRU per-commit decay ({kind})", y=1.02)
        out_png = Path(str(args.out_prefix) + f"_{kind}.png")
        out_pdf = Path(str(args.out_prefix) + f"_{kind}.pdf")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_png}")
        print(f"  -> {out_pdf}")


if __name__ == "__main__":
    main()
