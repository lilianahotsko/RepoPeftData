#!/usr/bin/env python3
"""Aggregate per-(repo, commit) metrics into a decay curve.

Reads the ``per_repo`` dict from a ``run_repopeft_bench.py`` output JSON whose
keys are ``<repo_id>@@<commit_sha>`` (produced by
``build_static_commit_bench_jsons.py`` / the static-commit splits). Cross-
references the bench JSON's ``metadata.n_commits_after_anchor`` to bucket
each (repo, commit) item, then prints EM% / EditSim / CodeBLEU per bucket so
you can see how accuracy decays as you drift further from the training cut.

Optionally compares against a second bench JSON (e.g. the GRU_commit eval)
side-by-side, so the static-vs-streaming trend is visible in one table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_BUCKETS = [
    (-1, "<=anchor (warmup)"),
    (0, "0"),
    (1, "1..4"),
    (5, "5..19"),
    (20, "20..49"),
    (50, "50..99"),
    (100, "100..499"),
    (500, "500..1999"),
    (2000, ">=2000"),
]


def _bucket_label(n: int, edges: List[Tuple[int, str]]) -> str:
    """Return label of the right-most bucket whose start <= n."""
    label = edges[0][1]
    for start, lab in edges:
        if n >= start:
            label = lab
    return label


def _load_bench_meta(bench_json: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``{repo_key: metadata}`` for every entry in a static-commit bench."""
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    return {k: (v.get("metadata") or {}) for k, v in data.get("repositories", {}).items()}


def aggregate(per_repo: Dict[str, Dict[str, Any]],
              meta: Dict[str, Dict[str, Any]],
              buckets: List[Tuple[int, str]]
              ) -> Dict[str, Dict[str, Any]]:
    grp: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"em": 0, "edit_sum": 0.0, "bleu_sum": 0.0,
                 "n": 0, "n_commits": 0, "n_repos": set()})
    for rkey, m in per_repo.items():
        bm = meta.get(rkey)
        if bm is None:
            continue
        n_after = int(bm.get("n_commits_after_anchor", 0))
        label = _bucket_label(n_after, buckets)
        g = grp[label]
        g["em"] += float(m.get("exact_match", 0.0)) * float(m.get("n", 0))
        g["edit_sum"] += float(m.get("edit_similarity", 0.0)) * float(m.get("n", 0))
        g["bleu_sum"] += float(m.get("code_bleu", 0.0)) * float(m.get("n", 0))
        g["n"] += int(m.get("n", 0))
        g["n_commits"] += 1
        g["n_repos"].add(bm.get("repo_id", ""))
    # Finalize
    out: Dict[str, Dict[str, Any]] = {}
    for label, g in grp.items():
        n = max(g["n"], 1)
        out[label] = {
            "em_pct": 100.0 * g["em"] / n,
            "edit_similarity": g["edit_sum"] / n,
            "code_bleu": g["bleu_sum"] / n,
            "n_pairs": g["n"],
            "n_commits": g["n_commits"],
            "n_repos": len(g["n_repos"]),
        }
    return out


def print_table(label: str, agg: Dict[str, Dict[str, Any]],
                buckets: List[Tuple[int, str]]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'bucket':<22} {'EM %':>7} {'EditSim':>9} {'CodeBLEU':>9} "
          f"{'n_pairs':>10} {'n_commits':>10} {'n_repos':>8}")
    print("-" * 80)
    seen = set()
    for _, lab in buckets:
        if lab in agg and lab not in seen:
            seen.add(lab)
            r = agg[lab]
            print(f"{lab:<22} {r['em_pct']:>7.2f} {r['edit_similarity']:>9.4f} "
                  f"{r['code_bleu']:>9.4f} {r['n_pairs']:>10,} "
                  f"{r['n_commits']:>10,} {r['n_repos']:>8,}")
    # Any extra (unmapped) labels
    for lab in agg:
        if lab not in seen:
            r = agg[lab]
            print(f"{lab:<22} {r['em_pct']:>7.2f} {r['edit_similarity']:>9.4f} "
                  f"{r['code_bleu']:>9.4f} {r['n_pairs']:>10,} "
                  f"{r['n_commits']:>10,} {r['n_repos']:>8,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-result", type=Path, required=True,
                    help="Output JSON from run_repopeft_bench.py (per_repo block).")
    ap.add_argument("--bench-json", type=Path, required=True,
                    help="The static-commit bench JSON used to produce the result "
                         "(needed for n_commits_after_anchor metadata).")
    ap.add_argument("--label", type=str, default="static-commit")
    ap.add_argument("--compare-result", type=Path, default=None,
                    help="Second bench result to compare (same bench-json keys).")
    ap.add_argument("--compare-label", type=str, default="other")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    res = json.loads(args.bench_result.read_text(encoding="utf-8"))
    per_repo = res.get("per_repo") or {}
    meta = _load_bench_meta(args.bench_json)
    buckets = DEFAULT_BUCKETS

    agg = aggregate(per_repo, meta, buckets)
    print_table(args.label, agg, buckets)

    compare = None
    if args.compare_result is not None:
        res2 = json.loads(args.compare_result.read_text(encoding="utf-8"))
        per_repo2 = res2.get("per_repo") or {}
        compare = aggregate(per_repo2, meta, buckets)
        print_table(args.compare_label, compare, buckets)

    if args.output_json is not None:
        out = {args.label: agg}
        if compare is not None:
            out[args.compare_label] = compare
        args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
