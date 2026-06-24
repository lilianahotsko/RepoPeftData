#!/usr/bin/env python3
"""Merge sharded ``run_code2lora_gru_v2_eval.py`` outputs into one summary.

Concatenates the per-shard ``raw_samples`` (exact_match / edit_similarity /
code_bleu) and recomputes the headline metrics + bootstrap 95% CIs over the
full pooled sample, so a sharded array job yields the same number a single
run would.

Usage::

    python scripts/merge_gru_v2_shards.py \
        --dir evaluation/results_indist/4bit/cr_test --suite cr_test \
        --bootstrap 5000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "hypernetwork"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.metrics import aggregate_metrics_with_ci  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with shard JSONs")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--out", default=None,
                    help="output path (default: <dir>/gru_v2_<suite>.json)")
    args = ap.parse_args()

    pat = os.path.join(args.dir, f"gru_v2_{args.suite}_shard*of*.json")
    shard_files = sorted(glob.glob(pat))
    if not shard_files:
        raise SystemExit(f"no shard files matched: {pat}")

    em, ed, cb = [], [], []
    n_commits = 0
    repos: set = set()
    n_total_groups = 0
    finalized_all = True
    for f in shard_files:
        d = json.load(open(f))
        rs = d.get("raw_samples") or {}
        em += rs.get("exact_match", [])
        ed += rs.get("edit_similarity", [])
        cb += rs.get("code_bleu", [])
        n_commits += len((d.get("per_commit") or []))
        for rec in (d.get("per_commit") or []):
            if rec.get("repo_id") is not None:
                repos.add(rec["repo_id"])
        n_total_groups += int(d.get("n_total_groups") or 0)
        finalized_all = finalized_all and bool(d.get("finalized"))

    n = len(em)
    if n == 0:
        raise SystemExit("merged sample is empty")

    metric_dicts = [
        {"exact_match": bool(a), "edit_similarity": b, "code_bleu": c}
        for a, b, c in zip(em, ed, cb)
    ]
    agg = aggregate_metrics_with_ci(metric_dicts, n_resamples=int(args.bootstrap))

    summary = {"n_qnas": n, "n": n, "suite": args.suite,
               "n_scored_commits": n_commits, "n_repos": len(repos),
               "n_shards": len(shard_files), "all_shards_finalized": finalized_all}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            summary[k] = float(v["mean"])
            summary[f"{k}_ci"] = [float(v.get("low", 0.0)), float(v.get("high", 0.0))]
        else:
            summary[k] = v

    out = args.out or os.path.join(args.dir, f"gru_v2_{args.suite}.json")
    json.dump({"finalized": finalized_all, "merged_from": shard_files,
               "summary": summary}, open(out, "w"), indent=2)

    print(f"[merge] {args.suite}: {len(shard_files)} shards, n={n} qnas, "
          f"{len(repos)} repos, finalized={finalized_all}")
    print(f"  exact_match    = {summary.get('exact_match'):.4f}  "
          f"CI={summary.get('exact_match_ci')}")
    print(f"  edit_similarity= {summary.get('edit_similarity'):.4f}")
    print(f"  code_bleu      = {summary.get('code_bleu'):.4f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
