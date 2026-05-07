#!/usr/bin/env python3
"""Quality-check the smart-capped GRU-commit QnA parquet snapshot.

Compares the smart-capped train.parquet against the original commit_parquet_hf
along several axes:

* total rows / repos / commits before vs after (in_repo='train' subset);
* per-commit and per-(commit, file) QnA distribution;
* assertion-type histogram parity;
* test-file diversity (same files reached pre/post cap?);
* repo coverage (no repo dropped to zero);
* random sample of N (prefix, target) pairs for human eyeballing.

Outputs:
    analysis/output/smart_cap_qna_quality/<tag>_summary.json
    analysis/output/smart_cap_qna_quality/<tag>_samples.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    Path(os.environ.get("SCRATCH", str(Path.home() / "scratch")))
    / "REPO_DATASET"
)


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def dist(values: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "p10": percentile(vals, 10),
        "p25": percentile(vals, 25),
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }


def collect_light_stats(
    qna_path: Path,
    *,
    in_repo_split: str = "train",
    batch_size: int = 16384,
) -> Dict[str, Any]:
    """Light-column scan: per-commit / per-(commit, file) counts and
    type histograms. Memory <1 GB even on the 15 GB train.parquet."""
    cols = [
        "repo_id", "commit_index", "in_repo_split",
        "test_file", "test_function", "assertion_type",
    ]
    ds = pads.dataset(str(qna_path), format="parquet")
    flt = pc.equal(pc.field("in_repo_split"), in_repo_split)
    scanner = ds.scanner(columns=cols, filter=flt, batch_size=batch_size)

    per_commit_counts: Counter[Tuple[str, int]] = Counter()
    per_commit_files: Dict[Tuple[str, int], set] = defaultdict(set)
    per_commit_funcs: Dict[Tuple[str, int], set] = defaultdict(set)
    per_file_counts: Counter[Tuple[str, int, str]] = Counter()
    type_hist: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    n_total = 0

    t0 = time.time()
    for bi, rb in enumerate(scanner.to_batches()):
        if rb.num_rows == 0:
            continue
        n_total += rb.num_rows
        repo = rb.column("repo_id").to_pylist()
        ci = rb.column("commit_index").to_pylist()
        tf = rb.column("test_file").to_pylist()
        tfu = rb.column("test_function").to_pylist()
        at = rb.column("assertion_type").to_pylist()
        for j in range(rb.num_rows):
            key = (repo[j], int(ci[j]))
            per_commit_counts[key] += 1
            repo_counts[repo[j]] += 1
            if tf[j]:
                per_commit_files[key].add(tf[j])
                per_file_counts[(repo[j], int(ci[j]), tf[j])] += 1
            if tfu[j]:
                per_commit_funcs[key].add(tfu[j])
            if at[j]:
                type_hist[at[j]] += 1
        if bi % 50 == 0:
            print(
                f"  scanned {n_total:,} rows in {time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"  scan done: {n_total:,} rows in {time.time() - t0:.0f}s", flush=True)

    return {
        "qna_path": str(qna_path),
        "in_repo_split": in_repo_split,
        "n_rows": n_total,
        "n_commits": len(per_commit_counts),
        "n_repos": len(repo_counts),
        "qna_per_commit": dist(per_commit_counts.values()),
        "files_per_commit": dist(len(s) for s in per_commit_files.values()),
        "funcs_per_commit": dist(len(s) for s in per_commit_funcs.values()),
        "rows_per_file_per_commit": dist(per_file_counts.values()),
        "qna_per_repo": dist(repo_counts.values()),
        "top_assertion_types": dict(type_hist.most_common(25)),
        "_repos_set": repo_counts,
        "_per_commit_keys": set(per_commit_counts.keys()),
        "_per_file_keys": set(per_file_counts.keys()),
        "_per_commit_files": per_commit_files,
    }


def diff_coverage(orig: Dict[str, Any], capped: Dict[str, Any]) -> Dict[str, Any]:
    orig_repos = set(orig["_repos_set"].keys())
    capped_repos = set(capped["_repos_set"].keys())
    orig_commits = orig["_per_commit_keys"]
    capped_commits = capped["_per_commit_keys"]
    orig_per_file = orig["_per_file_keys"]
    capped_per_file = capped["_per_file_keys"]
    # Per-commit: how many *files* did we drop entirely?
    dropped_files_per_commit = []
    for k, files in orig["_per_commit_files"].items():
        if k not in capped["_per_commit_files"]:
            dropped_files_per_commit.append(len(files))
            continue
        kept_files = capped["_per_commit_files"][k]
        dropped_files_per_commit.append(len(files - kept_files))
    return {
        "repos_dropped": sorted(orig_repos - capped_repos),
        "n_repos_in_orig": len(orig_repos),
        "n_repos_in_capped": len(capped_repos),
        "n_commits_in_orig": len(orig_commits),
        "n_commits_in_capped": len(capped_commits),
        "n_commits_dropped_entirely": len(orig_commits - capped_commits),
        "n_unique_(commit, file)_in_orig": len(orig_per_file),
        "n_unique_(commit, file)_in_capped": len(capped_per_file),
        "fraction_(commit, file)_kept": (
            len(capped_per_file) / max(len(orig_per_file), 1)
        ),
        "files_dropped_per_commit": dist(dropped_files_per_commit),
    }


def sample_capped_qnas(
    qna_path: Path,
    *,
    in_repo_split: str = "train",
    n_samples: int = 30,
    seed: int = 1234,
) -> List[Dict[str, Any]]:
    """Reservoir-sample N rows from the capped parquet for human eyeballing."""
    cols = [
        "repo_id", "commit_index", "test_file", "test_function",
        "assertion_type", "assertion_event_type", "prefix", "target",
    ]
    ds = pads.dataset(str(qna_path), format="parquet")
    flt = pc.equal(pc.field("in_repo_split"), in_repo_split)
    scanner = ds.scanner(columns=cols, filter=flt, batch_size=4096)

    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    n_seen = 0
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        for j in range(rb.num_rows):
            n_seen += 1
            if len(reservoir) < n_samples:
                reservoir.append(
                    {col: rb.column(col)[j].as_py() for col in cols}
                )
            else:
                idx = rng.randrange(n_seen)
                if idx < n_samples:
                    reservoir[idx] = {
                        col: rb.column(col)[j].as_py() for col in cols
                    }
    return reservoir


def render_samples(samples: List[Dict[str, Any]], max_prefix_chars: int = 1200) -> str:
    out: List[str] = []
    for i, s in enumerate(samples, 1):
        out.append(f"--- sample {i} -----------------------------------")
        out.append(f"repo:           {s.get('repo_id')}")
        out.append(f"commit_index:   {s.get('commit_index')}")
        out.append(f"file:           {s.get('test_file')}")
        out.append(f"function:       {s.get('test_function')}")
        out.append(f"assertion_type: {s.get('assertion_type')}")
        out.append(f"event_type:     {s.get('assertion_event_type')}")
        pre = s.get("prefix") or ""
        if len(pre) > max_prefix_chars:
            head = pre[: max_prefix_chars // 2]
            tail = pre[-max_prefix_chars // 2 :]
            pre_render = (
                f"{head}\n... [{len(pre) - max_prefix_chars} chars elided] ...\n{tail}"
            )
        else:
            pre_render = pre
        out.append("PREFIX (last lines):")
        out.append("\n".join(pre_render.splitlines()[-30:]))
        out.append("TARGET:")
        out.append(s.get("target") or "")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--orig-dir", type=Path, default=DEFAULT_BASE / "commit_parquet_hf",
    )
    ap.add_argument(
        "--capped-dir", type=Path, default=DEFAULT_BASE / "commit_parquet_hf_smartcap",
    )
    ap.add_argument(
        "--cross-split", default="train",
        help="Cross-repo split file to compare (train/cr_val/cr_test).",
    )
    ap.add_argument(
        "--in-repo-split", default="train",
        help="In-repo split (only train rows are capped).",
    )
    ap.add_argument(
        "--n-samples", type=int, default=30,
        help="Random sample size for human eyeballing.",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "analysis" / "output" / "smart_cap_qna_quality",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig_path = args.orig_dir / "qna" / f"{args.cross_split}.parquet"
    capped_path = args.capped_dir / "qna" / f"{args.cross_split}.parquet"
    if not orig_path.exists():
        raise SystemExit(f"missing {orig_path}")
    if not capped_path.exists():
        raise SystemExit(f"missing {capped_path}")

    print(f"=== Original {orig_path} ===", flush=True)
    orig_stats = collect_light_stats(
        orig_path, in_repo_split=args.in_repo_split,
    )
    print(f"=== Capped   {capped_path} ===", flush=True)
    capped_stats = collect_light_stats(
        capped_path, in_repo_split=args.in_repo_split,
    )

    coverage = diff_coverage(orig_stats, capped_stats)

    # Strip private debug fields before serializing.
    def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    summary = {
        "cross_split": args.cross_split,
        "in_repo_split": args.in_repo_split,
        "orig": _clean(orig_stats),
        "capped": _clean(capped_stats),
        "ratio_kept": (
            capped_stats["n_rows"] / max(orig_stats["n_rows"], 1)
        ),
        "coverage": coverage,
    }
    out_json = (
        args.out_dir / f"{args.cross_split}_in={args.in_repo_split}_summary.json"
    )
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}", flush=True)

    # Sample QnAs from the capped set for human eyeballing.
    print("\nSampling capped QnAs for human review...", flush=True)
    samples = sample_capped_qnas(
        capped_path, in_repo_split=args.in_repo_split,
        n_samples=args.n_samples,
    )
    rendered = render_samples(samples)
    out_txt = (
        args.out_dir / f"{args.cross_split}_in={args.in_repo_split}_samples.txt"
    )
    out_txt.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_txt}", flush=True)

    # Print a compact one-liner so reviewers see the headline numbers.
    print(
        "\n=== Headline ===\n"
        f"  rows kept:        {capped_stats['n_rows']:,} / {orig_stats['n_rows']:,} "
        f"({summary['ratio_kept']*100:.1f}%)\n"
        f"  repos:            {capped_stats['n_repos']} / {orig_stats['n_repos']} "
        f"({coverage['repos_dropped'] or 'no repo dropped'})\n"
        f"  commits:          {capped_stats['n_commits']} / {orig_stats['n_commits']} "
        f"(dropped {coverage['n_commits_dropped_entirely']})\n"
        f"  unique (commit, file) kept: "
        f"{coverage['n_unique_(commit, file)_in_capped']} / "
        f"{coverage['n_unique_(commit, file)_in_orig']} "
        f"({coverage['fraction_(commit, file)_kept']*100:.1f}%)\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
