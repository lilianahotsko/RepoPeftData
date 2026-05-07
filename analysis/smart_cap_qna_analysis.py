#!/usr/bin/env python3
"""Distribution analysis for the GRU-commit "smart cap" of QnAs per commit.

Streams the QnA parquet to compute, per (cross_repo_split, in_repo_split):

* per-(repo, commit) counts of QnA rows, distinct ``test_file``s,
  distinct ``test_function``s, and distinct ``assertion_type``s;
* per-(repo, commit, test_file) QnA counts;
* prefix / target character lengths;
* exact (prefix, target) duplicate rates;
* simulated retained-QnA counts under several cap strategies:

  - random-K (the existing ``--max-assertions-per-commit``);
  - per-file-K (cap K rows per (commit, test_file));
  - per-file-K then random-M (per-file then per-commit cap);
  - per-event-mix (added, modified, parametrize / pytest.raises rebalanced);

Outputs:
    analysis/output/smart_cap_qna/<split>_summary.json
    analysis/output/smart_cap_qna/<split>_per_commit.parquet  (optional)
    analysis/output/smart_cap_qna/cap_simulation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
DEFAULT_DATA_DIR = (
    Path(os.environ.get("SCRATCH", str(Path.home() / "scratch")))
    / "REPO_DATASET"
    / "commit_parquet_hf"
)
SPLIT_FILES = {
    "train": "train.parquet",
    "cr_val": "cr_val.parquet",
    "cr_test": "cr_test.parquet",
}
QNA_COLUMNS_LIGHT = [
    "repo_id",
    "commit_index",
    "in_repo_split",
    "test_file",
    "test_function",
    "assertion_type",
    "assertion_event_type",
]
QNA_COLUMNS_FULL = QNA_COLUMNS_LIGHT + ["prefix", "target"]


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


def short_hash(prefix: str, target: str) -> int:
    h = hashlib.blake2b(
        f"{prefix}\x00{target}".encode("utf-8", errors="replace"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big", signed=False)


def stream_split(
    parquet_path: Path,
    columns: Sequence[str],
    keep_in_repo_splits: Optional[Sequence[str]] = ("train",),
    batch_size: int = 65536,
) -> Iterable["pa.RecordBatch"]:
    ds = pads.dataset(str(parquet_path), format="parquet")
    filt = None
    if keep_in_repo_splits:
        filt = pc.is_in(
            pc.field("in_repo_split"), pa.array(list(keep_in_repo_splits))
        )
    scanner = ds.scanner(
        columns=list(columns), filter=filt, batch_size=batch_size,
    )
    yield from scanner.to_batches()


def collect_per_commit_stats(
    parquet_path: Path,
    keep_in_repo_splits: Optional[Sequence[str]],
    *,
    batch_size: int = 65536,
    sample_lengths: int = 200_000,
    sample_dedup_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Stream the QnA parquet and compute per-commit / per-file stats.

    Pass 1 (light columns): per-commit / per-file counts, file/func/type
    diversity. Memory budget: O(n_commits + n_files_total).

    Pass 2 (heavy columns, sampled): prefix/target length distribution and
    a sub-sampled exact-duplicate rate (we *do not* try to dedup the full
    2.7M-row corpus in RAM).
    """
    # ── Pass 1: light columns (no prefix/target) ─────────────────────────
    per_commit_counts: Counter[Tuple[str, int]] = Counter()
    per_commit_files: Dict[Tuple[str, int], set] = defaultdict(set)
    per_commit_funcs: Dict[Tuple[str, int], set] = defaultdict(set)
    per_commit_types: Dict[Tuple[str, int], set] = defaultdict(set)
    per_file_counts: Counter[Tuple[str, int, str]] = Counter()
    per_commit_event_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    n_total = 0
    t0 = time.time()
    for bi, rb in enumerate(stream_split(
        parquet_path, QNA_COLUMNS_LIGHT,
        keep_in_repo_splits=keep_in_repo_splits,
        batch_size=batch_size,
    )):
        if rb.num_rows == 0:
            continue
        n_total += rb.num_rows
        repo = rb.column("repo_id").to_pylist()
        ci = rb.column("commit_index").to_pylist()
        tf = rb.column("test_file").to_pylist()
        tfu = rb.column("test_function").to_pylist()
        at = rb.column("assertion_type").to_pylist()
        ev = rb.column("assertion_event_type").to_pylist()
        for j in range(rb.num_rows):
            key = (repo[j], int(ci[j]))
            per_commit_counts[key] += 1
            if tf[j]:
                per_commit_files[key].add(tf[j])
                per_file_counts[(repo[j], int(ci[j]), tf[j])] += 1
            if tfu[j]:
                per_commit_funcs[key].add(tfu[j])
            if at[j]:
                per_commit_types[key].add(at[j])
            if ev[j]:
                per_commit_event_counts[key][ev[j]] += 1
        if bi % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  [pass1] {n_total:,} rows scanned in {elapsed:.0f}s",
                flush=True,
            )

    t1 = time.time()
    print(f"  [pass1] done: {n_total:,} rows in {t1 - t0:.0f}s", flush=True)

    # ── Pass 2: heavy columns (prefix/target). Aggregates length stats
    # directly inside Arrow (no Python string materialization), and stops
    # as soon as we've collected enough length samples + dedup samples.
    import random

    rng = random.Random(0)
    target_n_commits = max(
        1, int(len(per_commit_counts) * sample_dedup_fraction),
    )
    sample_keys = set(rng.sample(
        list(per_commit_counts.keys()),
        min(target_n_commits, len(per_commit_counts)),
    ))

    dedup_sample_budget = 200_000
    per_commit_hashes: Dict[Tuple[str, int], set] = defaultdict(set)
    per_commit_dup_counts: Counter[Tuple[str, int]] = Counter()
    n_kept_after_dedup = 0
    dedup_rows_used = 0

    prefix_len_hist: List[int] = []
    target_len_hist: List[int] = []

    # Aggregate-counters computed only over the sample we actually scan
    # (so n_total_pass2 < n_total in early-exit case). We extrapolate for
    # the full parquet later.
    n_short_prefix = 0
    n_short_target = 0
    n_trivial_target = 0
    n_seen_in_pass2 = 0

    pass2_target_rows = max(
        sample_lengths, int(0.05 * n_total),  # 5% of rows is plenty
        dedup_sample_budget,
    )

    t2 = time.time()
    done_pass2 = False
    for bi, rb in enumerate(stream_split(
        parquet_path, QNA_COLUMNS_FULL,
        keep_in_repo_splits=keep_in_repo_splits,
        batch_size=min(batch_size, 4096),
    )):
        if rb.num_rows == 0:
            continue

        # ── Length stats (Arrow-native, avoids decoding prefix strings) ──
        pre_arr = rb.column("prefix")
        tgt_arr = rb.column("target")
        pre_lens_arr = pc.binary_length(pre_arr)
        tgt_lens_arr = pc.binary_length(tgt_arr)

        # Aggregate: count short-prefix / short-target without per-row Python.
        n_short_prefix += int(
            pc.sum(pc.less(pre_lens_arr, 64).cast(pa.int32())).as_py() or 0
        )
        n_short_target += int(
            pc.sum(pc.less(tgt_lens_arr, 4).cast(pa.int32())).as_py() or 0
        )

        # Save up to sample_lengths length values from the very first batches.
        if len(prefix_len_hist) < sample_lengths:
            need = sample_lengths - len(prefix_len_hist)
            prefix_len_hist.extend(pre_lens_arr.to_pylist()[:need])
            target_len_hist.extend(tgt_lens_arr.to_pylist()[:need])

        # Trivial-target check on the small target column only (no prefix).
        tgts = tgt_arr.to_pylist()
        for t in tgts:
            ts = (t or "").strip()
            if len(ts) <= 6 and (ts.endswith(")") or ts.startswith(",")):
                n_trivial_target += 1
        del tgts

        # Bounded dedup sample. We restrict to the chosen commit subset
        # (sample_keys) and bail early once we hit dedup_sample_budget.
        if dedup_rows_used < dedup_sample_budget and sample_keys:
            repo = rb.column("repo_id").to_pylist()
            ci = rb.column("commit_index").to_pylist()
            keep_idx = [
                j for j in range(rb.num_rows)
                if (repo[j], int(ci[j])) in sample_keys
            ][: dedup_sample_budget - dedup_rows_used]
            if keep_idx:
                idx_arr = pa.array(keep_idx, type=pa.int64())
                sub = rb.take(idx_arr)
                sub_repo = sub.column("repo_id").to_pylist()
                sub_ci = sub.column("commit_index").to_pylist()
                sub_pre = sub.column("prefix").to_pylist()
                sub_tgt = sub.column("target").to_pylist()
                for k in range(sub.num_rows):
                    key = (sub_repo[k], int(sub_ci[k]))
                    h = short_hash(sub_pre[k] or "", sub_tgt[k] or "")
                    if h in per_commit_hashes[key]:
                        per_commit_dup_counts[key] += 1
                    else:
                        per_commit_hashes[key].add(h)
                        n_kept_after_dedup += 1
                dedup_rows_used += sub.num_rows
                del sub_pre, sub_tgt, sub

        n_seen_in_pass2 += rb.num_rows

        # Early exit once we have enough samples to characterize quality.
        if (
            len(prefix_len_hist) >= sample_lengths
            and dedup_rows_used >= dedup_sample_budget
            and n_seen_in_pass2 >= 500_000
        ):
            done_pass2 = True

        if bi % 50 == 0:
            elapsed = time.time() - t2
            print(
                f"  [pass2] {n_seen_in_pass2:,}/{n_total:,} rows in {elapsed:.0f}s "
                f"(dedup_rows_used={dedup_rows_used:,})",
                flush=True,
            )

        if done_pass2:
            print(
                f"  [pass2] early-exit at {n_seen_in_pass2:,} rows "
                f"(targets reached)",
                flush=True,
            )
            break

    t3 = time.time()
    print(f"  [pass2] done: {n_seen_in_pass2:,} rows in {t3 - t2:.0f}s", flush=True)

    # Extrapolate quality counters to the whole parquet for honest reporting.
    if n_seen_in_pass2 > 0 and n_seen_in_pass2 < n_total:
        scale = n_total / n_seen_in_pass2
        n_short_prefix = int(round(n_short_prefix * scale))
        n_short_target = int(round(n_short_target * scale))
        n_trivial_target = int(round(n_trivial_target * scale))

    # Assemble distributions.
    qna_per_commit = list(per_commit_counts.values())
    files_per_commit = [len(s) for s in per_commit_files.values()]
    funcs_per_commit = [len(s) for s in per_commit_funcs.values()]
    types_per_commit = [len(s) for s in per_commit_types.values()]
    rows_per_file = list(per_file_counts.values())

    per_commit_keys = list(per_commit_counts.keys())
    n_commits = len(per_commit_keys)
    sampled_for_dedup = list(per_commit_hashes.keys())
    dup_per_commit = [per_commit_dup_counts.get(k, 0) for k in sampled_for_dedup]
    dup_rate_per_commit = [
        per_commit_dup_counts.get(k, 0) / per_commit_counts[k]
        for k in sampled_for_dedup
    ]

    # Repo coverage.
    repos = {k[0] for k in per_commit_keys}
    commits_per_repo: Dict[str, int] = Counter(k[0] for k in per_commit_keys)
    qna_per_repo: Dict[str, int] = Counter()
    for k, cnt in per_commit_counts.items():
        qna_per_repo[k[0]] += cnt

    return {
        "n_rows": n_total,
        "n_rows_in_dedup_sample": dedup_rows_used,
        "n_rows_after_dedup_in_sample": n_kept_after_dedup,
        "n_commits": n_commits,
        "n_commits_in_dedup_sample": len(sampled_for_dedup),
        "n_repos": len(repos),
        "n_short_prefix_lt64": n_short_prefix,
        "n_short_target_lt4": n_short_target,
        "n_trivial_target": n_trivial_target,
        "qna_per_commit": dist(qna_per_commit),
        "files_per_commit": dist(files_per_commit),
        "funcs_per_commit": dist(funcs_per_commit),
        "types_per_commit": dist(types_per_commit),
        "rows_per_file_per_commit": dist(rows_per_file),
        "dup_rate_per_commit": dist(dup_rate_per_commit),
        "exact_dup_per_commit": dist(dup_per_commit),
        "prefix_chars": dist(prefix_len_hist),
        "target_chars": dist(target_len_hist),
        "qna_per_repo": dist(qna_per_repo.values()),
        "commits_per_repo": dist(commits_per_repo.values()),
        # Embedded raw maps for cap simulation:
        "_raw": {
            "per_commit_counts": per_commit_counts,
            "per_file_counts": per_file_counts,
            "per_commit_dup_counts": per_commit_dup_counts,
            "per_commit_event_counts": per_commit_event_counts,
        },
    }


def simulate_caps(
    raw: Dict[str, Any],
    cap_random: Sequence[int] = (4, 6, 8, 12, 16, 24, 32),
    per_file: Sequence[int] = (1, 2, 3, 4, 6),
    combos: Sequence[Tuple[int, int]] = (
        (1, 6), (1, 8), (2, 8), (2, 12), (2, 16),
        (3, 12), (3, 16), (3, 24),
        (4, 12), (4, 16), (4, 24), (4, 32),
        (6, 12), (6, 16), (6, 24), (6, 32),
    ),
) -> Dict[str, Any]:
    """Estimate kept-QnA counts under several capping strategies."""
    per_commit_counts: Counter = raw["per_commit_counts"]
    per_file_counts: Counter = raw["per_file_counts"]
    n_total = sum(per_commit_counts.values())

    out: Dict[str, Any] = {
        "n_total": n_total,
        "n_commits": len(per_commit_counts),
        "random_K": {},
        "per_file_K": {},
        "per_file_then_random": {},
    }

    # random-K: keep min(count, K)
    for K in cap_random:
        kept = sum(min(c, K) for c in per_commit_counts.values())
        out["random_K"][f"K={K}"] = {
            "kept": kept,
            "keep_ratio": kept / max(n_total, 1),
            "kept_per_commit": dist([min(c, K) for c in per_commit_counts.values()]),
        }

    # per-file-K: for each (commit, file), keep min(count, K).
    # Collapsing per (commit) gives a per-commit kept count.
    per_commit_per_file_kept: Dict[Tuple[str, int], int] = defaultdict(int)
    for (rid, ci, _), cnt in per_file_counts.items():
        # Only used for per-file_K per-K accumulation below.
        pass

    for K in per_file:
        per_commit_kept: Counter = Counter()
        kept_total = 0
        for (rid, ci, _tf), cnt in per_file_counts.items():
            keep = min(cnt, K)
            per_commit_kept[(rid, ci)] += keep
            kept_total += keep
        out["per_file_K"][f"K={K}"] = {
            "kept": kept_total,
            "keep_ratio": kept_total / max(n_total, 1),
            "kept_per_commit": dist(per_commit_kept.values()),
        }

    # per-file-K then random-M (M >= per-file kept, else trim)
    for (Kf, M) in combos:
        per_commit_kept: Counter = Counter()
        for (rid, ci, _tf), cnt in per_file_counts.items():
            per_commit_kept[(rid, ci)] += min(cnt, Kf)
        # Now trim per-commit to min(kept, M).
        trimmed_per_commit = [min(v, M) for v in per_commit_kept.values()]
        kept_total = sum(trimmed_per_commit)
        out["per_file_then_random"][f"per_file={Kf}_per_commit={M}"] = {
            "kept": kept_total,
            "keep_ratio": kept_total / max(n_total, 1),
            "kept_per_commit": dist(trimmed_per_commit),
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "analysis" / "output" / "smart_cap_qna",
    )
    ap.add_argument(
        "--splits", nargs="+", default=["train"],
        choices=list(SPLIT_FILES.keys()),
    )
    ap.add_argument(
        "--keep-in-repo-splits", nargs="+", default=["train"],
        help="Filter QnAs by in_repo_split before stats (default: train only).",
    )
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--sample-lengths", type=int, default=200_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall: Dict[str, Any] = {}
    for split in args.splits:
        path = args.data_dir / "qna" / SPLIT_FILES[split]
        if not path.exists():
            print(f"Missing: {path}", flush=True)
            continue
        print(f"=== {split} :: {path} ===", flush=True)
        stats = collect_per_commit_stats(
            path,
            keep_in_repo_splits=args.keep_in_repo_splits,
            batch_size=args.batch_size,
            sample_lengths=args.sample_lengths,
        )
        raw = stats.pop("_raw")
        sim = simulate_caps(raw)
        out = {"stats": stats, "cap_simulation": sim}
        out_path = args.out_dir / f"{split}_in={'+'.join(args.keep_in_repo_splits)}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}", flush=True)
        overall[split] = out

    if len(overall) > 1:
        merged_path = args.out_dir / "all_splits_summary.json"
        merged_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
        print(f"  wrote {merged_path}", flush=True)


if __name__ == "__main__":
    main()
