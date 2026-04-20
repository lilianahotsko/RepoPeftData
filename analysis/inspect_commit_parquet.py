#!/usr/bin/env python3
"""
Inspect the Parquet dataset produced by
``create_dataset/build_commit_parquet_db.py``.

Reports:

  * number of repos / kept commits / qna pairs per cross-repo split and per
    in-repo split (train / val / test).
  * repos with zero kept commits (should be dropped from training).
  * commits-per-repo distribution (min / mean / p50 / p90 / p95 / max).
  * new-assertions-per-commit distribution.
  * prefix-tokens and diff-tokens histograms, optionally tokenized with a
    HuggingFace tokenizer (e.g. Qwen3-Embedding-0.6B). Falls back to
    len(str)/4 as a rough token estimate when ``--tokenizer`` is omitted.
  * sanity: every ``qna_pairs.commit_index`` exists in ``commits`` for the
    same ``repo_id``.

Example:
    python analysis/inspect_commit_parquet.py \\
        --dir $SCRATCH/REPO_DATASET/commit_parquet \\
        --tokenizer hf:Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow.compute as pc  # type: ignore
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def describe(name: str, vals: List[float], unit: str = "") -> None:
    if not vals:
        print(f"  {name:<32} (empty)")
        return
    s = sorted(vals)
    mean = sum(s) / len(s)
    print(
        f"  {name:<32} "
        f"n={len(s):<7d} "
        f"min={s[0]:.0f} "
        f"p50={_percentile(s, 50):.0f} "
        f"mean={mean:.1f} "
        f"p90={_percentile(s, 90):.0f} "
        f"p95={_percentile(s, 95):.0f} "
        f"p99={_percentile(s, 99):.0f} "
        f"max={s[-1]:.0f}{unit}"
    )


# ---------------------------------------------------------------------------
# Tokenizer plumbing
# ---------------------------------------------------------------------------

def _build_tokenizer(spec: Optional[str]) -> Callable[[str], int]:
    """Return a function str -> token-count. ``None`` -> crude char/4 proxy."""
    if not spec:
        return lambda s: max(0, len(s) // 4)

    if spec.startswith("hf:"):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                f"transformers is required for --tokenizer {spec}: {exc}"
            )
        tok = AutoTokenizer.from_pretrained(spec[len("hf:"):])
        return lambda s: len(tok.encode(s, add_special_tokens=False))

    raise SystemExit(f"Unknown tokenizer spec: {spec}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise SystemExit("pyarrow is required: pip install pyarrow")


def _value_counts(tbl: "pa.Table", col: str) -> List[tuple]:
    ser = tbl.column(col).to_pylist()
    counts: dict = {}
    for v in ser:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect commit-level Parquet dataset",
    )
    default_dir = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
        "commit_parquet",
    )
    ap.add_argument("--dir", type=str, default=default_dir,
                    help="Directory containing commits.parquet + qna_pairs.parquet")
    ap.add_argument("--commits-file", type=str, default=None,
                    help="Override path to commits.parquet")
    ap.add_argument("--qna-file", type=str, default=None,
                    help="Override path to qna_pairs.parquet")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HF tokenizer spec like 'hf:Qwen/Qwen3-Embedding-0.6B' "
                         "(default: crude char/4 proxy)")
    ap.add_argument("--sample-tokens", type=int, default=5000,
                    help="Max rows to sample for token stats (cost control)")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()

    _require_pyarrow()

    dir_ = Path(args.dir).expanduser().resolve()
    commits_path = Path(args.commits_file) if args.commits_file else dir_ / "commits.parquet"
    qna_path = Path(args.qna_file) if args.qna_file else dir_ / "qna_pairs.parquet"

    if not commits_path.exists():
        raise SystemExit(f"Missing {commits_path}")
    if not qna_path.exists():
        raise SystemExit(f"Missing {qna_path}")

    print(f"Reading {commits_path}")
    commits = pq.read_table(commits_path)
    print(f"Reading {qna_path}")
    qna = pq.read_table(qna_path)

    print(f"\nCommits rows: {commits.num_rows}")
    print(f"QnA rows:     {qna.num_rows}")

    # ---- Per-split summaries ----
    print("\n[Cross-repo split]")
    cr = _value_counts(commits, "cross_repo_split")
    for v, n in cr:
        sub = commits.filter(pc.equal(commits["cross_repo_split"], v))
        n_repos = len(set(sub.column("repo_id").to_pylist()))
        n_q = qna.filter(pc.equal(qna["cross_repo_split"], v)).num_rows
        print(f"  {v:<10} commits={n:<7d} repos={n_repos:<5d} qna={n_q}")

    print("\n[In-repo split]")
    for v, n in _value_counts(commits, "in_repo_split"):
        n_q = qna.filter(pc.equal(qna["in_repo_split"], v)).num_rows
        print(f"  {v:<10} commits={n:<7d} qna={n_q}")

    # ---- Repos with zero kept commits: harvested from _done.jsonl if present
    done_path = dir_ / "_done.jsonl"
    if done_path.exists():
        import json
        zero_kept: List[str] = []
        total_done = 0
        for line in done_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_done += 1
            if int(rec.get("kept_commits", 0)) == 0:
                zero_kept.append(rec.get("repo_id", "?"))
        print(f"\n[Repos with zero kept commits] {len(zero_kept)} / {total_done}")
        for r in zero_kept[:15]:
            print(f"  {r}")
        if len(zero_kept) > 15:
            print(f"  ... and {len(zero_kept) - 15} more")

    # ---- Commits per repo ----
    repo_commit_counts: dict = {}
    for r in commits.column("repo_id").to_pylist():
        repo_commit_counts[r] = repo_commit_counts.get(r, 0) + 1
    describe(
        "kept_commits per repo",
        [float(v) for v in repo_commit_counts.values()],
    )

    # ---- New assertions per commit ----
    describe(
        "new_assertions per commit",
        [float(v) for v in commits.column("n_new_assertions").to_pylist()],
    )

    # ---- Token stats (sampled) ----
    tok_fn = _build_tokenizer(args.tokenizer)
    tok_name = args.tokenizer or "chars/4 proxy"

    n_q = qna.num_rows
    sample_n = min(args.sample_tokens, n_q) if args.sample_tokens > 0 else n_q
    if sample_n and n_q:
        import random
        idx = list(range(n_q))
        random.Random(3407).shuffle(idx)
        idx = sorted(idx[:sample_n])
        prefix_sample = qna.column("prefix").take(pa.array(idx)).to_pylist()
        target_sample = qna.column("target").take(pa.array(idx)).to_pylist()
        print(f"\n[Token stats — tokenizer: {tok_name}  sample: {sample_n}/{n_q}]")
        describe("prefix tokens (qna)", [float(tok_fn(s)) for s in prefix_sample])
        describe("target tokens (qna)", [float(tok_fn(s)) for s in target_sample])

    n_c = commits.num_rows
    sample_c = min(args.sample_tokens, n_c) if args.sample_tokens > 0 else n_c
    if sample_c and n_c:
        import random
        idx = list(range(n_c))
        random.Random(3407).shuffle(idx)
        idx = sorted(idx[:sample_c])
        diff_sample = commits.column("production_code_diff").take(pa.array(idx)).to_pylist()
        print(f"\n[Diff token stats — sample: {sample_c}/{n_c}]")
        describe("production_code_diff tokens", [float(tok_fn(s)) for s in diff_sample])
        describe("production_code_diff lines",
                 [float(s.count("\n")) if s else 0.0 for s in diff_sample])

    # ---- Join key sanity ----
    if not args.no_sanity:
        print("\n[Join sanity: every qna (repo_id, commit_index) exists in commits]")
        commit_keys = set(zip(
            commits.column("repo_id").to_pylist(),
            commits.column("commit_index").to_pylist(),
        ))
        q_repo = qna.column("repo_id").to_pylist()
        q_ci = qna.column("commit_index").to_pylist()
        missing = 0
        for r, c in zip(q_repo, q_ci):
            if (r, c) not in commit_keys:
                missing += 1
        print(f"  missing join rows: {missing} / {qna.num_rows}")


if __name__ == "__main__":
    main()
