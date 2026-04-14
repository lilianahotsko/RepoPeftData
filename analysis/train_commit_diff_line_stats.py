#!/usr/bin/env python3
"""
Statistics for training split: commit counts, line counts, and tokenizer token
counts for production_code_diff (matches diff embedding tokenizer by default).

Uses the same split resolution as hypernetwork/train_code2lora_gru_commits.py
(gru_train.json preferred over train.json).

Usage:
  python analysis/train_commit_diff_line_stats.py \\
      --db-path src/commits_assertions.db --splits-dir src

  python analysis/train_commit_diff_line_stats.py \\
      --db-path $SCRATCH/REPO_DATASET/commits_assertions.db \\
      --splits-dir $SCRATCH/REPO_DATASET \\
      --per-commit-csv /tmp/train_commit_lines.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_split_repo_ids(splits_dir: Path, split_name: str) -> List[str]:
    for prefix in ("gru_", ""):
        path = splits_dir / f"{prefix}{split_name}"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return sorted(data.get("repositories", {}).keys())
    return []


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def diff_line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def make_diff_token_counter(
    tokenizer_model: str,
    chunk_chars: int,
) -> Tuple[Callable[[str], int], str]:
    """Token count per diff; long texts are encoded in chunks (embedder-style)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            total = 0
            for i in range(0, len(text), chunk_chars):
                chunk = text[i : i + chunk_chars]
                total += len(tok.encode(chunk, add_special_tokens=False))
            return total

        return count_tokens, f"hf:{tokenizer_model}"

    except Exception as exc:  # noqa: BLE001

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return max(1, len(text) // 4)

        return count_tokens, f"heuristic_chars_div_4_fallback ({exc!r})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train-split commit & diff line statistics")
    ap.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="commits_assertions.db",
    )
    ap.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory with gru_train.json / train.json",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train.json",
        help="Split file name (default: train.json)",
    )
    ap.add_argument(
        "--per-commit-csv",
        type=Path,
        default=None,
        help="Optional path: repo_id, commit_index, diff_lines, diff_tokens",
    )
    ap.add_argument(
        "--tokenizer-model",
        type=str,
        default=DEFAULT_TOKENIZER_MODEL,
        help=f"HuggingFace tokenizer for diff token counts (default: {DEFAULT_TOKENIZER_MODEL})",
    )
    ap.add_argument(
        "--token-chunk-chars",
        type=int,
        default=200_000,
        help="Encode diffs in this many UTF-8 chars per tokenizer.encode call (avoids huge single calls)",
    )
    ap.add_argument(
        "--no-tokens",
        action="store_true",
        help="Skip token counting (lines and repo stats only)",
    )
    args = ap.parse_args()

    db_path = args.db_path.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    repo_ids = load_split_repo_ids(splits_dir, args.split)
    if not repo_ids:
        raise SystemExit(
            f"No repos found for split {args.split!r} under {splits_dir} "
            "(expected gru_train.json or train.json)",
        )

    placeholders = ",".join("?" for _ in repo_ids)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT repo_id, commit_index, production_code_diff FROM commits "
            f"WHERE repo_id IN ({placeholders}) ORDER BY repo_id, commit_index",
            repo_ids,
        ).fetchall()
    finally:
        conn.close()

    per_repo_commits: dict[str, int] = {}
    line_counts: List[int] = []
    token_counts: List[int] = []
    per_commit_rows: List[Tuple[str, int, int, int]] = []

    count_tokens: Callable[[str], int] | None = None
    token_backend = "none"
    if not args.no_tokens:
        count_tokens, token_backend = make_diff_token_counter(
            args.tokenizer_model,
            args.token_chunk_chars,
        )

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_kw):
            return it

    row_iter = tqdm(rows, desc="commits", unit="commit") if count_tokens else rows

    for repo_id, commit_index, diff_text in row_iter:
        nlines = diff_line_count(diff_text)
        ntokens = count_tokens(diff_text) if count_tokens else 0
        per_repo_commits[repo_id] = per_repo_commits.get(repo_id, 0) + 1
        line_counts.append(nlines)
        token_counts.append(ntokens)
        per_commit_rows.append((repo_id, commit_index, nlines, ntokens))

    n_commits = len(line_counts)
    line_sorted = sorted(float(x) for x in line_counts)
    token_sorted = sorted(float(x) for x in token_counts) if token_counts else []

    print("=== Training split (DB ∩ split repos) ===")
    print(f"  Split file resolved from: {splits_dir}")
    print(f"  Repos in split file:     {len(repo_ids)}")
    print(f"  Repos with ≥1 commit in DB: {len(per_repo_commits)}")
    print(f"  Total commits (rows):  {n_commits}")
    if n_commits == 0:
        return

    print()
    print("=== Lines per unified diff (production_code_diff) ===")
    print(f"  min:    {int(min(line_counts))}")
    print(f"  max:    {int(max(line_counts))}")
    print(f"  mean:   {mean(line_counts):.2f}")
    print(f"  median: {median(line_counts):.2f}")
    if len(line_counts) > 1:
        print(f"  std:    {pstdev(line_counts):.2f}")
    print(f"  p90:    {_percentile(line_sorted, 90):.1f}")
    print(f"  p95:    {_percentile(line_sorted, 95):.1f}")
    print(f"  p99:    {_percentile(line_sorted, 99):.1f}")

    if not args.no_tokens:
        print()
        print("=== Tokens per unified diff (production_code_diff) ===")
        print(f"  tokenizer: {token_backend}")
        print(f"  min:    {int(min(token_counts))}")
        print(f"  max:    {int(max(token_counts))}")
        print(f"  mean:   {mean(token_counts):.2f}")
        print(f"  median: {median(token_counts):.2f}")
        if len(token_counts) > 1:
            print(f"  std:    {pstdev(token_counts):.2f}")
        print(f"  p90:    {_percentile(token_sorted, 90):.1f}")
        print(f"  p95:    {_percentile(token_sorted, 95):.1f}")
        print(f"  p99:    {_percentile(token_sorted, 99):.1f}")

    print()
    print("=== Commits per repo (repos that appear in DB) ===")
    counts = sorted(per_repo_commits.values())
    cr_sorted = [float(x) for x in counts]
    print(f"  min:    {min(counts)}")
    print(f"  max:    {max(counts)}")
    print(f"  mean:   {mean(counts):.2f}")
    print(f"  median: {median(counts):.2f}")
    if len(counts) > 1:
        print(f"  std:    {pstdev(counts):.2f}")
    print(f"  p90:    {_percentile(cr_sorted, 90):.1f}")

    if args.per_commit_csv:
        args.per_commit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_commit_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["repo_id", "commit_index", "diff_lines", "diff_tokens"])
            w.writerows(per_commit_rows)
        print()
        print(f"Wrote per-commit table: {args.per_commit_csv}")


if __name__ == "__main__":
    main()
