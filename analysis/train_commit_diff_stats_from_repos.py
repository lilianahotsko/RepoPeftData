#!/usr/bin/env python3
"""
Same statistics as train_commit_diff_line_stats.py (lines + tokens per filtered
unified diff), but computed from **cloned git repos** instead of SQLite.

Replays the DB builder logic: first-parent history, optional uniform sampling
(same as the DB when max-commits is positive), then `git diff` + path filtering — see
create_dataset/build_commit_assertion_db.py.

Use **`--max-commits 0`** for **no cap**: every first-parent commit in each repo.

Usage:
  python analysis/train_commit_diff_stats_from_repos.py \\
      --splits-dir $SCRATCH/REPO_DATASET \\
      --repos-root $SCRATCH/REPO_DATASET/repositories \\
      --split train.json \\
      --max-commits 0

  python analysis/train_commit_diff_stats_from_repos.py \\
      --splits-dir src \\
      --repos-root /path/to/repositories \\
      --max-commits 200 \\
      --diff-mode single_commit \\
      --per-commit-csv /tmp/git_commit_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "create_dataset"))
sys.path.insert(0, str(ROOT / "analysis"))

from build_commit_assertion_db import (  # noqa: E402
    filter_diff,
    get_commit_list,
    get_diff,
    repo_dir_for_name,
    sample_commit_indices,
)

import train_commit_diff_line_stats as line_stats  # noqa: E402


def _percentile(sorted_vals: List[float], p: float) -> float:
    return line_stats._percentile(sorted_vals, p)


def select_commit_indices(n_total: int, max_commits: int) -> List[int]:
    """Indices into first-parent history. max_commits <= 0: all commits (no sampling)."""
    if n_total <= 0:
        return []
    if max_commits <= 0:
        return list(range(n_total))
    return sample_commit_indices(n_total, max_commits)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train-split diff line/token stats from git repos (not DB)",
    )
    ap.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory with gru_train.json / train.json",
    )
    ap.add_argument(
        "--repos-root",
        type=Path,
        default=None,
        help="Clone root (default: <splits-dir>/repositories)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train.json",
        help="Split JSON basename (default: train.json)",
    )
    ap.add_argument(
        "--max-commits",
        type=int,
        default=200,
        help=(
            "Max commits per repo after uniform sampling (same as DB build). "
            "Use 0 for no limit: all first-parent commits in each repo."
        ),
    )
    ap.add_argument(
        "--diff-mode",
        type=str,
        choices=["single_commit", "inter_sample"],
        default="single_commit",
        help="single_commit: diff vs parent; inter_sample: vs previous sample",
    )
    ap.add_argument(
        "--limit-repos",
        type=int,
        default=None,
        help="Only first N repos from split (debug)",
    )
    ap.add_argument(
        "--per-commit-csv",
        type=Path,
        default=None,
        help="repo_id, commit_index, commit_sha, diff_lines, diff_tokens",
    )
    ap.add_argument(
        "--tokenizer-model",
        type=str,
        default=line_stats.DEFAULT_TOKENIZER_MODEL,
        help="HF tokenizer for token counts",
    )
    ap.add_argument(
        "--token-chunk-chars",
        type=int,
        default=200_000,
    )
    ap.add_argument(
        "--no-tokens",
        action="store_true",
    )
    args = ap.parse_args()

    splits_dir = args.splits_dir.expanduser().resolve()
    repos_root = (
        args.repos_root.expanduser().resolve()
        if args.repos_root
        else splits_dir / "repositories"
    )

    repo_ids = line_stats.load_split_repo_ids(splits_dir, args.split)
    if not repo_ids:
        raise SystemExit(
            f"No repos for split {args.split!r} under {splits_dir}",
        )
    if args.limit_repos is not None:
        repo_ids = repo_ids[: args.limit_repos]

    count_tokens: Callable[[str], int] | None = None
    token_backend = "none"
    if not args.no_tokens:
        count_tokens, token_backend = line_stats.make_diff_token_counter(
            args.tokenizer_model,
            args.token_chunk_chars,
        )

    line_counts: List[int] = []
    token_counts: List[int] = []
    per_commit_rows: List[Tuple[str, int, str, int, int]] = []
    n_skipped_no_git = 0
    n_skipped_empty = 0

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_kw):
            return x

    for repo_id in tqdm(repo_ids, desc="repos", unit="repo"):
        rdir = repo_dir_for_name(repos_root, repo_id)
        if not rdir.is_dir() or not (rdir / ".git").exists():
            n_skipped_no_git += 1
            continue

        full_commits = get_commit_list(rdir)
        if not full_commits:
            n_skipped_empty += 1
            continue

        sampled_indices = select_commit_indices(
            len(full_commits), args.max_commits,
        )
        prev_sampled_sha: str | None = None

        for si, ci in enumerate(sampled_indices):
            c = full_commits[ci]
            sha = c["sha"]
            if args.diff_mode == "inter_sample":
                parent = prev_sampled_sha
            else:
                parent = c.get("parent_sha")

            raw = get_diff(rdir, parent, sha)
            filtered = filter_diff(raw)
            prev_sampled_sha = sha

            nlines = line_stats.diff_line_count(filtered)
            ntokens = count_tokens(filtered) if count_tokens else 0
            line_counts.append(nlines)
            token_counts.append(ntokens)
            per_commit_rows.append((repo_id, si, sha, nlines, ntokens))

    n_commits = len(line_counts)
    line_sorted = sorted(float(x) for x in line_counts)
    token_sorted = sorted(float(x) for x in token_counts) if token_counts else []

    print("=== From git repos (first-parent history + filtered diffs) ===")
    print(f"  Splits dir:    {splits_dir}")
    print(f"  Repos root:    {repos_root}")
    print(f"  Split:         {args.split}")
    print(f"  Diff mode:     {args.diff_mode}")
    if args.max_commits <= 0:
        print(f"  Max commits:   0 (all commits per repo, no sampling)")
    else:
        print(f"  Max commits:   {args.max_commits} (uniform sample if history longer)")
    print(f"  Repos in split:{len(repo_ids)}")
    print(f"  Skipped (no .git): {n_skipped_no_git}")
    print(f"  Skipped (no commits): {n_skipped_empty}")
    print(f"  Total commit diffs: {n_commits}")
    if n_commits == 0:
        return

    print()
    print("=== Lines per filtered unified diff ===")
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
        print("=== Tokens per filtered unified diff ===")
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

    per_repo: dict[str, int] = {}
    for repo_id, _si, _sha, _l, _t in per_commit_rows:
        per_repo[repo_id] = per_repo.get(repo_id, 0) + 1
    counts = sorted(per_repo.values())
    cr_sorted = [float(x) for x in counts]
    print()
    print("=== Sampled commits per repo ===")
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
            w.writerow(
                ["repo_id", "commit_index", "commit_sha", "diff_lines", "diff_tokens"],
            )
            w.writerows(per_commit_rows)
        print()
        print(f"Wrote per-commit table: {args.per_commit_csv}")


if __name__ == "__main__":
    main()
