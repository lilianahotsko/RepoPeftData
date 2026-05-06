"""Visualize when commits that still contain test files in their (filtered)
``production_code_diff`` occur within each repo's kept-commit timeline, and how
much of the diff byte-budget they account for.

A "test-file commit" here is a kept commit whose ``production_code_diff``
contains at least one ``diff --git`` header pointing at a test file
(``tests/``, ``testing/``, ``test_*.py``, ``*_test.py``, ``conftest.py``).

Outputs four PNG figures + a text summary under ``--out-dir``.

Usage:
    python preprocessing/plot_test_file_commits.py \
        [DATA_DIR] [--out-dir preprocessing/figures]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

SPLITS = ("train", "cr_val", "cr_test")
TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]+\.py$|(^|/)[^/]+_test\.py$|(^|/)conftest\.py$",
    re.IGNORECASE,
)
DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def commit_has_test_file_in_diff(diff: str) -> bool:
    if not diff:
        return False
    for m in DIFF_FILE_HEADER_RE.finditer(diff):
        if TEST_FILE_RE.search(m.group(1)) or TEST_FILE_RE.search(m.group(2)):
            return True
    return False


def load(data_dir: Path):
    rows = []
    for split in SPLITS:
        cpath = data_dir / "commits" / f"{split}.parquet"
        print(f"[load] {cpath}")
        tbl = pq.read_table(
            cpath,
            columns=["repo_id", "commit_index", "production_code_diff"],
        )
        repos = tbl.column("repo_id").to_pylist()
        idxs = tbl.column("commit_index").to_pylist()
        diffs = tbl.column("production_code_diff").to_pylist()
        for r, i, d in zip(repos, idxs, diffs):
            rows.append((split, r, int(i or 0), len(d or ""), bool(d) and commit_has_test_file_in_diff(d)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", nargs="?", default=None)
    p.add_argument("--out-dir", default="preprocessing/figures")
    args = p.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        env = os.environ.get("COMMIT_DATA_DIR")
        if env:
            data_dir = Path(env)
        else:
            scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
            data_dir = Path(scratch) / "REPO_DATASET" / "commit_parquet_hf"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(data_dir)
    print(f"[load] {len(rows):,} rows total")

    # repo_id -> list[(commit_index, diff_bytes, is_test_commit)]
    per_repo: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    repo_split: dict[str, str] = {}
    for split, repo, idx, dbytes, is_test in rows:
        per_repo[repo].append((idx, dbytes, is_test))
        repo_split[repo] = split

    repos_total = len(per_repo)
    repos_with_test = 0
    total_commits = 0
    total_test_commits = 0
    total_diff_bytes = 0
    test_diff_bytes = 0

    # Position summaries (normalized 0..1 within each repo's kept sequence).
    positions = []              # every test-file commit (normalized)
    positions_by_split = {s: [] for s in SPLITS}
    first_test_pos = []         # per repo: earliest normalized position
    last_test_pos = []          # per repo: latest   normalized position
    span_diff_share = []        # per repo: fraction of diff-bytes lying between
                                # first and last test-file commit (inclusive)

    for repo, lst in per_repo.items():
        lst.sort(key=lambda t: t[0])
        n = len(lst)
        total_commits += n
        repo_bytes = sum(b for _, b, _ in lst)
        total_diff_bytes += repo_bytes

        denom = max(n - 1, 1)
        test_indices = [i for (i, _, t) in lst if t]
        total_test_commits += len(test_indices)
        test_diff_bytes += sum(b for (_, b, t) in lst if t)

        if not test_indices:
            continue
        repos_with_test += 1
        lo = min(test_indices)
        hi = max(test_indices)
        first_test_pos.append(lo / denom)
        last_test_pos.append(hi / denom)
        for ti in test_indices:
            np_pos = ti / denom
            positions.append(np_pos)
            positions_by_split[repo_split[repo]].append(np_pos)

        if repo_bytes > 0:
            bytes_between = sum(b for (i, b, _) in lst if lo <= i <= hi)
            span_diff_share.append(bytes_between / repo_bytes)

    # ---- text summary ----
    lines = []
    lines.append(f"Dataset root: {data_dir}")
    lines.append(f"Repos total:                      {repos_total}")
    lines.append(
        f"Repos with >=1 test-file commit:  {repos_with_test} "
        f"({100 * repos_with_test / repos_total:.2f}%)"
    )
    lines.append(f"Kept commits total:               {total_commits:,}")
    lines.append(
        f"Test-file commits:                {total_test_commits:,} "
        f"({100 * total_test_commits / max(total_commits, 1):.3f}%)"
    )
    lines.append(
        f"Diff bytes in test-file commits:  "
        f"{test_diff_bytes:,} / {total_diff_bytes:,} "
        f"({100 * test_diff_bytes / max(total_diff_bytes, 1):.3f}%)"
    )
    if positions:
        arr = np.asarray(positions)
        lines.append("")
        lines.append("Position (0=first kept commit, 1=last) — test-file commits:")
        lines.append(
            f"  n={len(arr)}  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
            f"p25={np.percentile(arr,25):.3f}  p75={np.percentile(arr,75):.3f}"
        )
        for s in SPLITS:
            a = np.asarray(positions_by_split[s]) if positions_by_split[s] else np.asarray([])
            if a.size:
                lines.append(
                    f"  {s:7s}: n={a.size:3d}  mean={a.mean():.3f}  median={np.median(a):.3f}"
                )
    if first_test_pos:
        a = np.asarray(first_test_pos)
        b = np.asarray(last_test_pos)
        lines.append("")
        lines.append(f"First test-file commit position per repo (n={a.size}):")
        lines.append(
            f"  mean={a.mean():.3f}  median={np.median(a):.3f}  "
            f"min={a.min():.3f}  max={a.max():.3f}"
        )
        lines.append("Last test-file commit position per repo:")
        lines.append(
            f"  mean={b.mean():.3f}  median={np.median(b):.3f}  "
            f"min={b.min():.3f}  max={b.max():.3f}"
        )
    if span_diff_share:
        s = np.asarray(span_diff_share)
        lines.append("")
        lines.append(
            "Fraction of per-repo diff bytes inside the span "
            "[first_test_commit .. last_test_commit]:"
        )
        lines.append(
            f"  n={s.size}  mean={s.mean():.3f}  median={np.median(s):.3f}  "
            f"p25={np.percentile(s,25):.3f}  p75={np.percentile(s,75):.3f}"
        )

    summary_path = out_dir / "test_file_commits_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {summary_path}")

    # ---- plots ----
    # 1) Histogram of normalized positions across all repos.
    if positions:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(positions, bins=20, range=(0, 1), color="#3b82f6", edgecolor="white")
        ax.set_xlabel("Normalized position in repo (0 = first kept commit, 1 = last)")
        ax.set_ylabel("# test-file commits")
        ax.set_title(
            f"When do test-file commits occur? (n={len(positions)} commits, "
            f"{repos_with_test} repos)"
        )
        ax.axvline(np.median(positions), color="#ef4444", linestyle="--",
                   label=f"median = {np.median(positions):.2f}")
        ax.legend()
        fig.tight_layout()
        f1 = out_dir / "test_file_commit_positions_hist.png"
        fig.savefig(f1, dpi=140)
        plt.close(fig)
        print(f"[wrote] {f1}")

    # 2) Per-split overlay.
    if any(positions_by_split.values()):
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"train": "#3b82f6", "cr_val": "#10b981", "cr_test": "#f59e0b"}
        for s in SPLITS:
            a = positions_by_split[s]
            if a:
                ax.hist(a, bins=20, range=(0, 1), alpha=0.55,
                        label=f"{s} (n={len(a)})", color=colors[s], edgecolor="white")
        ax.set_xlabel("Normalized position in repo")
        ax.set_ylabel("# test-file commits")
        ax.set_title("Test-file commit positions, split overlay")
        ax.legend()
        fig.tight_layout()
        f2 = out_dir / "test_file_commit_positions_by_split.png"
        fig.savefig(f2, dpi=140)
        plt.close(fig)
        print(f"[wrote] {f2}")

    # 3) Per-repo scatter: each test-file commit as a dot. Repos sorted by
    #    first-test-commit position so clustering patterns are visible.
    if first_test_pos:
        repos_with_tests = [
            r for r in per_repo
            if any(t for (_, _, t) in per_repo[r])
        ]
        repos_with_tests.sort(
            key=lambda r: min(
                i for (i, _, t) in per_repo[r] if t
            ) / max(len(per_repo[r]) - 1, 1)
        )
        xs = []
        ys = []
        split_colors = []
        for y, r in enumerate(repos_with_tests):
            lst = sorted(per_repo[r], key=lambda t: t[0])
            n = len(lst)
            denom = max(n - 1, 1)
            for i, _, is_test in lst:
                if is_test:
                    xs.append(i / denom)
                    ys.append(y)
                    split_colors.append(
                        {"train": "#3b82f6",
                         "cr_val": "#10b981",
                         "cr_test": "#f59e0b"}[repo_split[r]]
                    )
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(repos_with_tests))))
        ax.scatter(xs, ys, c=split_colors, s=22)
        ax.set_yticks(range(len(repos_with_tests)))
        ax.set_yticklabels(repos_with_tests, fontsize=7)
        ax.set_xlabel("Normalized position in repo")
        ax.set_title(f"Test-file commits per repo (n={len(repos_with_tests)} repos)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", alpha=0.25)
        # Legend
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="", color="#3b82f6", label="train"),
                Line2D([0], [0], marker="o", linestyle="", color="#10b981", label="cr_val"),
                Line2D([0], [0], marker="o", linestyle="", color="#f59e0b", label="cr_test"),
            ],
            loc="lower right",
        )
        fig.tight_layout()
        f3 = out_dir / "test_file_commits_per_repo.png"
        fig.savefig(f3, dpi=140)
        plt.close(fig)
        print(f"[wrote] {f3}")

    # 4) Histogram of per-repo diff-byte share inside the [first,last] test
    #    commit span — how much of the code change sits "around" these commits.
    if span_diff_share:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(span_diff_share, bins=20, range=(0, 1),
                color="#8b5cf6", edgecolor="white")
        ax.set_xlabel("Fraction of repo's diff bytes within [first..last] test-file commit span")
        ax.set_ylabel("# repos")
        ax.set_title("Diff-bytes share covered by the test-file-commit span")
        ax.axvline(np.median(span_diff_share), color="#ef4444", linestyle="--",
                   label=f"median = {np.median(span_diff_share):.2f}")
        ax.legend()
        fig.tight_layout()
        f4 = out_dir / "test_file_span_diff_share_hist.png"
        fig.savefig(f4, dpi=140)
        plt.close(fig)
        print(f"[wrote] {f4}")


if __name__ == "__main__":
    main()
