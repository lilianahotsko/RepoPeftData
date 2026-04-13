#!/usr/bin/env python3
"""
Compare full first-parent history length vs commits stored in commits_assertions.db.

- **Original**: `git rev-list --first-parent --count HEAD` (same lineage as DB builder).
- **In database**: `COUNT(*)` per repo_id in `commits` table (sampled rows, capped by
  `--max-commits` at build time, default 200).

Writes JSON with per-repo rows and dataset-level summary (means, medians, coverage).

Usage:
    python analysis/commit_sampling_stats.py \\
        --db-path $SCRATCH/REPO_DATASET/commits_assertions.db \\
        --repos-root $SCRATCH/REPO_DATASET/repositories
    python analysis/commit_sampling_stats.py --output-json analysis/output/commit_sampling_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "create_dataset"))

from build_commit_assertion_db import repo_dir_for_name  # noqa: E402


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def count_first_parent_commits(repo_dir: Path, timeout: int = 300) -> Optional[int]:
    """Match build_commit_assertion_db: first-parent line only."""
    if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
        return None
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-list", "--first-parent", "--count", "HEAD"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return None
        s = r.stdout.strip()
        return int(s) if s.isdigit() else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def load_db_commit_counts(db_path: Path) -> Dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT repo_id, COUNT(*) AS n FROM commits GROUP BY repo_id ORDER BY repo_id",
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}
    finally:
        conn.close()


def main() -> None:
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap = argparse.ArgumentParser(
        description="Stats: full first-parent commit count vs commits table rows",
    )
    ap.add_argument("--db-path", type=str, default=os.path.join(default_root, "commits_assertions.db"))
    ap.add_argument(
        "--repos-root",
        type=str,
        default=None,
        help="Clone root (default: <parent of db>/repositories or REPO_DATASET/repositories)",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Default: analysis/output/commit_sampling_stats.json",
    )
    ap.add_argument("--limit-repos", type=int, default=None, help="Process only first N repos (sorted id)")
    args = ap.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    repos_root = Path(
        args.repos_root or (db_path.parent / "repositories"),
    ).expanduser().resolve()

    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    db_counts = load_db_commit_counts(db_path)
    repo_ids = sorted(db_counts.keys())
    if args.limit_repos is not None:
        repo_ids = repo_ids[: args.limit_repos]

    per_repo: List[Dict[str, Any]] = []
    n_ok = 0
    n_missing_git = 0

    for rid in repo_ids:
        rdir = repo_dir_for_name(repos_root, rid)
        n_full = count_first_parent_commits(rdir)
        n_db = db_counts[rid]
        if n_full is None:
            n_missing_git += 1
            ratio = None
        else:
            n_ok += 1
            ratio = n_db / n_full if n_full > 0 else None

        per_repo.append({
            "repo_id": rid,
            "first_parent_commits_git": n_full,
            "commits_in_database": n_db,
            "coverage": ratio,
            "skipped_commits": (n_full - n_db) if n_full is not None else None,
        })

    ratios = [p["coverage"] for p in per_repo if p["coverage"] is not None]
    ratios.sort()
    totals_git = [p["first_parent_commits_git"] for p in per_repo if p["first_parent_commits_git"] is not None]
    totals_db = [p["commits_in_database"] for p in per_repo]
    skipped = [p["skipped_commits"] for p in per_repo if p["skipped_commits"] is not None]

    summary: Dict[str, Any] = {
        "db_path": str(db_path),
        "repos_root": str(repos_root),
        "n_repos_in_database": len(db_counts),
        "n_repos_analyzed": len(per_repo),
        "n_repos_with_git_count": n_ok,
        "n_repos_missing_clone_or_git_error": n_missing_git,
        "note": (
            "first_parent_commits_git = rev-list --first-parent --count HEAD; "
            "commits_in_database = rows in commits table (builder sample, max 200 default)."
        ),
    }

    if totals_git:
        summary["first_parent_commits_git"] = {
            "min": min(totals_git),
            "max": max(totals_git),
            "mean": mean(totals_git),
            "median": median(totals_git),
            "std": pstdev(totals_git) if len(totals_git) > 1 else 0.0,
            "p90": _percentile(sorted(float(x) for x in totals_git), 90),
            "p99": _percentile(sorted(float(x) for x in totals_git), 99),
        }
    if totals_db:
        summary["commits_in_database"] = {
            "min": min(totals_db),
            "max": max(totals_db),
            "mean": mean(totals_db),
            "median": median(totals_db),
            "std": pstdev(totals_db) if len(totals_db) > 1 else 0.0,
        }
    if ratios:
        summary["coverage_ratio_db_over_git"] = {
            "min": min(ratios),
            "max": max(ratios),
            "mean": mean(ratios),
            "median": median(ratios),
            "p10": _percentile(ratios, 10),
            "p90": _percentile(ratios, 90),
        }
    if skipped:
        summary["skipped_commits_git_minus_db"] = {
            "min": min(skipped),
            "max": max(skipped),
            "mean": mean(skipped),
            "median": median(skipped),
        }

    out_path = Path(
        args.output_json or (ROOT / "analysis" / "output" / "commit_sampling_stats.json"),
    ).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "per_repo": per_repo}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
