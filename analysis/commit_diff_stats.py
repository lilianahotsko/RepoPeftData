#!/usr/bin/env python3
"""
Commit-version and unified-diff token statistics for the paper and JSON export.

1) Sampled versions per repository: from gru_*.json (unique repos across splits),
   using len(commit_history.commits) — same sampled history as Code2LoRA-GRU.

2) Tokens per unified diff: from commits_assertions.db (production_code_diff) when
   the table is populated, else optional --sample-git-if-empty-db over local clones.

Tokenizer: Qwen3-Embedding-0.6B by default (matches diff embedding in commit-level
training). If transformers is not installed, falls back to a rough heuristic
(~4 characters per token).

Usage:
    python analysis/commit_diff_stats.py --splits-dir $SCRATCH/REPO_DATASET
    python analysis/commit_diff_stats.py --db-path $SCRATCH/REPO_DATASET/commits_assertions.db \\
        --splits-dir $SCRATCH/REPO_DATASET
    python analysis/commit_diff_stats.py --sample-git-if-empty-db --git-sample-repos 64 \\
        --git-sample-max-commits 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "create_dataset"))

from build_commit_assertion_db import (  # noqa: E402
    build_production_diff,
    first_parent_sha,
    repo_dir_for_name,
)
from extract_commit_history import extract_repo_history, get_files_at_commit  # noqa: E402


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def discover_gru_splits(splits_dir: Path) -> List[Path]:
    return sorted(splits_dir.glob("gru_*.json"))


def aggregate_versions_from_gru(splits_dir: Path) -> Dict[str, Any]:
    seen = set()
    counts: List[int] = []
    for path in discover_gru_splits(splits_dir):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for repo_id, rdata in data.get("repositories", {}).items():
            if repo_id in seen:
                continue
            seen.add(repo_id)
            ch = rdata.get("commit_history")
            if not ch:
                continue
            n = len(ch.get("commits", []))
            counts.append(n)
    counts.sort()
    if not counts:
        return {
            "n_repos": 0,
            "note": "No gru_*.json or missing commit_history.",
        }
    return {
        "n_repos": len(counts),
        "versions_per_repo_min": min(counts),
        "versions_per_repo_max": max(counts),
        "versions_per_repo_mean": mean(counts),
        "versions_per_repo_median": median(counts),
        "versions_per_repo_std": pstdev(counts) if len(counts) > 1 else 0.0,
        "versions_per_repo_p90": _percentile(counts, 90),
        "versions_per_repo_p95": _percentile(counts, 95),
        "versions_per_repo_p99": _percentile(counts, 99),
        "note": (
            "Sampled commit count per repo (git log --first-parent, uniform sample, "
            "max 200 commits, first and last always included)."
        ),
    }


def _make_token_counter(tokenizer_model: str):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(tok.encode(text, add_special_tokens=False))

        return count_tokens, f"hf:{tokenizer_model}"
    except Exception as exc:  # noqa: BLE001
        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return max(1, len(text) // 4)

        return count_tokens, f"heuristic_chars_div_4_fallback ({exc!r})"


def stats_from_sqlite(db_path: Path, count_tokens) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        n = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        if n == 0:
            return {"n_diff_rows": 0, "source": "sqlite_empty"}
        rows = conn.execute(
            "SELECT repo_id, commit_index, production_code_diff FROM commits "
            "ORDER BY repo_id, commit_index"
        ).fetchall()
    finally:
        conn.close()

    token_lens: List[int] = []
    per_repo_versions: Dict[str, int] = {}
    for repo_id, _ci, diff in rows:
        per_repo_versions[repo_id] = per_repo_versions.get(repo_id, 0) + 1
        token_lens.append(count_tokens(diff or ""))
    token_lens.sort()
    vc = sorted(per_repo_versions.values())
    return {
        "n_diff_rows": len(token_lens),
        "n_repos_in_db": len(per_repo_versions),
        "diff_tokens_min": min(token_lens),
        "diff_tokens_max": max(token_lens),
        "diff_tokens_mean": mean(token_lens),
        "diff_tokens_median": median(token_lens),
        "diff_tokens_std": pstdev(token_lens) if len(token_lens) > 1 else 0.0,
        "diff_tokens_p90": _percentile(token_lens, 90),
        "diff_tokens_p95": _percentile(token_lens, 95),
        "diff_tokens_p99": _percentile(token_lens, 99),
        "db_versions_per_repo_min": min(vc),
        "db_versions_per_repo_max": max(vc),
        "db_versions_per_repo_mean": mean(vc),
        "db_versions_per_repo_median": median(vc),
        "source": "sqlite",
    }


def _all_unique_repo_ids(splits_dir: Path) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for path in discover_gru_splits(splits_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        for rid in sorted(data.get("repositories", {}).keys()):
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
    return ordered


def sample_git_diff_tokens(
    repos_root: Path,
    count_tokens,
    repo_ids: List[str],
    max_commits_per_repo: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    token_lens: List[int] = []
    repos_touched = 0
    for repo_id in repo_ids:
        rdir = repo_dir_for_name(repos_root, repo_id)
        if not rdir.is_dir() or not (rdir / ".git").exists():
            continue
        hist = extract_repo_history(rdir, max_commits=200, preamble_frac=0.1)
        if hist is None:
            continue
        commits_list = hist["commits"]
        if len(commits_list) > max_commits_per_repo:
            idxs = rng.sample(
                range(len(commits_list)),
                k=max_commits_per_repo,
            )
            commits_list = [commits_list[i] for i in sorted(idxs)]
        else:
            commits_list = list(commits_list)
        repos_touched += 1
        for entry in commits_list:
            sha = entry["sha"]
            paths = get_files_at_commit(rdir, sha)
            parent = first_parent_sha(rdir, sha)
            parent_paths = set(get_files_at_commit(rdir, parent)) if parent else set()
            diff_paths = parent_paths | set(paths)
            diff_text = build_production_diff(rdir, parent, sha, diff_paths)
            token_lens.append(count_tokens(diff_text))
    token_lens.sort()
    if not token_lens:
        return {
            "n_diff_rows": 0,
            "source": "git_sample",
            "repos_with_git": repos_touched,
        }
    return {
        "n_diff_rows": len(token_lens),
        "diff_tokens_min": min(token_lens),
        "diff_tokens_max": max(token_lens),
        "diff_tokens_mean": mean(token_lens),
        "diff_tokens_median": median(token_lens),
        "diff_tokens_std": pstdev(token_lens) if len(token_lens) > 1 else 0.0,
        "diff_tokens_p90": _percentile(token_lens, 90),
        "diff_tokens_p95": _percentile(token_lens, 95),
        "diff_tokens_p99": _percentile(token_lens, 99),
        "source": "git_sample",
        "repos_sampled": len(repo_ids),
        "repos_with_clones": repos_touched,
        "max_commits_per_repo": max_commits_per_repo,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Commit / diff statistics for paper")
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_root)
    ap.add_argument("--db-path", type=str, default=None)
    ap.add_argument("--repos-root", type=str, default=None)
    ap.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Default: analysis/output/commit_diff_stats.json",
    )
    ap.add_argument(
        "--sample-git-if-empty-db",
        action="store_true",
        help="If commits table is empty, sample unified diffs via git on local clones",
    )
    ap.add_argument("--git-sample-repos", type=int, default=64)
    ap.add_argument("--git-sample-max-commits", type=int, default=25)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    db_path = Path(
        args.db_path or (splits_dir / "commits_assertions.db"),
    ).expanduser().resolve()
    repos_root = Path(
        args.repos_root or (splits_dir / "repositories"),
    ).expanduser().resolve()
    out_path = Path(
        args.output_json or (ROOT / "analysis" / "output" / "commit_diff_stats.json"),
    ).expanduser().resolve()

    count_tokens, token_counter_desc = _make_token_counter(args.tokenizer_model)
    version_stats = aggregate_versions_from_gru(splits_dir)

    diff_stats: Dict[str, Any]
    if db_path.exists():
        diff_stats = stats_from_sqlite(db_path, count_tokens)
    else:
        diff_stats = {"n_diff_rows": 0, "source": "sqlite_missing"}

    if diff_stats.get("n_diff_rows", 0) == 0 and args.sample_git_if_empty_db:
        all_ids = _all_unique_repo_ids(splits_dir)
        rng = random.Random(args.seed)
        if len(all_ids) > args.git_sample_repos:
            repo_sample = rng.sample(all_ids, k=args.git_sample_repos)
        else:
            repo_sample = list(all_ids)
        repo_sample.sort()
        diff_stats = sample_git_diff_tokens(
            repos_root,
            count_tokens,
            repo_sample,
            args.git_sample_max_commits,
            args.seed,
        )

    payload = {
        "splits_dir": str(splits_dir),
        "db_path": str(db_path),
        "repos_root": str(repos_root),
        "token_counter": token_counter_desc,
        "sampled_versions_per_repo": version_stats,
        "unified_diff_tokens": diff_stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(version_stats, indent=2))
    print(json.dumps(diff_stats, indent=2))


if __name__ == "__main__":
    main()
