#!/usr/bin/env python3
"""
Extract commit history for Code2LoRA-GRU: walk git log per repository,
record per-commit file snapshots, map existing QnA pairs to the latest
commit, define a preamble cutoff, and write an extended JSON split.

For each cloned repository (full history at $SCRATCH/REPO_DATASET/repositories/):
  1. git log --first-parent to get chronological commit list
  2. For each commit: record which non-test source files exist
  3. Track when each file first appeared (first_commit_idx)
  4. Map QnA pairs to the HEAD commit (they describe tests at HEAD)
  5. Define preamble as files present in the first 10% of commits
  6. Produce extended split JSON with commit history metadata

Usage:
    python create_dataset/extract_commit_history.py
    python create_dataset/extract_commit_history.py --splits-dir $SCRATCH/REPO_DATASET
    python create_dataset/extract_commit_history.py --preamble-frac 0.1 --max-commits 200
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".tox", ".mypy_cache",
    "TEST_HYPERNET",
}
SOURCE_EXTS = {".py", ".md", ".rst"}
MAX_FILE_BYTES = 2_000_000


def _is_test_path(path: str) -> bool:
    """Heuristic: skip files whose path contains 'test' (matching 2_separate_tests.py)."""
    for part in Path(path).parts:
        if "test" in part.lower():
            return True
    return False


def _run_git(repo_dir: Path, args: List[str], timeout: int = 120) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir)] + args,
            capture_output=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", errors="replace")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_commit_list(repo_dir: Path) -> List[Dict[str, str]]:
    """Return chronological list of commits [{sha, timestamp}, ...] (oldest first)."""
    out = _run_git(repo_dir, [
        "log", "--first-parent", "--reverse",
        "--format=%H %aI",
    ])
    if not out:
        return []
    commits = []
    for line in out.strip().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            commits.append({"sha": parts[0], "timestamp": parts[1]})
    return commits


def get_files_at_commit(repo_dir: Path, sha: str) -> List[str]:
    """List non-test tracked source files at a given commit.

    Applies the same filtering as embed_repos.py (SKIP_DIRS, SOURCE_EXTS)
    and the same test-path heuristic as 2_separate_tests.py so that only
    non-test source files are included in the GRU file sequence.
    """
    out = _run_git(repo_dir, ["ls-tree", "-r", "--name-only", sha])
    if not out:
        return []
    files = []
    for line in out.strip().splitlines():
        path = line.strip()
        if not path:
            continue
        parts = Path(path).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        ext = Path(path).suffix.lower()
        if ext not in SOURCE_EXTS:
            continue
        if _is_test_path(path):
            continue
        files.append(path)
    return sorted(files)


def sample_commits(
    commits: List[Dict[str, str]],
    max_commits: int,
) -> List[int]:
    """Return indices of sampled commits. Always includes first and last."""
    n = len(commits)
    if n <= max_commits:
        return list(range(n))
    indices = set()
    indices.add(0)
    indices.add(n - 1)
    step = max(1, n / max_commits)
    pos = 0.0
    while len(indices) < max_commits and pos < n:
        indices.add(int(pos))
        pos += step
    return sorted(indices)


def extract_repo_history(
    repo_dir: Path,
    max_commits: int = 200,
    preamble_frac: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Extract commit history for a single repository.

    Returns dict with commits, file_first_seen, preamble_commit_cutoff, etc.
    Returns None on failure.
    """
    commits = get_commit_list(repo_dir)
    if not commits:
        return None

    sampled_indices = sample_commits(commits, max_commits)
    sampled_commits = []
    all_files_ever: Dict[str, int] = {}  # path -> first_commit_idx (in sampled list)

    for si, ci in enumerate(sampled_indices):
        c = commits[ci]
        files = get_files_at_commit(repo_dir, c["sha"])

        new_files = []
        for f in files:
            if f not in all_files_ever:
                all_files_ever[f] = si
                new_files.append(f)

        sampled_commits.append({
            "original_idx": ci,
            "sha": c["sha"],
            "timestamp": c["timestamp"],
            "files_present": files,
            "files_added": new_files,
        })

    preamble_cutoff = max(1, int(len(sampled_commits) * preamble_frac))

    preamble_files = set()
    for sc in sampled_commits[:preamble_cutoff]:
        preamble_files.update(sc["files_present"])

    return {
        "total_commits_in_repo": len(commits),
        "sampled_commit_count": len(sampled_commits),
        "commits": sampled_commits,
        "file_first_seen": all_files_ever,
        "preamble_commit_cutoff": preamble_cutoff,
        "preamble_files": sorted(preamble_files),
    }


def build_gru_splits(
    splits_dir: Path,
    repos_root: Path,
    output_dir: Path,
    max_commits: int = 200,
    preamble_frac: float = 0.1,
    limit_repos: Optional[int] = None,
):
    """Read existing split JSONs, augment each repo with commit history, write new splits."""
    split_names = ["train.json", "cr_val.json", "cr_test.json", "ir_val.json", "ir_test.json"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in split_names:
        split_path = splits_dir / split_name
        if not split_path.exists():
            print(f"  Skipping {split_name} (not found)")
            continue

        print(f"\nProcessing {split_name}...")
        data = json.loads(split_path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})
        repo_names = sorted(repos.keys())
        if limit_repos:
            repo_names = repo_names[:limit_repos]

        augmented_repos = {}
        n_ok = 0
        n_fail = 0

        for repo_name in tqdm(repo_names, desc=f"  {split_name}"):
            rdata = repos[repo_name]
            parts = repo_name.split("/")
            if len(parts) == 2:
                repo_dir = repos_root / parts[0] / parts[1]
            else:
                repo_dir = repos_root / repo_name

            history = None
            if repo_dir.exists() and (repo_dir / ".git").exists():
                history = extract_repo_history(
                    repo_dir, max_commits=max_commits, preamble_frac=preamble_frac,
                )

            entry = {
                "qna_pairs": rdata.get("qna_pairs", []),
                "embedding": rdata.get("embedding"),
            }

            if rdata.get("file_embeddings") is not None:
                entry["file_embeddings"] = rdata["file_embeddings"]

            if history is not None:
                n_ok += 1
                last_commit_idx = len(history["commits"]) - 1

                fe_paths = set()
                if entry.get("file_embeddings"):
                    fe_paths = {fe["path"] for fe in entry["file_embeddings"]}

                file_order = []
                for path, first_idx in sorted(
                    history["file_first_seen"].items(), key=lambda x: x[1]
                ):
                    file_order.append({
                        "path": path,
                        "first_commit_idx": first_idx,
                        "has_embedding": path in fe_paths,
                    })

                entry["commit_history"] = {
                    "total_commits_in_repo": history["total_commits_in_repo"],
                    "sampled_commit_count": history["sampled_commit_count"],
                    "preamble_commit_cutoff": history["preamble_commit_cutoff"],
                    "preamble_files": history["preamble_files"],
                    "file_order": file_order,
                    "commits": history["commits"],
                }

                for qna in entry["qna_pairs"]:
                    qna["commit_idx"] = last_commit_idx
            else:
                n_fail += 1
                entry["commit_history"] = None

            augmented_repos[repo_name] = entry

        out_data = {"repositories": augmented_repos}
        out_path = output_dir / f"gru_{split_name}"
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        total_pairs = sum(
            len(r["qna_pairs"]) for r in augmented_repos.values()
        )
        n_with_history = sum(
            1 for r in augmented_repos.values()
            if r.get("commit_history") is not None
        )
        print(f"  {split_name}: {len(augmented_repos)} repos "
              f"({n_with_history} with history, {n_fail} without), "
              f"{total_pairs} QnA pairs -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract commit history for Code2LoRA-GRU dataset"
    )
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Directory containing train.json, cr_val.json, etc.")
    ap.add_argument("--repos-root", type=str, default=None,
                    help="Root of repositories/ (default: splits-dir/repositories)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output dir for gru_*.json splits (default: splits-dir)")
    ap.add_argument("--max-commits", type=int, default=200,
                    help="Max commits to sample per repo")
    ap.add_argument("--preamble-frac", type=float, default=0.1,
                    help="Fraction of early commits defining the preamble")
    ap.add_argument("--limit-repos", type=int, default=None)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = Path(args.repos_root or (splits_dir / "repositories")).expanduser().resolve()
    output_dir = Path(args.output_dir or splits_dir).expanduser().resolve()

    print(f"Splits dir:  {splits_dir}")
    print(f"Repos root:  {repos_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Max commits: {args.max_commits}")
    print(f"Preamble %:  {args.preamble_frac * 100:.0f}%")

    build_gru_splits(
        splits_dir=splits_dir,
        repos_root=repos_root,
        output_dir=output_dir,
        max_commits=args.max_commits,
        preamble_frac=args.preamble_frac,
        limit_repos=args.limit_repos,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
