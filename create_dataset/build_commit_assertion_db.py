#!/usr/bin/env python3
"""
Build a SQLite database for commit-level Code2LoRA-GRU training.

Two-table schema:
  commits(repo_id, commit_index, commit_sha, production_code_diff)
  assertions(repo_id, commit_index, assertion_prefix, assertion_target)

Compared to v1:
  - No production_code column (commit-level training only uses diffs).
  - Blame-based assertion→commit mapping for intermediate supervision.
  - Efficient git ops: one `git log` per repo for the commit list, one
    `git diff` per sampled commit (no per-file git show).
  - Idempotent assertion insertion (deletes before re-insert).
  - Two diff modes: single_commit (diff vs immediate parent, one commit's
    change) and inter_sample (diff vs previous *sampled* commit, capturing
    all accumulated changes between samples).

Usage:
    python create_dataset/build_commit_assertion_db.py \\
        --db-path $SCRATCH/REPO_DATASET/commits_assertions.db
    python create_dataset/build_commit_assertion_db.py --limit-repos 5
    python create_dataset/build_commit_assertion_db.py --diff-mode inter_sample
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from extract_commit_history import _run_git

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_EXTS = {".py", ".md", ".rst"}
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".tox", ".mypy_cache",
    "TEST_HYPERNET",
}
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
QNA_FILENAME = "QNA_HYPERNET.json"
DEFAULT_SPLIT_NAMES = [
    "train.json", "cr_val.json", "cr_test.json",
    "ir_val.json", "ir_test.json",
]


# ---------------------------------------------------------------------------
# Path filtering
# ---------------------------------------------------------------------------

def _is_test_path(path: str) -> bool:
    for part in Path(path).parts:
        if "test" in part.lower():
            return True
    return False


def _should_skip_diff_path(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in SOURCE_EXTS:
        return True
    if any(part in SKIP_DIRS for part in p.parts):
        return True
    if _is_test_path(path):
        return True
    return False


# ---------------------------------------------------------------------------
# Git operations (efficient — minimal subprocess spawning)
# ---------------------------------------------------------------------------

def get_commit_list(repo_dir: Path) -> List[Dict[str, Any]]:
    """Chronological first-parent commit list in a single git call.

    Returns [{sha, parent_sha, timestamp}, ...] oldest-first.
    """
    out = _run_git(
        repo_dir,
        ["log", "--first-parent", "--reverse", "--format=%H%x00%P%x00%aI"],
        timeout=300,
    )
    if not out:
        return []
    commits: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = line.strip().split("\x00", 2)
        if not parts or len(parts[0]) != 40:
            continue
        sha = parts[0]
        parents = parts[1].split() if len(parts) > 1 and parts[1] else []
        parent_sha = parents[0] if parents else None
        timestamp = parts[2] if len(parts) > 2 else ""
        commits.append({
            "sha": sha,
            "parent_sha": parent_sha,
            "timestamp": timestamp,
        })
    return commits


def sample_commit_indices(n_total: int, max_commits: int) -> List[int]:
    """Uniform sample of commit indices, always including first and last."""
    if n_total <= max_commits:
        return list(range(n_total))
    indices: Set[int] = {0, n_total - 1}
    step = max(1.0, n_total / max_commits)
    pos = 0.0
    while len(indices) < max_commits and pos < n_total:
        indices.add(int(pos))
        pos += step
    return sorted(indices)


def get_diff(
    repo_dir: Path,
    parent_sha: Optional[str],
    child_sha: str,
    timeout: int = 600,
) -> str:
    """Unified diff between two commits (or empty-tree → child)."""
    base = parent_sha if parent_sha else EMPTY_TREE_SHA
    out = _run_git(
        repo_dir,
        ["diff", "--no-color", "-U3", base, child_sha],
        timeout=timeout,
    )
    return out if out else ""


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)(?:\n|$)", re.MULTILINE)


def filter_diff(raw_diff: str) -> str:
    """Keep only hunks for non-test, source-ext files outside skip dirs."""
    if not raw_diff:
        return ""
    chunks = re.split(r"(?=^diff --git )", raw_diff, flags=re.MULTILINE)
    kept: List[str] = []
    for chunk in chunks:
        if not chunk.startswith("diff --git "):
            continue
        m = _DIFF_HEADER_RE.match(chunk)
        if not m:
            continue
        path_b = m.group(2)
        if _should_skip_diff_path(path_b):
            continue
        kept.append(chunk)
    return "".join(kept)


# ---------------------------------------------------------------------------
# Git blame → commit-index mapping
# ---------------------------------------------------------------------------

def run_blame(
    repo_dir: Path,
    commit_sha: str,
    file_path: str,
    timeout: int = 120,
) -> Dict[int, str]:
    """Run ``git blame --line-porcelain`` → {final_lineno: introducing_sha}."""
    out = _run_git(
        repo_dir,
        ["blame", "--line-porcelain", commit_sha, "--", file_path],
        timeout=timeout,
    )
    if not out:
        return {}
    result: Dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or len(parts[0]) != 40:
            continue
        if not all(c in "0123456789abcdef" for c in parts[0]):
            continue
        try:
            final_lineno = int(parts[2])
        except (ValueError, IndexError):
            continue
        result[final_lineno] = parts[0]
    return result


def _find_sampled_index(sampled_indices: List[int], orig_idx: int) -> int:
    """First sampled index whose original position >= orig_idx."""
    for si, ci in enumerate(sampled_indices):
        if ci >= orig_idx:
            return si
    return len(sampled_indices) - 1


def build_blame_map(
    repo_dir: Path,
    last_sampled_sha: str,
    qna_metadata: Dict[Tuple[str, str], Tuple[str, int]],
    full_sha_to_idx: Dict[str, int],
    sampled_indices: List[int],
) -> Dict[Tuple[str, str], int]:
    """Map (prefix_key, target_key) → sampled commit_index via git blame.

    Args:
        qna_metadata:    {(prefix_key, target_key): (test_file, lineno)}
        full_sha_to_idx: {sha: index_in_full_history}
        sampled_indices: original indices of the sampled commits

    Returns:
        {(prefix_key, target_key): sampled_commit_index}
    """
    by_file: Dict[str, List[Tuple[Tuple[str, str], int]]] = defaultdict(list)
    for key, (file_path, lineno) in qna_metadata.items():
        by_file[file_path].append((key, lineno))

    blame_map: Dict[Tuple[str, str], int] = {}

    for file_path, entries in by_file.items():
        line_shas = run_blame(repo_dir, last_sampled_sha, file_path)
        if not line_shas:
            continue
        for key, lineno in entries:
            blame_sha = line_shas.get(lineno)
            if not blame_sha:
                continue
            orig_idx = full_sha_to_idx.get(blame_sha)
            if orig_idx is None:
                continue
            blame_map[key] = _find_sampled_index(sampled_indices, orig_idx)

    return blame_map


# ---------------------------------------------------------------------------
# QNA metadata loading
# ---------------------------------------------------------------------------

def repo_dir_for_name(repos_root: Path, repo_name: str) -> Path:
    parts = repo_name.split("/")
    if len(parts) == 2:
        return repos_root / parts[0] / parts[1]
    return repos_root / repo_name


def _qna_key(prefix: str, target: str) -> Tuple[str, str]:
    """Deterministic lookup key (truncated to avoid huge-string hashing)."""
    return (prefix.strip()[:200], target.strip()[:200])


def load_qna_metadata(repo_dir: Path) -> Dict[Tuple[str, str], Tuple[str, int]]:
    """Load (prefix_key, target_key) → (file, lineno) from QNA_HYPERNET.json."""
    qna_path = repo_dir / QNA_FILENAME
    if not qna_path.exists():
        return {}
    try:
        data = json.loads(qna_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for pair in data.get("pairs", []):
        prefix = (pair.get("prefix") or "").strip()
        target = (pair.get("target") or "").strip()
        meta = pair.get("metadata", {})
        file_path = meta.get("file", "")
        lineno = meta.get("lineno")
        if prefix and target and file_path and lineno is not None:
            result[_qna_key(prefix, target)] = (file_path, int(lineno))
    return result


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS commits (
            repo_id            TEXT    NOT NULL,
            commit_index       INTEGER NOT NULL,
            commit_sha         TEXT    NOT NULL,
            production_code_diff TEXT  NOT NULL,
            PRIMARY KEY (repo_id, commit_index)
        );

        CREATE TABLE IF NOT EXISTS assertions (
            repo_id          TEXT    NOT NULL,
            commit_index     INTEGER NOT NULL,
            assertion_prefix TEXT    NOT NULL,
            assertion_target TEXT    NOT NULL,
            FOREIGN KEY (repo_id, commit_index)
                REFERENCES commits (repo_id, commit_index)
        );

        CREATE INDEX IF NOT EXISTS idx_assertions_repo_commit
            ON assertions (repo_id, commit_index);
        """
    )


# ---------------------------------------------------------------------------
# Split-file helpers
# ---------------------------------------------------------------------------

def discover_split_paths(splits_dir: Path, use_gru: bool) -> List[Path]:
    if use_gru:
        gru = sorted(splits_dir.glob("gru_*.json"))
        if gru:
            return gru
    paths: List[Path] = []
    for name in DEFAULT_SPLIT_NAMES:
        p = splits_dir / name
        if p.exists():
            paths.append(p)
    return paths


def load_split(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build commits + assertions SQLite DB (v2, optimized)",
    )
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_root)
    ap.add_argument("--repos-root", type=str, default=None)
    ap.add_argument(
        "--db-path", type=str, default=None,
        help="Output .db path (default: <splits-dir>/commits_assertions.db)",
    )
    ap.add_argument("--use-gru-splits", action="store_true")
    ap.add_argument("--no-gru-splits", action="store_true")
    ap.add_argument("--max-commits", type=int, default=200)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument(
        "--diff-mode", type=str, default="single_commit",
        choices=["single_commit", "inter_sample"],
        help=(
            "single_commit: diff vs immediate first-parent (one commit's "
            "change). inter_sample: diff vs previous sampled commit "
            "(captures all accumulated changes between samples)."
        ),
    )
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = Path(
        args.repos_root or (splits_dir / "repositories"),
    ).expanduser().resolve()
    db_path = Path(
        args.db_path or (splits_dir / "commits_assertions.db"),
    ).expanduser().resolve()

    use_gru = not args.no_gru_splits
    if args.use_gru_splits:
        use_gru = True

    split_paths = discover_split_paths(splits_dir, use_gru=use_gru)
    if not split_paths:
        raise SystemExit(f"No split JSON files found under {splits_dir}")

    # Collect unique repo IDs across all splits
    repo_ids: Set[str] = set()
    for sp in split_paths:
        data = load_split(sp)
        repo_ids.update(data.get("repositories", {}).keys())
    repo_list = sorted(repo_ids)
    if args.limit_repos:
        repo_list = repo_list[: args.limit_repos]

    # ── Open DB ──
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        cur = conn.cursor()

        print(f"Database:    {db_path}")
        print(f"Repos root:  {repos_root}")
        print(f"Splits:      {[p.name for p in split_paths]}")
        print(f"Diff mode:   {args.diff_mode}")
        print(f"Max commits: {args.max_commits}")
        print(f"Repos:       {len(repo_list)}")

        already_done = set(
            row[0] for row in
            cur.execute("SELECT DISTINCT repo_id FROM commits").fetchall()
        )

        # ==================================================================
        # Phase 1: commits table + blame maps
        # ==================================================================
        blame_maps: Dict[str, Dict[Tuple[str, str], int]] = {}
        last_sampled_index: Dict[str, int] = {}

        # Pre-populate last_sampled_index for repos already in DB
        for repo_id in already_done:
            row = cur.execute(
                "SELECT MAX(commit_index) FROM commits WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
            if row and row[0] is not None:
                last_sampled_index[repo_id] = row[0]

        n_skipped = 0
        n_errors = 0

        for ri, repo_id in enumerate(tqdm(repo_list, desc="commits")):
            if repo_id in already_done:
                n_skipped += 1
                continue

            rdir = repo_dir_for_name(repos_root, repo_id)
            if not rdir.is_dir() or not (rdir / ".git").exists():
                print(f"  skip (no git): {repo_id}", flush=True)
                continue

            try:
                full_commits = get_commit_list(rdir)
                if not full_commits:
                    print(f"  skip (no commits): {repo_id}", flush=True)
                    continue

                sampled_indices = sample_commit_indices(
                    len(full_commits), args.max_commits,
                )
                full_sha_to_idx = {
                    c["sha"]: i for i, c in enumerate(full_commits)
                }

                # ── Insert diffs ──
                prev_sampled_sha: Optional[str] = None
                for si, ci in enumerate(sampled_indices):
                    c = full_commits[ci]
                    sha = c["sha"]

                    if args.diff_mode == "inter_sample":
                        parent = prev_sampled_sha
                    else:
                        parent = c["parent_sha"]

                    raw_diff = get_diff(rdir, parent, sha)
                    filtered = filter_diff(raw_diff)

                    cur.execute(
                        "INSERT OR REPLACE INTO commits "
                        "(repo_id, commit_index, commit_sha, production_code_diff) "
                        "VALUES (?, ?, ?, ?)",
                        (repo_id, si, sha, filtered),
                    )
                    prev_sampled_sha = sha

                last_si = len(sampled_indices) - 1
                last_sampled_index[repo_id] = last_si

                # ── Blame map ──
                qna_meta = load_qna_metadata(rdir)
                if qna_meta:
                    last_sha = full_commits[sampled_indices[-1]]["sha"]
                    bmap = build_blame_map(
                        rdir, last_sha, qna_meta,
                        full_sha_to_idx, sampled_indices,
                    )
                    if bmap:
                        blame_maps[repo_id] = bmap

            except Exception as exc:
                n_errors += 1
                print(f"  ERROR on {repo_id}: {exc!r}", flush=True)

            if (ri + 1) % 50 == 0:
                conn.commit()

        conn.commit()
        print(
            f"  Commits done. errors={n_errors}/{len(repo_list)}, "
            f"skipped={n_skipped}"
        )

        # ==================================================================
        # Phase 2: assertions table (idempotent — delete then re-insert)
        # ==================================================================
        cur.execute("DELETE FROM assertions")
        conn.commit()

        n_assertions = 0
        n_blame_mapped = 0
        n_fallback = 0

        for sp in split_paths:
            data = load_split(sp)
            repos = data.get("repositories", {})
            for repo_id, rdata in repos.items():
                if repo_id not in last_sampled_index:
                    continue
                default_ci = last_sampled_index[repo_id]
                bmap = blame_maps.get(repo_id, {})
                pairs = rdata.get("qna_pairs") or []

                for q in pairs:
                    prefix = (q.get("prefix") or "").strip()
                    target = (q.get("target") or "").strip()
                    if not prefix or not target:
                        continue
                    if target.lstrip().startswith(","):
                        continue

                    key = _qna_key(prefix, target)
                    ci = bmap.get(key)
                    if ci is not None:
                        n_blame_mapped += 1
                    else:
                        ci = default_ci
                        n_fallback += 1

                    cur.execute(
                        "INSERT INTO assertions "
                        "(repo_id, commit_index, assertion_prefix, assertion_target) "
                        "VALUES (?, ?, ?, ?)",
                        (repo_id, ci, prefix, target),
                    )
                    n_assertions += 1

        conn.commit()

        # ── Summary ──
        n_commits = cur.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        n_repos_in_db = cur.execute(
            "SELECT COUNT(DISTINCT repo_id) FROM commits",
        ).fetchone()[0]
        print(f"\nRows: commits={n_commits}, assertions={n_assertions}")
        print(f"Repos in DB: {n_repos_in_db}")
        print(
            f"Blame: {n_blame_mapped} assertions mapped to introducing commit, "
            f"{n_fallback} fell back to last commit "
            f"({len(blame_maps)} repos had blame data)"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
