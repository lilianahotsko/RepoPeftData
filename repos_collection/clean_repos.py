#!/usr/bin/env python3
"""
Download repositories from pytest_repos_5k.jsonl, clone them,
and split files into test_files/ and regular_files/ directories.

Output structure:
    scratch/pulled_repos/<owner>/<repo>/
        test_files/      — files/dirs matching test patterns
        regular_files/   — everything else
"""

import os
import json
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_repos_5k.jsonl")
OUTPUT_ROOT = "/home/lhotsko/scratch/pulled_repos"

# Directories to skip entirely when walking repos
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn",
}

# File extensions to skip (binaries / bytecode)
SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe"}


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def is_test_item(name: str) -> bool:
    """Return True if a file or directory name is test-related.

    Matches:
      - anything with 'test' in the name (test_*, *_test.py, tests/, …)
      - conftest.py  (already caught by the 'test' substring)
    """
    return "test" in name.lower()


def should_skip_item(name: str) -> bool:
    """Items that should be excluded from both test and regular trees."""
    if name.startswith("."):
        return True
    if name in SKIP_DIRS:
        return True
    return False


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------

def clone_repo(url: str, dest: str, timeout: int = 300) -> bool:
    """Shallow-clone a repository. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, dest],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Kill the lingering process tree if possible
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Walk & split
# ---------------------------------------------------------------------------

def walk_and_copy(src_root: str, test_dest: str, regular_dest: str):
    """Walk the cloned repo and copy every file into test_files/ or regular_files/.

    Directory structure is preserved relative to the repo root.
    """
    test_count = 0
    regular_count = 0

    for dirpath, dirnames, filenames in os.walk(src_root):
        # Prune directories we don't care about (in-place)
        dirnames[:] = [d for d in dirnames if not should_skip_item(d)]

        rel_dir = os.path.relpath(dirpath, src_root)
        parts = Path(rel_dir).parts if rel_dir != "." else ()

        # Are we already inside a test directory?
        in_test_dir = any(is_test_item(p) for p in parts)

        for fname in filenames:
            if should_skip_item(fname):
                continue
            _, ext = os.path.splitext(fname)
            if ext in SKIP_EXTENSIONS:
                continue

            src_file = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(src_file, src_root)

            if in_test_dir or is_test_item(fname):
                dest_file = os.path.join(test_dest, rel_path)
                test_count += 1
            else:
                dest_file = os.path.join(regular_dest, rel_path)
                regular_count += 1

            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            try:
                shutil.copy2(src_file, dest_file)
            except (PermissionError, OSError):
                pass

    return test_count, regular_count


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(repo_info: dict, output_root: str, tmp_dir: str):
    """Clone a single repository and split its files.

    Returns (status, test_file_count, regular_file_count).
    status is one of: 'success', 'skipped', 'failed'.
    """
    full_name = repo_info["full_name"]  # e.g. "owner/repo"

    repo_output = os.path.join(output_root, full_name)
    test_dest = os.path.join(repo_output, "test_files")
    regular_dest = os.path.join(repo_output, "regular_files")

    # Skip if both subdirectories already exist (resume-friendly)
    if os.path.isdir(test_dest) and os.path.isdir(regular_dest):
        return "skipped", 0, 0

    # Clone into a temp directory
    clone_dest = os.path.join(tmp_dir, full_name.replace("/", "__"))
    if os.path.exists(clone_dest):
        shutil.rmtree(clone_dest)

    clone_url = f"https://github.com/{full_name}.git"
    if not clone_repo(clone_url, clone_dest):
        return "failed", 0, 0

    # Remove any partial previous run
    if os.path.exists(repo_output):
        shutil.rmtree(repo_output)

    os.makedirs(test_dest, exist_ok=True)
    os.makedirs(regular_dest, exist_ok=True)

    # Walk the clone and split
    test_count, regular_count = walk_and_copy(clone_dest, test_dest, regular_dest)

    # Clean up clone to free disk space
    shutil.rmtree(clone_dest, ignore_errors=True)

    return "success", test_count, regular_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clone repos from JSONL and split into test/regular files"
    )
    parser.add_argument(
        "--jsonl", default=JSONL_PATH,
        help="Path to JSONL file with repo metadata (default: pytest_repos_5k.jsonl)",
    )
    parser.add_argument(
        "--output", default=OUTPUT_ROOT,
        help="Root output directory (default: /home/lhotsko/scratch/pulled_repos)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N repos (useful for testing)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Git clone timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # ---- Load repo list ----
    repos = []
    with open(args.jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                repos.append(json.loads(line))

    if args.limit:
        repos = repos[: args.limit]

    print(f"Repositories to process : {len(repos)}")
    print(f"Output directory        : {args.output}")
    print()

    os.makedirs(args.output, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_test = 0
    total_regular = 0

    with tempfile.TemporaryDirectory(prefix="repo_clone_") as tmp_dir:
        pbar = tqdm(repos, desc="Processing repos")
        for repo in pbar:
            status, tc, rc = process_repo(repo, args.output, tmp_dir)

            if status == "success":
                success_count += 1
                total_test += tc
                total_regular += rc
            elif status == "skipped":
                skip_count += 1
            else:
                fail_count += 1

            pbar.set_postfix(
                ok=success_count, skip=skip_count, fail=fail_count, ordered=True
            )

    print(f"\n{'=' * 40}")
    print(f"  Cloned successfully : {success_count}")
    print(f"  Skipped (exists)    : {skip_count}")
    print(f"  Failed              : {fail_count}")
    print(f"  Total test files    : {total_test}")
    print(f"  Total regular files : {total_regular}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
