#!/usr/bin/env python3
"""
For each repo: detect test files, move them to TEST_HYPERNET/, create REPO_METADATA.json.

Test file = contains tests (pytest/unittest/test_*) OR has "test" in path.
"""

import argparse
import ast
import os
import re
import shutil
import json
import warnings
from pathlib import Path
from tqdm import tqdm

REPOSITORIES_DIR = "/home/lhotsko/scratch/REPO_DATASET/repositories"
TEST_HYPERNET = "TEST_HYPERNET"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn", "site-packages",
}


def is_test_by_path(rel_path: str) -> bool:
    """True if path/name suggests a test file."""
    parts = Path(rel_path).parts
    for p in parts:
        if "test" in p.lower():
            return True
    return False


def contains_test_content(text: str) -> bool:
    """True if file uses pytest, unittest, or defines test_* functions."""
    try:
        low = text.lower()
        if "import pytest" in low or "from pytest" in low or "pytest." in low:
            return True
        if "import unittest" in low or "from unittest" in low or "unittest." in low:
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    return True
            if isinstance(node, ast.ClassDef):
                if "test" in node.name.lower():
                    return True
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("test_"):
                            return True
    except (SyntaxError, ValueError):
        pass
    return False


def collect_py_files(repo_path: str) -> list[tuple[str, str]]:
    """Return [(rel_path, abs_path), ...] for all .py files (excludes TEST_HYPERNET)."""
    results = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.startswith(TEST_HYPERNET):
            continue
        for f in files:
            if f.endswith(".py"):
                rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                abs_path = os.path.join(root, f)
                results.append((rel_path, abs_path))
    return results


def detect_test_files(repo_path: str) -> tuple[list[str], list[str]]:
    """Returns (test_file_rel_paths, regular_file_rel_paths)."""
    py_files = collect_py_files(repo_path)
    test_files = []
    regular_files = []

    for rel_path, abs_path in py_files:
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        has_test_content = contains_test_content(text)
        has_test_path = is_test_by_path(rel_path)
        if has_test_content or has_test_path:
            test_files.append(rel_path)
        else:
            regular_files.append(rel_path)

    return test_files, regular_files


def count_assertions_in_files(repo_path: str, rel_paths: list[str]) -> int:
    """Count assert statements in the given files."""
    count = 0
    for rel_path in rel_paths:
        abs_path = os.path.join(repo_path, rel_path)
        if not os.path.exists(abs_path):
            continue
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
            count += len(re.findall(r"\bassert\b", text))
            count += len(re.findall(r"self\.assert", text))
        except Exception:
            pass
    return count


def get_dir_size(path: str) -> int:
    """Total size of directory in bytes."""
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def process_repo(repo_path: str, author: str, repo_name: str) -> dict | None:
    """Process one repo: detect tests, move to TEST_HYPERNET, return metadata."""
    test_files, regular_files = detect_test_files(repo_path)
    if not test_files:
        return None

    test_hypernet_dir = os.path.join(repo_path, TEST_HYPERNET)
    os.makedirs(test_hypernet_dir, exist_ok=True)

    moved = []
    for rel_path in test_files:
        src = os.path.join(repo_path, rel_path)
        if not os.path.exists(src):
            continue
        dest = os.path.join(test_hypernet_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.move(src, dest)
            moved.append(rel_path)
        except (OSError, shutil.Error):
            pass

    # Count assertions in moved files (they're now in TEST_HYPERNET)
    assertions = count_assertions_in_files(test_hypernet_dir, moved)

    # Extension stats for regular files
    ext_counts = {}
    for rp in regular_files:
        ext = Path(rp).suffix or "no_ext"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    repo_size = get_dir_size(repo_path)
    metadata = {
        "repo_name": repo_name,
        "repo_full_name": f"{author}/{repo_name}",
        "repo_url": f"https://github.com/{author}/{repo_name}",
        "repo_size_bytes": repo_size,
        "repo_size_kb": round(repo_size / 1024, 2),
        "number_of_test_files": len(moved),
        "test_files": moved,
        "number_of_regular_files": len(regular_files),
        "regular_files_by_extension": ext_counts,
        "number_of_assertions_in_test_files": assertions,
        "embedding": None,
    }

    meta_path = os.path.join(repo_path, "REPO_METADATA.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Separate test files into TEST_HYPERNET")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N repos")
    parser.add_argument("--repo", type=str, default=None, help="Process single repo: owner/repo_name")
    args = parser.parse_args()

    if not os.path.isdir(REPOSITORIES_DIR):
        print(f"Repositories dir not found: {REPOSITORIES_DIR}")
        return

    processed = 0
    skipped_repos = []
    repos_iter = []

    if args.repo:
        parts = args.repo.split("/", 1)
        if len(parts) != 2:
            print("--repo must be owner/repo_name")
            return
        author, repo_name = parts
        repo_path = os.path.join(REPOSITORIES_DIR, author, repo_name)
        if not os.path.isdir(repo_path):
            print(f"Repo not found: {repo_path}")
            return
        repos_iter = [(author, repo_name, repo_path)]
    else:
        for author in os.listdir(REPOSITORIES_DIR):
            author_path = os.path.join(REPOSITORIES_DIR, author)
            if not os.path.isdir(author_path):
                continue
            for repo_name in os.listdir(author_path):
                repo_path = os.path.join(author_path, repo_name)
                if not os.path.isdir(repo_path) or repo_name == TEST_HYPERNET:
                    continue
                repos_iter.append((author, repo_name, repo_path))
                if args.limit and len(repos_iter) >= args.limit:
                    break
            if args.limit and len(repos_iter) >= args.limit:
                break

    for author, repo_name, repo_path in tqdm(repos_iter, desc="Repos"):
        result = process_repo(repo_path, author, repo_name)
        if result:
            processed += 1
            print(f"  {author}/{repo_name}: {result['number_of_test_files']} test files moved")
        else:
            skipped_repos.append(f"{author}/{repo_name}")

    print(f"\nProcessed: {processed}, Skipped (no test files): {len(skipped_repos)}")
    if skipped_repos:
        print("\nSkipped repos:")
        for r in skipped_repos:
            print(f"  {r}")


if __name__ == "__main__":
    main()
