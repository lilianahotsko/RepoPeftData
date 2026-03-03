#!/usr/bin/env python3
"""
Calculate distribution of assert methods across all repos (TEST_HYPERNET files).
Also: test files, test functions, assertions per function, etc.
"""

import ast
import json
import os
import re
import warnings
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

REPOSITORIES_DIR = "/home/lhotsko/scratch/REPO_DATASET/repositories"
TEST_HYPERNET = "TEST_HYPERNET"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn", "site-packages",
}

# Known unittest methods (for self.assert*)
SELF_ASSERT_METHODS = {
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn", "assertIsInstance", "assertNotIsInstance",
    "assertRaises", "assertRaisesRegex", "assertAlmostEqual", "assertNotAlmostEqual",
    "assertDictEqual", "assertListEqual", "assertTupleEqual", "assertSetEqual",
    "assertSequenceEqual", "assertRegex", "assertNotRegex", "assertCountEqual",
    "assertWarns", "assertLogs", "assertMultiLineEqual",
    "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual",
}


def count_test_functions_in_text(text: str) -> int:
    """Count test_* functions (standalone and in test classes)."""
    count = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    count += 1
            if isinstance(node, ast.ClassDef):
                if "test" in node.name.lower():
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name.startswith("test_"):
                                count += 1
    except (SyntaxError, ValueError):
        pass
    return count


def count_asserts_in_text(text: str) -> dict[str, int]:
    """Count each assert method type in text. Returns {method_name: count}."""
    counts = defaultdict(int)
    # Plain assert
    counts["assert"] = len(re.findall(r"\bassert\s+", text))
    # self.assert* - extract method name
    for m in re.finditer(r"self\.assert(\w+)\b", text):
        method = "self.assert" + m.group(1)
        if m.group(1) in SELF_ASSERT_METHODS:
            counts[method] += 1
        else:
            counts["self.assert*"] += 1
    # pytest
    counts["pytest.raises"] = len(re.findall(r"pytest\.raises\b", text))
    counts["pytest.approx"] = len(re.findall(r"pytest\.approx\b", text))
    # numpy/testing assert_*
    counts["assert_*"] = len(re.findall(r"\bassert_\w+\b", text))
    return dict(counts)


def iter_test_files(repos_root: str):
    """Yield (author, repo_name, file_path) for each .py file in TEST_HYPERNET."""
    for author in os.listdir(repos_root):
        author_path = os.path.join(repos_root, author)
        if not os.path.isdir(author_path):
            continue
        for repo_name in os.listdir(author_path):
            repo_path = os.path.join(author_path, repo_name)
            test_dir = os.path.join(repo_path, TEST_HYPERNET)
            if not os.path.isdir(test_dir):
                continue
            for root, dirs, files in os.walk(test_dir):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for f in files:
                    if f.endswith(".py"):
                        yield author, repo_name, os.path.join(root, f)




def main():
    if not os.path.isdir(REPOSITORIES_DIR):
        print(f"Repositories dir not found: {REPOSITORIES_DIR}")
        return

    # Per-repo counts: repo -> {method: count}
    repo_counts = defaultdict(lambda: defaultdict(int))
    # Per-repo: test files, test functions, assertions
    repo_stats = defaultdict(lambda: {"test_files": 0, "test_functions": 0, "assertions": 0})
    # Global totals
    global_counts = defaultdict(int)

    for author, repo_name, file_path in tqdm(iter_test_files(REPOSITORIES_DIR), desc="Scanning"):
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        full_name = f"{author}/{repo_name}"
        counts = count_asserts_in_text(text)
        n_funcs = count_test_functions_in_text(text)
        n_asserts = sum(counts.values())
        repo_stats[full_name]["test_files"] += 1
        repo_stats[full_name]["test_functions"] += n_funcs
        repo_stats[full_name]["assertions"] += n_asserts
        for method, n in counts.items():
            repo_counts[full_name][method] += n
            global_counts[method] += n

    # Remove repos with no asserts
    repo_counts = {r: dict(c) for r, c in repo_counts.items() if sum(c.values()) > 0}

    # Print distribution
    print("\n=== Global distribution (total across all repos) ===")
    total = sum(global_counts.values())
    for method in sorted(global_counts.keys(), key=lambda m: -global_counts[m]):
        n = global_counts[method]
        pct = 100 * n / total if total else 0
        print(f"  {method:30} {n:8,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30} {total:8,}")

    # Per-repo distribution (which methods each repo uses)
    print("\n=== Per-repo: which assert methods appear in each repo ===")
    method_usage = defaultdict(int)  # how many repos use each method
    for repo, counts in repo_counts.items():
        for method in counts:
            method_usage[method] += 1

    print("  (number of repos that use each method at least once)")
    for method in sorted(method_usage.keys(), key=lambda m: -method_usage[m]):
        print(f"  {method:30} {method_usage[method]:4} repos")

    # Total asserts per repo - exact counts and percentile stats
    repo_totals = {r: sum(c.values()) for r, c in repo_counts.items()}
    totals_sorted = sorted(repo_totals.values())
    n_repos = len(totals_sorted)

    print("\n=== Total asserts per repo (exact) ===")
    for repo, total in sorted(repo_totals.items(), key=lambda x: -x[1]):
        print(f"  {total:6,}  {repo}")

    print("\n=== Percentile stats (asserts per repo) ===")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = min(int(n_repos * p / 100), n_repos - 1) if n_repos else 0
        idx = max(0, idx)
        val = totals_sorted[idx] if totals_sorted else 0
        print(f"  {p:3}%: {val:,}")

    if totals_sorted:
        print(f"  min:  {totals_sorted[0]:,}")
        print(f"  max:  {totals_sorted[-1]:,}")
        print(f"  mean: {sum(totals_sorted) / n_repos:,.0f}")

    # Per-repo stats: test files, test functions, assertions per function
    repo_stats = dict(repo_stats)
    print("\n=== Per-repo: test files, test functions, assertions per function ===")
    for repo, s in sorted(repo_stats.items(), key=lambda x: -x[1]["test_functions"])[:15]:
        n_files = s["test_files"]
        n_funcs = s["test_functions"]
        n_asserts = s["assertions"]
        asserts_per_func = round(n_asserts / n_funcs, 2) if n_funcs else 0
        funcs_per_file = round(n_funcs / n_files, 2) if n_files else 0
        print(f"  {repo}: {n_files} files, {n_funcs} functions, {asserts_per_func} asserts/func, {funcs_per_file} funcs/file")

    # Percentile stats for test_functions and asserts_per_function
    funcs_list = sorted([s["test_functions"] for s in repo_stats.values()])
    asserts_per_func_list = []
    for s in repo_stats.values():
        if s["test_functions"] > 0:
            asserts_per_func_list.append(s["assertions"] / s["test_functions"])
    asserts_per_func_list.sort()
    n_repos_with_funcs = len(funcs_list)
    n_apf = len(asserts_per_func_list)

    print("\n=== Percentile stats: test_functions per repo ===")
    for p in percentiles:
        idx = min(int(n_repos_with_funcs * p / 100), n_repos_with_funcs - 1) if n_repos_with_funcs else 0
        val = funcs_list[idx] if funcs_list else 0
        print(f"  {p:3}%: {val:,}")
    if funcs_list:
        print(f"  mean: {sum(funcs_list) / n_repos_with_funcs:,.0f}")

    print("\n=== Percentile stats: assertions per test function ===")
    for p in percentiles:
        idx = min(int(n_apf * p / 100), n_apf - 1) if n_apf else 0
        val = round(asserts_per_func_list[idx], 2) if asserts_per_func_list else 0
        print(f"  {p:3}%: {val}")
    if asserts_per_func_list:
        print(f"  mean: {round(sum(asserts_per_func_list) / n_apf, 2)}")

    # Save to JSON (in create_dataset dir)
    output = {
        "global_distribution": dict(global_counts),
        "per_repo_counts": {r: c for r, c in repo_counts.items()},
        "per_repo_totals": repo_totals,
        "per_repo_stats": {
            r: {
                "test_files": s["test_files"],
                "test_functions": s["test_functions"],
                "assertions": s["assertions"],
                "assertions_per_test_function": round(s["assertions"] / s["test_functions"], 2) if s["test_functions"] else 0,
                "test_functions_per_file": round(s["test_functions"] / s["test_files"], 2) if s["test_files"] else 0,
            }
            for r, s in repo_stats.items()
        },
        "percentile_stats": {f"p{p}": totals_sorted[min(int(n_repos * p / 100), n_repos - 1)] if n_repos else 0 for p in percentiles},
        "method_usage_count": dict(method_usage),
    }
    out_path = Path(__file__).parent / "assert_methods_distribution.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
