#!/usr/bin/env python3

import os
import re
import json
import ast
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


DEFAULT_MIN_LINES_IN_FILE = 10
DEFAULT_MIN_LINES_IN_TEST = 4

DEFAULT_MAX_PREFIX_LINES = 300     
DEFAULT_MAX_TARGET_LINES = 10     
DEFAULT_MAX_PAIRS_PER_FILE = 5     
DEFAULT_MAX_PAIRS_PER_TEST = 3     
DEFAULT_SEED = 1234


def detect_framework(file_text: str) -> str:
    low = file_text.lower()
    has_unittest = ("import unittest" in low) or ("from unittest" in low) or ("unittest." in low)
    has_pytest = ("import pytest" in low) or ("from pytest" in low) or ("pytest." in low)
    if has_unittest and has_pytest:
        return "mixed"
    if has_unittest:
        return "unittest"
    if has_pytest:
        return "pytest"
    return "unknown"


def safe_split_lines(text: str) -> List[str]:
    return text.splitlines()


def find_test_nodes(tree: ast.AST) -> List[ast.AST]:
    tests: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "test" in node.name.lower():
                tests.append(node)
    return tests


def get_node_span_lines(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None:
        return (0, 0)
    start_i = max(0, start - 1)
    if end is None:
        end_i = start_i + 1
    else:
        end_i = max(start_i + 1, end)
    return (start_i, end_i)


def collect_test_type_tags(file_text: str) -> List[str]:
    low = file_text.lower()
    tags = []
    if "class " in low:
        tags.append("class_based")
    if "unittest.testcase" in low or "unittest" in low:
        tags.append("unittest_class")
    if "mock" in low or "patch(" in low or "magicmock" in low:
        tags.append("mock")
    if "@pytest.fixture" in low:
        tags.append("fixture_based")
    if "parametrize" in low:
        tags.append("parametrized")
    if "async def " in low or "pytest.mark.asyncio" in low:
        tags.append("async")
    return tags


def choose_cut_points_in_test(lines,test_start,test_end,max_target_lines):

    candidates = []

    def is_except_line(s: str) -> bool:
        return bool(re.match(r"^\s*except\b", s))

    def is_assert_line(s: str) -> bool:
        s2 = s.strip()
        return (
            s2.startswith("assert ")
            or "self.assert" in s2
            or "assert_" in s2 
        )

    def is_with_raises_line(s: str) -> bool:
        s2 = s.strip()
        return ("pytest.raises" in s2) or ("self.assertRaises" in s2)

    for i in range(test_start + 1, test_end):
        line = lines[i]
        if is_except_line(line):
            candidates.append((i, "except_block"))
        elif is_with_raises_line(line):
            candidates.append((i, "raises_context"))
        elif is_assert_line(line):
            candidates.append((i, "assert_block"))

    if not candidates:
        internal = list(range(test_start + 2, max(test_start + 2, test_end - 1)))
        if internal:
            random.shuffle(internal)
            for i in internal[:2]:
                candidates.append((i, "random"))

    filtered = []
    for cut_i, kind in candidates:
        if cut_i < test_end:
            filtered.append((cut_i, kind))
    return filtered


def extract_next_block(lines, cut_i, max_target_lines):
    if cut_i >= len(lines):
        return ("", 0)

    target_lines = []
    first = lines[cut_i]
    base_indent = len(first) - len(first.lstrip(" "))

    for j in range(cut_i, min(len(lines), cut_i + max_target_lines)):
        s = lines[j]
        if s.strip() == "":
            if target_lines:
                break
            else:
                continue

        indent = len(s) - len(s.lstrip(" "))
        if target_lines and indent < base_indent:
            break

        target_lines.append(s)

        if re.match(r"^\s*(return|raise)\b", s):
            break

    target_text = "\n".join(target_lines).rstrip() + ("\n" if target_lines else "")
    return target_text, len(target_lines)


def cap_prefix(lines: List[str], cut_i: int, max_prefix_lines: int) -> Tuple[str, int]:
    prefix_lines = lines[:cut_i]
    if len(prefix_lines) > max_prefix_lines:
        prefix_lines = prefix_lines[-max_prefix_lines:]
        trimmed = 1
    else:
        trimmed = 0
    prefix_text = "\n".join(prefix_lines).rstrip() + "\n"
    return prefix_text, trimmed


class TestNextBlockDatasetBuilder:
    def __init__(self, test_files_root, output_root, max_pairs_per_repo, max_pairs_per_file, 
        max_pairs_per_test, max_prefix_lines, max_target_lines, tiered_caps, seed):
        self.test_files_root = test_files_root
        self.output_root = output_root

        self.max_pairs_per_repo = max_pairs_per_repo
        self.max_pairs_per_file = max_pairs_per_file
        self.max_pairs_per_test = max_pairs_per_test
        self.max_prefix_lines = max_prefix_lines
        self.max_target_lines = max_target_lines
        self.tiered_caps = tiered_caps

        random.seed(seed)

    def get_repo_cap(self, repo_test_fn_count):
        if not self.tiered_caps:
            if self.max_pairs_per_repo is None:
                return 10**9
            return self.max_pairs_per_repo

        if repo_test_fn_count <= 100:
            return 10**9  
        if repo_test_fn_count <= 500:
            return 200
        if repo_test_fn_count <= 1000:
            return 200
        if repo_test_fn_count <= 5000:
            return 150
        return 50  # 5000+ extreme outliers

    def process_repo(self, repo, repo_test_fn_count_hint=None):
        repo_dir = os.path.join(self.test_files_root, repo)
        if not os.path.isdir(repo_dir):
            return []

        cap = self.get_repo_cap(repo_test_fn_count_hint or 0)

        pairs = []
        per_file_counts = {}
        used_cut_lines = {} 

        for root, _, files in os.walk(repo_dir):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                rel_file = os.path.relpath(os.path.join(root, fn), repo_dir).replace("\\", "/")

                if per_file_counts.get(rel_file, 0) >= self.max_pairs_per_file:
                    continue
                
                if rel_file not in used_cut_lines:
                    used_cut_lines[rel_file] = set()

                file_path = os.path.join(root, fn)
                try:
                    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                lines = safe_split_lines(text)
                if len(lines) < DEFAULT_MIN_LINES_IN_FILE:
                    continue

                framework = detect_framework(text)
                tags = collect_test_type_tags(text)

                try:
                    tree = ast.parse(text)
                except Exception:
                    continue

                test_nodes = find_test_nodes(tree)
                if not test_nodes:
                    continue

                random.shuffle(test_nodes)

                file_pairs_added = 0

                for node in test_nodes:
                    if file_pairs_added >= self.max_pairs_per_file:
                        break
                    if len(pairs) >= cap:
                        break

                    test_start, test_end = get_node_span_lines(node)
                    if test_end - test_start < DEFAULT_MIN_LINES_IN_TEST:
                        continue

                    candidates = choose_cut_points_in_test(
                        lines=lines,
                        test_start=test_start,
                        test_end=test_end,
                        max_target_lines=self.max_target_lines,
                    )
                    if not candidates:
                        continue

                    random.shuffle(candidates)
                    candidates = candidates[: self.max_pairs_per_test]

                    for cut_i, cut_kind in candidates:
                        if file_pairs_added >= self.max_pairs_per_file:
                            break
                        if len(pairs) >= cap:
                            break
                        
                        if cut_i in used_cut_lines[rel_file]:
                            continue

                        prefix_text, trimmed = cap_prefix(lines, cut_i, self.max_prefix_lines)
                        target_text, n_tgt = extract_next_block(lines, cut_i, self.max_target_lines)

                        if n_tgt == 0:
                            continue

                        if target_text.strip() == "pass":
                            continue

                        test_name = getattr(node, "name", "<unknown>")
                        cls_name = self._find_enclosing_class_name(tree, node)

                        pair = {
                            "repo": repo,
                            "task": "test_next_block",
                            "framework": framework,
                            "test_type": tags,
                            "prefix": prefix_text,
                            "target": target_text,
                            "metadata": {
                                "file": rel_file,
                                "class": cls_name,
                                "function": test_name,
                                "cut_kind": cut_kind,
                                "cut_line": cut_i + 1,  # 1-indexed for humans
                                "prefix_trimmed": bool(trimmed),
                            },
                        }
                        pairs.append(pair)
                        file_pairs_added += 1
                        per_file_counts[rel_file] = per_file_counts.get(rel_file, 0) + 1
                        used_cut_lines[rel_file].add(cut_i)  # Mark this cut line as used

                if len(pairs) >= cap:
                    break

        if (not self.tiered_caps) and self.max_pairs_per_repo is not None and len(pairs) > self.max_pairs_per_repo:
            random.shuffle(pairs)
            pairs = pairs[: self.max_pairs_per_repo]

        return pairs

    def _find_enclosing_class_name(self, tree, target_node):
        try:
            t_start, t_end = get_node_span_lines(target_node)
        except Exception:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                c_start = getattr(node, "lineno", None)
                c_end = getattr(node, "end_lineno", None)
                if c_start is None or c_end is None:
                    continue
                c_start -= 1
                if c_start <= t_start and t_end <= c_end:
                    return node.name
        return None


def load_repo_list(repos_file):
    repos = []
    with open(repos_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace("https://github.com/", "").strip()
            repos.append(s)
    return repos


def main():
    p = argparse.ArgumentParser(description="Generate test_next_block dataset from test files only")
    p.add_argument("--repos-file", type=str, default="filtered_repo_urls.txt")
    p.add_argument("--test-files-root", type=str, default="/home/lhotsko/scratch/test_files")
    p.add_argument("--output-root", type=str, default="/home/lhotsko/scratch/qna_pairs/test_next_block")

    p.add_argument("--max-pairs-per-repo", type=int, default=30,
                   help="If --tiered-caps is off, cap pairs per repo to this value.")
    p.add_argument("--tiered-caps", action="store_true",
                   help="Enable tiered caps based on repo test function counts (requires --repo-counts-json or hints).")

    p.add_argument("--max-pairs-per-file", type=int, default=DEFAULT_MAX_PAIRS_PER_FILE)
    p.add_argument("--max-pairs-per-test", type=int, default=DEFAULT_MAX_PAIRS_PER_TEST)

    p.add_argument("--max-prefix-lines", type=int, default=DEFAULT_MAX_PREFIX_LINES)
    p.add_argument("--max-target-lines", type=int, default=DEFAULT_MAX_TARGET_LINES)

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Optional: provide a json mapping repo -> test_fn_count for tiered caps
    p.add_argument("--repo-counts-json", type=str, default=None,
                   help="Optional JSON file mapping repo -> test_function_count (for tiered caps).")

    args = p.parse_args()

    repos = load_repo_list(args.repos_file)

    repo_counts = {}
    if args.repo_counts_json:
        try:
            repo_counts = json.loads(Path(args.repo_counts_json).read_text(encoding="utf-8"))
        except Exception:
            repo_counts = {}

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = TestNextBlockDatasetBuilder(
        test_files_root=args.test_files_root,
        output_root=args.output_root,
        max_pairs_per_repo=args.max_pairs_per_repo,
        max_pairs_per_file=args.max_pairs_per_file,
        max_pairs_per_test=args.max_pairs_per_test,
        max_prefix_lines=args.max_prefix_lines,
        max_target_lines=args.max_target_lines,
        tiered_caps=args.tiered_caps,
        seed=args.seed,
    )

    out_path = out_dir / "test_next_block.jsonl"
    stats_path = out_dir / "statistics.json"

    total_pairs = 0
    repos_with_pairs = 0
    per_repo_pairs = {}

    with out_path.open("w", encoding="utf-8") as w:
        for repo in tqdm(repos, desc="Processing repos"):
            hint = repo_counts.get(repo)
            pairs = builder.process_repo(repo, repo_test_fn_count_hint=hint)

            if pairs:
                repos_with_pairs += 1
                per_repo_pairs[repo] = len(pairs)

            for row in pairs:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)

    stats = {
        "total_pairs": total_pairs,
        "repos_processed": len(repos),
        "repos_with_pairs": repos_with_pairs,
        "avg_pairs_per_repo_with_pairs": (total_pairs / repos_with_pairs) if repos_with_pairs else 0,
        "max_pairs_in_repo": max(per_repo_pairs.values()) if per_repo_pairs else 0,
        "min_pairs_in_repo": min(per_repo_pairs.values()) if per_repo_pairs else 0,
        "max_pairs_per_repo_arg": args.max_pairs_per_repo,
        "tiered_caps": bool(args.tiered_caps),
        "max_pairs_per_file": args.max_pairs_per_file,
        "max_pairs_per_test": args.max_pairs_per_test,
        "max_prefix_lines": args.max_prefix_lines,
        "max_target_lines": args.max_target_lines,
        "output_jsonl": str(out_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nWrote JSONL: {out_path}")
    print(f"Wrote stats: {stats_path}")
    print(f"Total pairs: {total_pairs}")


if __name__ == "__main__":
    main()
