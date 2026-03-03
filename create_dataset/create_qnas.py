#!/usr/bin/env python3
"""
Parsers for extracting Q&A pairs from assertions in test files.
Covers: plain assert, self.assert*, pytest.raises, pytest.approx, assert_* (numpy).

Run over all repos: extracts pairs from TEST_HYPERNET, saves best to QNA_HYPERNET.json.
"""

import argparse
import ast
import json
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPOSITORIES_DIR = "/home/lhotsko/scratch/REPO_DATASET/repositories"
TEST_HYPERNET = "TEST_HYPERNET"
QNA_HYPERNET = "QNA_HYPERNET.json"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn", "site-packages",
}

# Unittest assert methods (for self.assert*)
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


@dataclass
class AssertionPair:
    """Extracted (prefix, target) for one assertion."""
    assertion_type: str
    prefix: str
    target: str
    lineno: int
    col_offset: int


def get_source_segment(source: str, node: ast.AST) -> str | None:
    """Get source code for node. Handles Python < 3.8."""
    if hasattr(ast, "get_source_segment"):
        return ast.get_source_segment(source, node)
    # Fallback: use line range
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1) - 1
    if start < 0 or end >= len(lines):
        return None
    return "\n".join(lines[start : end + 1])


def get_line_at(source: str, lineno: int) -> str:
    """Get line at 1-based lineno."""
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def extract_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def flatten_to_oneliner(s: str) -> str:
    """Convert multi-line string to single line: newlines -> space, collapse spaces."""
    if not s or "\n" not in s:
        return s
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# 1. Plain assert
# ---------------------------------------------------------------------------

def _parse_plain_assert(source: str, node: ast.Assert, lines: list[str]) -> AssertionPair | None:
    """Extract prefix/target from plain assert. Target = RHS of comparison or full expr."""
    src = get_source_segment(source, node)
    if not src:
        return None
    lineno = node.lineno
    col = node.col_offset

    # assert <test>
    test = node.test
    if isinstance(test, ast.Compare):
        # assert a == b, assert a in b, etc.
        left_src = get_source_segment(source, test.left)
        if not left_src:
            return None
        # Find first comparator
        op = test.ops[0]
        comparators = test.comparators
        if not comparators:
            return None
        rhs = comparators[0]
        rhs_src = get_source_segment(source, rhs)
        if not rhs_src:
            return None
        op_map = {
            ast.Eq: " == ", ast.NotEq: " != ", ast.Lt: " < ", ast.LtE: " <= ",
            ast.Gt: " > ", ast.GtE: " >= ", ast.In: " in ", ast.NotIn: " not in ",
            ast.Is: " is ", ast.IsNot: " is not ",
        }
        op_str = op_map.get(type(op), " ??? ")
        prefix = f"assert {left_src}{op_str}"
        target = rhs_src
    else:
        # assert x, assert foo()
        prefix = "assert "
        target = get_source_segment(source, test) or ""
    return AssertionPair("assert", prefix, target, lineno, col)


# ---------------------------------------------------------------------------
# 2. self.assert*
# ---------------------------------------------------------------------------

def _parse_self_assert(source: str, node: ast.Call, lines: list[str]) -> AssertionPair | None:
    """Extract from self.assertEqual(a, b) etc. Target = last arg or first for assertRaises."""
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name):
        return None
    if node.func.value.id != "self":
        return None
    method = node.func.attr
    if method not in SELF_ASSERT_METHODS:
        return None

    args = node.args
    if not args:
        return None

    full_src = get_source_segment(source, node)
    if not full_src:
        return None

    # assertRaises(Exc, fn, *args) - target = "Exc" or "Exc, fn"
    if method == "assertRaises":
        exc_src = get_source_segment(source, args[0])
        if exc_src:
            return AssertionPair(f"self.{method}", f"self.assertRaises(", exc_src + ")", node.lineno, node.col_offset)
        return None

    # assertRaisesRegex(Exc, pat, fn)
    if method == "assertRaisesRegex":
        if len(args) >= 2:
            exc_src = get_source_segment(source, args[0])
            pat_src = get_source_segment(source, args[1])
            if exc_src and pat_src:
                return AssertionPair(f"self.{method}", f"self.assertRaisesRegex(", f"{exc_src}, {pat_src})", node.lineno, node.col_offset)
        return None

    # assertTrue(x), assertFalse(x), assertIsNone(x) - single arg
    if method in ("assertTrue", "assertFalse", "assertIsNone", "assertIsNotNone"):
        arg_src = get_source_segment(source, args[0])
        if arg_src:
            return AssertionPair(f"self.{method}", f"self.{method}(", arg_src + ")", node.lineno, node.col_offset)
        return None

    # assertEqual(a, b), assertIn(a, b), etc. - target = last arg
    if len(args) >= 2:
        last_arg = args[-1]
        last_src = get_source_segment(source, last_arg)
        if last_src:
            # Prefix = method name + all args except last
            prefix_parts = [f"self.{method}("]
            for i, arg in enumerate(args[:-1]):
                arg_src = get_source_segment(source, arg)
                if arg_src:
                    prefix_parts.append(arg_src)
                    if i < len(args) - 2:
                        prefix_parts.append(", ")
            prefix = "".join(prefix_parts) + ", "
            return AssertionPair(f"self.{method}", prefix, last_src + ")", node.lineno, node.col_offset)
    return None


# ---------------------------------------------------------------------------
# 3. pytest.raises
# ---------------------------------------------------------------------------

def _parse_pytest_raises(source: str, node: ast.Call, lines: list[str]) -> AssertionPair | None:
    """Extract from pytest.raises(ValueError) or pytest.raises(ValueError, match='...')."""
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "raises":
        return None

    args = node.args
    if not args:
        return None
    exc_src = get_source_segment(source, args[0])
    if not exc_src:
        return None

    # Include keyword args in target if present: match="..."
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f', {kw.arg}={kw_src}')
    target = exc_src + "".join(kw_parts) + ")"
    return AssertionPair("pytest.raises", "pytest.raises(", target, node.lineno, node.col_offset)


# ---------------------------------------------------------------------------
# 4. pytest.approx
# ---------------------------------------------------------------------------

def _parse_pytest_approx(source: str, node: ast.Call, lines: list[str]) -> AssertionPair | None:
    """Extract from assert x == pytest.approx(3.14). Target = arg to approx."""
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "approx":
        return None

    args = node.args
    if not args:
        return None
    arg_src = get_source_segment(source, args[0])
    if not arg_src:
        return None

    # Include rel, abs kwargs if present
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f', {kw.arg}={kw_src}')
    target = arg_src + "".join(kw_parts) + ")"
    return AssertionPair("pytest.approx", "pytest.approx(", target, node.lineno, node.col_offset)


# ---------------------------------------------------------------------------
# 5. assert_* (numpy/testing)
# ---------------------------------------------------------------------------

def _parse_assert_underscore(source: str, node: ast.Call, lines: list[str]) -> AssertionPair | None:
    """Extract from assert_equal(a, b), np.testing.assert_array_equal(x, y)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        # np.testing.assert_equal, unittest.mock.assert_called_once
        name = func.attr
        if not name.startswith("assert_") or name == "assert_":
            return None
        # Skip mock methods
        if isinstance(func.value, ast.Attribute):
            if func.value.attr == "mock" or (isinstance(func.value.value, ast.Name) and func.value.value.id == "unittest"):
                return None
    elif isinstance(func, ast.Name):
        name = func.id
        if not name.startswith("assert_") or name == "assert_":
            return None
    else:
        return None

    args = node.args
    if not args:
        return None

    # Most assert_* take (actual, expected) - target = last arg
    last_arg = args[-1]
    last_src = get_source_segment(source, last_arg)
    if not last_src:
        return None

    prefix_parts = [f"{get_source_segment(source, func) or name}("]
    for i, arg in enumerate(args[:-1]):
        arg_src = get_source_segment(source, arg)
        if arg_src:
            prefix_parts.append(arg_src)
            if i < len(args) - 2:
                prefix_parts.append(", ")
    prefix = "".join(prefix_parts) + ", "
    return AssertionPair("assert_*", prefix, last_src + ")", node.lineno, node.col_offset)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def _find_enclosing_test(tree: ast.AST, node: ast.AST) -> ast.AST | None:
    """Find the test function/class that contains this node."""
    node_start = getattr(node, "lineno", 0)
    node_end = getattr(node, "end_lineno", node_start)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "test" in n.name.lower():
                s, e = getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)
                if s <= node_start and node_end <= e:
                    if best is None or (getattr(best, "lineno", 0) < s):
                        best = n
        if isinstance(n, ast.ClassDef) and "test" in n.name.lower():
            s, e = getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)
            if s <= node_start and node_end <= e:
                for item in n.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                        is_, ie = getattr(item, "lineno", 0), getattr(item, "end_lineno", 0)
                        if is_ <= node_start and node_end <= ie:
                            if best is None or getattr(best, "lineno", 0) < is_:
                                best = item
    return best


def extract_assertion_pairs(source: str, file_path: str = "") -> Iterator[tuple[AssertionPair, str, str]]:
    """
    Yield (AssertionPair, context_prefix, test_name) for each assertion in source.
    context_prefix = code before the assertion (for building full Q).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return

    lines = source.splitlines()
    max_prefix_lines = 300

    def get_context_before(lineno: int, col_offset: int) -> str:
        """Get code context before (lineno, col_offset), capped."""
        start = max(0, lineno - 1 - max_prefix_lines)
        if start >= lineno - 1:
            return ""
        ctx_lines = lines[start : lineno - 1]
        last_line = lines[lineno - 1] if lineno <= len(lines) else ""
        last_line_prefix = last_line[:col_offset] if col_offset <= len(last_line) else last_line
        ctx_lines.append(last_line_prefix.rstrip())
        return "\n".join(ctx_lines) + "\n"

    for node in ast.walk(tree):
        pair = None

        if isinstance(node, ast.Assert):
            pair = _parse_plain_assert(source, node, lines)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                val = node.func.value
                val_id = getattr(val, "id", None) if isinstance(val, ast.Name) else None
                if val_id == "self" and node.func.attr in SELF_ASSERT_METHODS:
                    pair = _parse_self_assert(source, node, lines)
                elif val_id == "pytest":
                    if node.func.attr == "raises":
                        pair = _parse_pytest_raises(source, node, lines)
                    elif node.func.attr == "approx":
                        pair = _parse_pytest_approx(source, node, lines)
                    else:
                        pair = _parse_assert_underscore(source, node, lines)
                else:
                    pair = _parse_assert_underscore(source, node, lines)
            elif isinstance(node.func, ast.Name) and node.func.id.startswith("assert_"):
                pair = _parse_assert_underscore(source, node, lines)
            elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert_"):
                pair = _parse_assert_underscore(source, node, lines)

        if pair and pair.target.strip():
            target = pair.target
            if target.lstrip().startswith(","):
                continue
            was_multiline = "\n" in target
            if was_multiline:
                target = flatten_to_oneliner(target)
            if not target.strip():
                continue
            pair_flat = AssertionPair(pair.assertion_type, pair.prefix, target, pair.lineno, pair.col_offset)
            test_node = _find_enclosing_test(tree, node)
            test_name = test_node.name if test_node else ""
            context = get_context_before(pair.lineno, pair.col_offset)
            yield pair_flat, context, test_name, was_multiline


def extract_all_from_file(file_path: str, source: str) -> list[dict]:
    """
    Extract all assertion pairs from a file. Returns list of dicts with
    prefix, target, assertion_type, context, metadata.
    Multi-line targets are flattened to one line.
    """
    results = []
    for pair, context, test_name, was_multiline in extract_assertion_pairs(source, file_path):
        full_prefix = context + pair.prefix
        results.append({
            "prefix": full_prefix,
            "target": pair.target,
            "assertion_type": pair.assertion_type,
            "context_lines": context.count("\n") + 1,
            "metadata": {
                "file": file_path,
                "lineno": pair.lineno,
                "col_offset": pair.col_offset,
                "test_function": test_name,
                "was_multiline": was_multiline,
            },
        })
    return results


# ---------------------------------------------------------------------------
# Run over all repos
# ---------------------------------------------------------------------------

def iter_repos_with_test_hypernet(repos_root: str) -> Iterator[tuple[str, str, str]]:
    """Yield (author, repo_name, repo_path) for repos that have TEST_HYPERNET."""
    for author in os.listdir(repos_root):
        author_path = os.path.join(repos_root, author)
        if not os.path.isdir(author_path):
            continue
        for repo_name in os.listdir(author_path):
            repo_path = os.path.join(author_path, repo_name)
            test_dir = os.path.join(repo_path, TEST_HYPERNET)
            if not os.path.isdir(test_dir):
                continue
            if repo_name == TEST_HYPERNET:
                continue
            yield author, repo_name, repo_path


def iter_py_files_in_test_hypernet(repo_path: str) -> Iterator[tuple[str, str]]:
    """Yield (rel_path, abs_path) for each .py file in TEST_HYPERNET."""
    test_dir = os.path.join(repo_path, TEST_HYPERNET)
    for root, dirs, files in os.walk(test_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = os.path.relpath(root, test_dir)
        for f in files:
            if f.endswith(".py"):
                rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                abs_path = os.path.join(root, f)
                yield rel_path, abs_path


def select_best_pairs(
    pairs: list[dict],
    max_per_repo: int = 100,
    max_per_function: int = 3,
    prefer_single_line: bool = True,
) -> list[dict]:
    """
    Select balanced pairs for training:
    - Limit per test function to avoid concentration (no file cap - single-file repos are fine)
    - Stratify by assertion_type for coverage
    - Prefer single-line, shorter targets
    """
    if not pairs:
        return []

    def quality_key(p: dict) -> tuple:
        return (
            p["metadata"].get("was_multiline", False),
            len(p["target"]),
        )

    # Group by (file, function), then by assertion_type
    by_file_func = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        f = p["metadata"].get("file", "")
        fn = p["metadata"].get("test_function", "")
        by_file_func[(f, fn)][p["assertion_type"]].append(p)

    # Within each (file, function): take best from each type (round-robin), cap at max_per_function
    capped = []
    for (f, fn), by_type_local in by_file_func.items():
        for t in by_type_local:
            by_type_local[t].sort(key=quality_key)
        types_ordered = sorted(by_type_local.keys())
        indices = {t: 0 for t in types_ordered}
        taken = []
        start_idx = 0
        while len(taken) < max_per_function:
            added = False
            for i in range(len(types_ordered)):
                t = types_ordered[(start_idx + i) % len(types_ordered)]
                if indices[t] < len(by_type_local[t]):
                    taken.append(by_type_local[t][indices[t]])
                    indices[t] += 1
                    start_idx = (start_idx + i + 1) % len(types_ordered)
                    added = True
                    break
            if not added:
                break
        capped.extend(taken)

    # Group by assertion_type
    by_type = defaultdict(list)
    for p in capped:
        by_type[p["assertion_type"]].append(p)
    for t in by_type:
        by_type[t].sort(key=quality_key)
        # Deduplicate by target within each type (keep best quality)
        seen_targets = set()
        deduped = []
        for p in by_type[t]:
            target = p["target"]
            if target not in seen_targets:
                seen_targets.add(target)
                deduped.append(p)
        by_type[t] = deduped

    # Proportional sampling: preserve repo's assertion-type distribution
    total = sum(len(by_type[t]) for t in by_type)
    if total == 0:
        return []

    # Allocate slots proportionally (largest remainder method)
    frac_parts = []
    slots = {}
    for t in by_type:
        count = len(by_type[t])
        exact = max_per_repo * count / total
        floor_slots = int(exact)
        frac = exact - floor_slots
        slots[t] = floor_slots
        frac_parts.append((t, frac))
    remainder = max_per_repo - sum(slots.values())
    # Give extra slots to types with largest fractional part
    frac_parts.sort(key=lambda x: -x[1])
    for i in range(min(remainder, len(frac_parts))):
        slots[frac_parts[i][0]] += 1

    selected = []
    for t in by_type:
        n = min(slots[t], len(by_type[t]))
        selected.extend(by_type[t][:n])
    return selected


def process_repo(
    repo_path: str,
    author: str,
    repo_name: str,
    max_per_repo: int = 200,
    max_per_function: int = 3,
    prefer_single_line: bool = True,
) -> tuple[int, int]:
    """
    Extract all pairs from repo, select best, save to QNA_HYPERNET.json.
    Returns (total_extracted, saved_count).
    """
    all_pairs = []
    for rel_path, abs_path in iter_py_files_in_test_hypernet(repo_path):
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        pairs = extract_all_from_file(rel_path, source)
        for p in pairs:
            p["metadata"]["repo"] = f"{author}/{repo_name}"
            p["metadata"]["source_file"] = str(Path(abs_path).resolve())
        all_pairs.extend(pairs)

    if not all_pairs:
        return 0, 0

    best = select_best_pairs(
        all_pairs,
        max_per_repo=max_per_repo,
        max_per_function=max_per_function,
        prefer_single_line=prefer_single_line,
    )

    out_path = os.path.join(repo_path, QNA_HYPERNET)
    output = {
        "repo": f"{author}/{repo_name}",
        "total_extracted": len(all_pairs),
        "saved_count": len(best),
        "selection_params": {
            "max_per_repo": max_per_repo,
            "max_per_function": max_per_function,
        },
        "pairs": best,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(all_pairs), len(best)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from test assertions, save best to QNA_HYPERNET.json per repo"
    )
    parser.add_argument(
        "--repos-dir",
        default=REPOSITORIES_DIR,
        help=f"Root of repositories (default: {REPOSITORIES_DIR})",
    )
    parser.add_argument(
        "--max-per-repo",
        type=int,
        default=200,
        help="Max pairs to save per repo (default: 200)",
    )
    parser.add_argument(
        "--max-per-function",
        type=int,
        default=3,
        help="Max pairs per test function (default: 3)",
    )
    parser.add_argument(
        "--no-prefer-single-line",
        action="store_true",
        help="Don't prefer single-line targets over multi-line",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N repos (for testing)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Process single repo: owner/repo_name",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repos_dir):
        print(f"Repositories dir not found: {args.repos_dir}")
        return

    if args.repo:
        parts = args.repo.split("/", 1)
        if len(parts) != 2:
            print("--repo must be owner/repo_name")
            return
        author, repo_name = parts
        repo_path = os.path.join(args.repos_dir, author, repo_name)
        if not os.path.isdir(repo_path):
            print(f"Repo not found: {repo_path}")
            return
        repos_iter = [(author, repo_name, repo_path)]
    else:
        repos_iter = list(iter_repos_with_test_hypernet(args.repos_dir))
        if args.limit:
            repos_iter = repos_iter[: args.limit]

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    total_extracted = 0
    total_saved = 0
    processed = 0

    for author, repo_name, repo_path in tqdm(repos_iter, desc="Repos"):
        n_ext, n_saved = process_repo(
            repo_path,
            author,
            repo_name,
            max_per_repo=args.max_per_repo,
            max_per_function=args.max_per_function,
            prefer_single_line=not args.no_prefer_single_line,
        )
        total_extracted += n_ext
        total_saved += n_saved
        if n_saved > 0:
            processed += 1

    print(f"\nProcessed: {processed} repos")
    print(f"Total extracted: {total_extracted}")
    print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        # Legacy: python create_qnas.py <file.py>
        path = Path(sys.argv[1])
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="ignore")
            pairs = extract_all_from_file(str(path), source)
            for p in pairs:
                print(f"[{p['assertion_type']}] L{p['metadata']['lineno']}: target={p['target'][:50]!r}")
            print(f"\nTotal: {len(pairs)} pairs")
            sys.exit(0)
    main()
