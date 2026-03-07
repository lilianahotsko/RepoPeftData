#!/usr/bin/env python3
"""
V2 QnA extraction: clean, structured, balanced.

Replaces create_qnas.py + rebuild_prefixes.py in a single pass.
Reads the full test file, extracts assertions with proper indentation,
builds structured prefixes (imports + class + helpers + test body),
applies quality filters, and selects a balanced set.

Usage:
    python create_dataset/create_qnas_v2.py
    python create_dataset/create_qnas_v2.py --repos-dir $SCRATCH/REPO_DATASET/repositories
    python create_dataset/create_qnas_v2.py --validate --splits-dir $SCRATCH/REPO_DATASET
"""

import argparse
import ast
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

REPOSITORIES_DIR = "/home/lhotsko/scratch/REPO_DATASET/repositories"
TEST_HYPERNET = "TEST_HYPERNET"
QNA_HYPERNET = "QNA_HYPERNET.json"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn", "site-packages",
}

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

SETUP_METHODS = {
    "setUp", "setUpClass", "tearDown", "tearDownClass",
    "setup_method", "teardown_method", "setup", "teardown",
}


@dataclass
class RawAssertion:
    assertion_type: str
    target_src: str
    target_node: ast.AST
    assertion_node: ast.AST
    lineno: int
    col_offset: int
    target_col: int  # column where target starts in source
    enclosing_func: Optional[str] = None
    enclosing_class: Optional[str] = None


# ======================================================================
# AST helpers
# ======================================================================

def _get_source_segment(source: str, node: ast.AST) -> Optional[str]:
    if hasattr(ast, "get_source_segment"):
        return ast.get_source_segment(source, node)
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    if 0 <= start < len(lines) and end <= len(lines):
        return "\n".join(lines[start:end])
    return None


def _find_enclosing(tree: ast.AST, target_lineno: int):
    """Find the enclosing function and class for a given line."""
    best_func = None
    best_class = None
    for node in ast.walk(tree):
        s = getattr(node, "lineno", 0)
        e = getattr(node, "end_lineno", 0)
        if not (s <= target_lineno <= e):
            continue
        if isinstance(node, ast.ClassDef):
            if best_class is None or s > getattr(best_class, "lineno", 0):
                best_class = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if best_func is None or s > getattr(best_func, "lineno", 0):
                best_func = node
    return best_func, best_class


# ======================================================================
# Assertion extractors (return RawAssertion or None)
# ======================================================================

def _extract_plain_assert(source: str, node: ast.Assert) -> Optional[RawAssertion]:
    test = node.test
    if not isinstance(test, ast.Compare):
        return None
    if not test.comparators:
        return None
    rhs = test.comparators[0]
    rhs_src = _get_source_segment(source, rhs)
    if not rhs_src:
        return None
    return RawAssertion(
        assertion_type="assert",
        target_src=rhs_src,
        target_node=rhs,
        assertion_node=node,
        lineno=node.lineno,
        col_offset=node.col_offset,
        target_col=rhs.col_offset,
    )


def _extract_self_assert(source: str, node: ast.Call) -> Optional[RawAssertion]:
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
        return None
    method = node.func.attr
    if method not in SELF_ASSERT_METHODS:
        return None
    args = node.args
    if not args:
        return None

    if method in ("assertRaises", "assertRaisesRegex"):
        exc_src = _get_source_segment(source, args[0])
        if not exc_src:
            return None
        target_col = args[0].col_offset
        kw = ""
        if method == "assertRaisesRegex" and len(args) >= 2:
            pat = _get_source_segment(source, args[1])
            if pat:
                kw = f", {pat}"
        return RawAssertion(
            assertion_type=f"self.{method}",
            target_src=exc_src + kw + ")",
            target_node=args[0],
            assertion_node=node,
            lineno=node.lineno,
            col_offset=node.col_offset,
            target_col=target_col,
        )

    if method in ("assertTrue", "assertFalse", "assertIsNone", "assertIsNotNone"):
        arg_src = _get_source_segment(source, args[0])
        if not arg_src:
            return None
        return RawAssertion(
            assertion_type=f"self.{method}",
            target_src=arg_src + ")",
            target_node=args[0],
            assertion_node=node,
            lineno=node.lineno,
            col_offset=node.col_offset,
            target_col=args[0].col_offset,
        )

    if len(args) >= 2:
        last = args[-1]
        last_src = _get_source_segment(source, last)
        if not last_src:
            return None
        return RawAssertion(
            assertion_type=f"self.{method}",
            target_src=last_src + ")",
            target_node=last,
            assertion_node=node,
            lineno=node.lineno,
            col_offset=node.col_offset,
            target_col=last.col_offset,
        )
    return None


def _extract_pytest_raises(source: str, node: ast.Call) -> Optional[RawAssertion]:
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "raises":
        return None
    args = node.args
    if not args:
        return None
    exc_src = _get_source_segment(source, args[0])
    if not exc_src:
        return None
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = _get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f", {kw.arg}={kw_src}")
    return RawAssertion(
        assertion_type="pytest.raises",
        target_src=exc_src + "".join(kw_parts) + ")",
        target_node=args[0],
        assertion_node=node,
        lineno=node.lineno,
        col_offset=node.col_offset,
        target_col=args[0].col_offset,
    )


def _extract_pytest_approx(source: str, node: ast.Call) -> Optional[RawAssertion]:
    if not isinstance(node.func, ast.Attribute):
        return None
    val = node.func.value
    if not isinstance(val, ast.Name) or val.id != "pytest" or node.func.attr != "approx":
        return None
    args = node.args
    if not args:
        return None
    arg_src = _get_source_segment(source, args[0])
    if not arg_src:
        return None
    kw_parts = []
    for kw in node.keywords:
        if kw.arg and kw.value:
            kw_src = _get_source_segment(source, kw.value)
            if kw_src:
                kw_parts.append(f", {kw.arg}={kw_src}")
    return RawAssertion(
        assertion_type="pytest.approx",
        target_src=arg_src + "".join(kw_parts) + ")",
        target_node=args[0],
        assertion_node=node,
        lineno=node.lineno,
        col_offset=node.col_offset,
        target_col=args[0].col_offset,
    )


def _extract_assert_underscore(source: str, node: ast.Call) -> Optional[RawAssertion]:
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        if not name.startswith("assert_") or name == "assert_":
            return None
        if isinstance(func.value, ast.Attribute):
            if func.value.attr == "mock":
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

    last = args[-1]
    last_src = _get_source_segment(source, last)
    if not last_src:
        return None

    # FIX: for 1-arg functions, target is the single arg (no leading comma)
    return RawAssertion(
        assertion_type="assert_*",
        target_src=last_src + ")",
        target_node=last,
        assertion_node=node,
        lineno=node.lineno,
        col_offset=node.col_offset,
        target_col=last.col_offset,
    )


# ======================================================================
# Structured prefix builder
# ======================================================================

def _get_node_source_lines(lines: list[str], node: ast.AST) -> list[str]:
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return lines[start:end]


def build_structured_prefix(
    source: str,
    lines: list[str],
    tree: ast.AST,
    assertion: RawAssertion,
    func_node: Optional[ast.AST],
    class_node: Optional[ast.AST],
    max_prefix_lines: int = 600,
) -> str:
    """
    Build a structured prefix from the full source.
    Returns the code from the file up to (and including) the assertion line,
    with irrelevant test functions removed.
    """
    target_lineno = assertion.lineno  # 1-based
    target_col = assertion.target_col

    # --- Section 1: Imports + module-level statements ---
    import_section = []
    module_end = 0
    for node in ast.iter_child_nodes(tree):
        end = getattr(node, "end_lineno", getattr(node, "lineno", 0))
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                             ast.AugAssign, ast.Expr)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    continue  # skip module docstrings
            for i in range(getattr(node, "lineno", 1) - 1, min(end, len(lines))):
                import_section.append(i)
            module_end = max(module_end, end)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break  # stop at first function/class definition

    # --- Section 2: Enclosing class header + setUp ---
    class_lines = []
    if class_node:
        cls_start = getattr(class_node, "lineno", 1) - 1
        class_lines.append(cls_start)  # class Foo(Base):
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in SETUP_METHODS:
                    for i in range(getattr(item, "lineno", 1) - 1,
                                   getattr(item, "end_lineno", getattr(item, "lineno", 1))):
                        class_lines.append(i)
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                for i in range(getattr(item, "lineno", 1) - 1,
                               getattr(item, "end_lineno", getattr(item, "lineno", 1))):
                    class_lines.append(i)

    # --- Section 3: Helpers and fixtures (non-test functions before the assertion) ---
    helper_lines = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node_end = getattr(node, "end_lineno", 0)
        if node_end >= target_lineno:
            continue
        if node.name.startswith("test_"):
            continue
        is_fixture = any(
            (isinstance(d, ast.Attribute) and d.attr == "fixture") or
            (isinstance(d, ast.Name) and d.id == "fixture") or
            (isinstance(d, ast.Call) and hasattr(d, "func") and
             isinstance(d.func, ast.Attribute) and d.func.attr == "fixture")
            for d in node.decorator_list
        )
        if is_fixture or not node.name.startswith("test_"):
            for i in range(getattr(node, "lineno", 1) - 1, node_end):
                helper_lines.append(i)

    # Also check helpers inside the enclosing class
    if class_node:
        for item in class_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            item_end = getattr(item, "end_lineno", 0)
            if item_end >= target_lineno:
                continue
            if item.name.startswith("test_") and item.name != (func_node.name if func_node else ""):
                continue
            if item.name not in SETUP_METHODS:
                for i in range(getattr(item, "lineno", 1) - 1, item_end):
                    helper_lines.append(i)

    # --- Section 4: Test function body up to assertion ---
    test_body_lines = []
    if func_node:
        # Include decorators
        func_start = getattr(func_node, "lineno", 1) - 1
        for dec in func_node.decorator_list:
            dec_line = getattr(dec, "lineno", func_start + 1) - 1
            func_start = min(func_start, dec_line)
        for i in range(func_start, target_lineno - 1):
            test_body_lines.append(i)
        # The assertion line itself, up to where the target starts
        assertion_line = lines[target_lineno - 1] if target_lineno <= len(lines) else ""
        assertion_prefix_text = assertion_line[:target_col]
    else:
        # No enclosing function -- take last 50 lines before assertion
        start = max(0, target_lineno - 51)
        for i in range(start, target_lineno - 1):
            test_body_lines.append(i)
        assertion_line = lines[target_lineno - 1] if target_lineno <= len(lines) else ""
        assertion_prefix_text = assertion_line[:target_col]

    # --- Assemble: collect unique line indices in order ---
    all_line_indices = set()
    all_line_indices.update(import_section)
    all_line_indices.update(class_lines)
    all_line_indices.update(helper_lines)
    all_line_indices.update(test_body_lines)

    sorted_indices = sorted(all_line_indices)

    # Build prefix from selected lines
    result_lines = []
    prev_idx = -2
    for idx in sorted_indices:
        if idx < 0 or idx >= len(lines):
            continue
        # Add blank line separator when there's a gap
        if prev_idx >= 0 and idx > prev_idx + 1:
            result_lines.append("")
        result_lines.append(lines[idx])
        prev_idx = idx

    # Add the assertion line (with original indentation, up to target start)
    if result_lines and prev_idx < target_lineno - 1:
        result_lines.append("")
    result_lines.append(assertion_prefix_text)

    # Truncate from the top if too long (preserve test body + assertion)
    if len(result_lines) > max_prefix_lines:
        result_lines = result_lines[-max_prefix_lines:]

    return "\n".join(result_lines)


# ======================================================================
# Target difficulty classifier
# ======================================================================

def classify_difficulty(target: str) -> str:
    t = target.strip().rstrip(")")
    if t in ("True", "False"):
        return "bool_literal"
    if t == "None":
        return "none_literal"
    if t.startswith(("'", '"', "b'", 'b"', "f'", 'f"')):
        return "string_literal"
    try:
        float(t.replace("-", "", 1))
        return "numeric_literal"
    except ValueError:
        pass
    if t.startswith(("[", "(", "{")):
        return "collection"
    if "(" in target:
        return "func_call"
    if t.isidentifier():
        return "variable"
    return "complex_expr"


# ======================================================================
# Quality filters
# ======================================================================

def is_valid_pair(prefix: str, target: str, assertion: RawAssertion, func_node) -> tuple[bool, str]:
    """Return (is_valid, rejection_reason)."""
    target_stripped = target.strip()

    if not target_stripped:
        return False, "empty_target"
    if target_stripped.lstrip().startswith(","):
        return False, "comma_leading_target"
    if target_stripped in (")", "))", ")))", "]", "]]", "}"):
        return False, "paren_only_target"
    if len(target_stripped) > 500:
        return False, "target_too_long"
    if func_node is None:
        return False, "no_enclosing_function"

    last_line = prefix.rstrip().split("\n")[-1].strip()
    if "(," in last_line:
        return False, "malformed_prefix_comma"

    return True, ""


# ======================================================================
# Selection algorithm
# ======================================================================

def select_balanced_pairs(
    pairs: list[dict],
    max_per_repo: int = 200,
    max_per_function: int = 5,
    max_per_file: int = 20,
) -> list[dict]:
    """
    Select a balanced set of pairs:
    1. Cap per function (5) and per file (20)
    2. Deduplicate by (file, target)
    3. Stratify by difficulty tier AND assertion type
    4. Cap per repo (200)
    """
    if not pairs:
        return []

    def quality_key(p):
        return (
            p["metadata"].get("was_multiline", False),
            len(p["target"]),
        )

    # --- Step 1: Per-function cap ---
    by_func = defaultdict(list)
    for p in pairs:
        key = (p["metadata"]["file"], p["metadata"]["test_function"])
        by_func[key].append(p)

    func_capped = []
    for key, group in by_func.items():
        group.sort(key=quality_key)
        func_capped.extend(group[:max_per_function])

    # --- Step 2: Per-file cap ---
    by_file = defaultdict(list)
    for p in func_capped:
        by_file[p["metadata"]["file"]].append(p)

    file_capped = []
    for f, group in by_file.items():
        group.sort(key=quality_key)
        file_capped.extend(group[:max_per_file])

    # --- Step 3: Dedup by (file, target) ---
    seen = set()
    deduped = []
    for p in file_capped:
        key = (p["metadata"]["file"], p["target"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    # --- Step 4: Stratified selection across difficulty tiers ---
    by_difficulty = defaultdict(list)
    for p in deduped:
        by_difficulty[p["difficulty"]].append(p)

    total_available = len(deduped)
    if total_available <= max_per_repo:
        return deduped

    # Proportional allocation across difficulty tiers
    selected = []
    slots = {}
    frac_parts = []
    for diff, group in by_difficulty.items():
        exact = max_per_repo * len(group) / total_available
        floor_val = int(exact)
        slots[diff] = floor_val
        frac_parts.append((diff, exact - floor_val))

    remainder = max_per_repo - sum(slots.values())
    frac_parts.sort(key=lambda x: -x[1])
    for i in range(min(remainder, len(frac_parts))):
        slots[frac_parts[i][0]] += 1

    for diff, group in by_difficulty.items():
        group.sort(key=quality_key)
        n = min(slots.get(diff, 0), len(group))
        selected.extend(group[:n])

    return selected


# ======================================================================
# Main extraction pipeline
# ======================================================================

def extract_from_file(source: str, file_path: str) -> list[dict]:
    """Extract all valid QnA pairs from a single test file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    lines = source.splitlines()
    results = []

    for node in ast.walk(tree):
        raw = None

        if isinstance(node, ast.Assert):
            raw = _extract_plain_assert(source, node)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                val = node.func.value
                val_id = getattr(val, "id", None) if isinstance(val, ast.Name) else None
                if val_id == "self" and node.func.attr in SELF_ASSERT_METHODS:
                    raw = _extract_self_assert(source, node)
                elif val_id == "pytest":
                    if node.func.attr == "raises":
                        raw = _extract_pytest_raises(source, node)
                    elif node.func.attr == "approx":
                        raw = _extract_pytest_approx(source, node)
                    else:
                        raw = _extract_assert_underscore(source, node)
                else:
                    raw = _extract_assert_underscore(source, node)
            elif isinstance(node.func, ast.Name) and node.func.id.startswith("assert_"):
                raw = _extract_assert_underscore(source, node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert_"):
                raw = _extract_assert_underscore(source, node)

        if raw is None:
            continue

        target = raw.target_src
        was_multiline = "\n" in target
        if was_multiline:
            target = " ".join(target.split())
        if not target.strip():
            continue

        func_node, class_node = _find_enclosing(tree, raw.lineno)

        # Build structured prefix
        prefix = build_structured_prefix(
            source, lines, tree, raw, func_node, class_node,
        )

        # Quality filter
        valid, reason = is_valid_pair(prefix, target, raw, func_node)
        if not valid:
            continue

        difficulty = classify_difficulty(target)
        has_imports = "import " in prefix[:1000]

        # Track which sections are in the prefix
        sections = []
        if has_imports:
            sections.append("imports")
        if class_node:
            sections.append("class_header")
        if func_node:
            sections.append("test_body")

        results.append({
            "prefix": prefix,
            "target": target,
            "assertion_type": raw.assertion_type,
            "difficulty": difficulty,
            "context_lines": prefix.count("\n") + 1,
            "metadata": {
                "file": file_path,
                "lineno": raw.lineno,
                "col_offset": raw.col_offset,
                "target_col_offset": raw.target_col,
                "test_function": func_node.name if func_node else "",
                "test_class": class_node.name if class_node else "",
                "was_multiline": was_multiline,
                "has_imports": has_imports,
                "prefix_type": "structured",
                "prefix_sections": sections,
            },
        })

    return results


def process_repo(
    repo_path: str,
    author: str,
    repo_name: str,
    max_per_repo: int = 200,
    max_per_function: int = 5,
    max_per_file: int = 20,
) -> tuple[int, int, dict]:
    """Extract, filter, select, and save QnA pairs for one repo."""
    test_dir = os.path.join(repo_path, TEST_HYPERNET)
    if not os.path.isdir(test_dir):
        return 0, 0, {}

    all_pairs = []
    rejection_counts = Counter()

    for root, dirs, files in os.walk(test_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, test_dir)
            try:
                source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            pairs = extract_from_file(source, rel_path)
            for p in pairs:
                p["metadata"]["repo"] = f"{author}/{repo_name}"
                p["metadata"]["source_file"] = abs_path
            all_pairs.extend(pairs)

    if not all_pairs:
        return 0, 0, {}

    selected = select_balanced_pairs(
        all_pairs,
        max_per_repo=max_per_repo,
        max_per_function=max_per_function,
        max_per_file=max_per_file,
    )

    out_path = os.path.join(repo_path, QNA_HYPERNET)
    output = {
        "repo": f"{author}/{repo_name}",
        "total_extracted": len(all_pairs),
        "saved_count": len(selected),
        "selection_params": {
            "max_per_repo": max_per_repo,
            "max_per_function": max_per_function,
            "max_per_file": max_per_file,
        },
        "pairs": selected,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Collect stats
    stats = {
        "extracted": len(all_pairs),
        "selected": len(selected),
        "difficulty": Counter(p["difficulty"] for p in selected),
        "assertion_type": Counter(p["assertion_type"] for p in selected),
        "has_imports": sum(1 for p in selected if p["metadata"]["has_imports"]),
        "indented": sum(1 for p in selected
                        if p["prefix"].rstrip().split("\n")[-1][:1] in (" ", "\t", "")),
    }
    return len(all_pairs), len(selected), stats


# ======================================================================
# Validation
# ======================================================================

def validate_splits(splits_dir: str):
    """Run validation on generated split files."""
    splits_dir = Path(splits_dir)
    for split_name in ["train", "cr_val", "cr_test", "ir_val", "ir_test"]:
        path = splits_dir / f"{split_name}.json"
        if not path.exists():
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})

        total = 0
        issues = Counter()
        difficulties = Counter()
        assertion_types = Counter()

        for repo_name, r in repos.items():
            for p in r.get("qna_pairs", []):
                total += 1
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                last_line = prefix.rstrip().split("\n")[-1]

                if target.lstrip().startswith(","):
                    issues["comma_target"] += 1
                if "(," in last_line:
                    issues["malformed_prefix"] += 1
                if last_line and last_line[0] not in (" ", "\t"):
                    issues["no_indent"] += 1
                if "import " not in prefix[:1000]:
                    issues["no_imports"] += 1

                difficulties[p.get("difficulty", classify_difficulty(target))] += 1
                assertion_types[p.get("assertion_type", "unknown")] += 1

        print(f"\n{'=' * 50}")
        print(f"  {split_name}: {total} pairs, {len(repos)} repos")
        print(f"{'=' * 50}")

        if issues:
            print("  ISSUES:")
            for k, v in issues.most_common():
                print(f"    {k}: {v} ({100 * v / total:.1f}%)")
        else:
            print("  No issues found")

        print("  Difficulty distribution:")
        for k, v in difficulties.most_common():
            print(f"    {k}: {v} ({100 * v / total:.1f}%)")

        print("  Assertion types:")
        for k, v in assertion_types.most_common(10):
            print(f"    {k}: {v} ({100 * v / total:.1f}%)")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="V2 QnA extraction: clean, structured, balanced")
    parser.add_argument("--repos-dir", default=REPOSITORIES_DIR)
    parser.add_argument("--max-per-repo", type=int, default=200)
    parser.add_argument("--max-per-function", type=int, default=5)
    parser.add_argument("--max-per-file", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--validate", action="store_true", help="Run validation on existing splits")
    parser.add_argument("--splits-dir", type=str, default=None)
    args = parser.parse_args()

    if args.validate:
        splits_dir = args.splits_dir or os.path.join(
            os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "REPO_DATASET")
        validate_splits(splits_dir)
        return

    repos_dir = args.repos_dir
    if not os.path.isdir(repos_dir):
        print(f"Repos dir not found: {repos_dir}")
        return

    # Collect repos
    if args.repo:
        parts = args.repo.split("/", 1)
        if len(parts) != 2:
            print("--repo must be owner/repo_name")
            return
        author, repo_name = parts
        repo_path = os.path.join(repos_dir, author, repo_name)
        repos_iter = [(author, repo_name, repo_path)]
    else:
        repos_iter = []
        for author in sorted(os.listdir(repos_dir)):
            author_path = os.path.join(repos_dir, author)
            if not os.path.isdir(author_path):
                continue
            for repo_name in sorted(os.listdir(author_path)):
                repo_path = os.path.join(author_path, repo_name)
                test_dir = os.path.join(repo_path, TEST_HYPERNET)
                if os.path.isdir(test_dir) and repo_name != TEST_HYPERNET:
                    repos_iter.append((author, repo_name, repo_path))
        if args.limit:
            repos_iter = repos_iter[:args.limit]

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    total_extracted = 0
    total_selected = 0
    processed = 0
    agg_difficulty = Counter()
    agg_assertion = Counter()
    agg_imports = 0
    agg_indented = 0

    for author, repo_name, repo_path in tqdm(repos_iter, desc="Repos"):
        n_ext, n_sel, stats = process_repo(
            repo_path, author, repo_name,
            max_per_repo=args.max_per_repo,
            max_per_function=args.max_per_function,
            max_per_file=args.max_per_file,
        )
        total_extracted += n_ext
        total_selected += n_sel
        if n_sel > 0:
            processed += 1
            agg_difficulty.update(stats["difficulty"])
            agg_assertion.update(stats["assertion_type"])
            agg_imports += stats["has_imports"]
            agg_indented += stats["indented"]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Repos processed: {processed}")
    print(f"  Total extracted: {total_extracted}")
    print(f"  Total selected:  {total_selected}")
    print(f"  Imports present: {agg_imports}/{total_selected} ({100 * agg_imports / max(1, total_selected):.1f}%)")
    print(f"  Properly indented: {agg_indented}/{total_selected} ({100 * agg_indented / max(1, total_selected):.1f}%)")
    print(f"\n  Difficulty distribution:")
    for k, v in agg_difficulty.most_common():
        print(f"    {k}: {v} ({100 * v / max(1, total_selected):.1f}%)")
    print(f"\n  Assertion types:")
    for k, v in agg_assertion.most_common():
        print(f"    {k}: {v} ({100 * v / max(1, total_selected):.1f}%)")


if __name__ == "__main__":
    main()
