#!/usr/bin/env python3
"""
Rebuild QnA prefixes using a structured approach instead of naive 300-line truncation.

Reads the full test file from disk and constructs a prefix from:
  1. File-level imports (always kept)
  2. Module-level constants/fixtures used by the test
  3. Enclosing class header + setUp/setUpClass (if unittest-style)
  4. The test function body up to the assertion line

Unrelated test functions in the same file are skipped.

Outputs new split files with "_structured" suffix (e.g. cr_test_structured.json)
alongside the originals, so existing experiments are unaffected.

Usage:
    python create_dataset/rebuild_prefixes.py
    python create_dataset/rebuild_prefixes.py --splits cr_test ir_test  # only specific splits
    python create_dataset/rebuild_prefixes.py --max-prefix-tokens 4096
"""

import argparse
import ast
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_HYPERNET = "TEST_HYPERNET"


def _get_node_lines(source_lines: list[str], node: ast.AST) -> str:
    """Extract source lines for an AST node."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(source_lines[start:end])


def build_structured_prefix(
    source: str,
    assertion_lineno: int,
    assertion_col_offset: int,
    test_function_name: str,
    max_prefix_lines: int = 500,
) -> Optional[str]:
    """
    Build a structured prefix that keeps the most important parts:
      1. Imports + module-level assignments (top of file)
      2. Enclosing class header + setUp (if applicable)
      3. Helper functions/fixtures referenced before the assertion
      4. The test function body up to the assertion

    Skips unrelated test functions to save space.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    lines = source.splitlines()
    if assertion_lineno > len(lines):
        return None

    # --- Section 1: Imports + module-level statements ---
    import_lines = set()
    module_level_end = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = getattr(node, "lineno", 1) - 1
            end = getattr(node, "end_lineno", start + 1)
            for i in range(start, end):
                import_lines.add(i)
            module_level_end = max(module_level_end, end)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            end = getattr(node, "end_lineno", getattr(node, "lineno", 1))
            module_level_end = max(module_level_end, end)

    # Collect contiguous import block (imports + module constants before first def/class)
    import_section_lines = []
    for i in range(min(module_level_end, len(lines))):
        import_section_lines.append(lines[i])

    # --- Section 2: Find enclosing class ---
    enclosing_class = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            cls_start = getattr(node, "lineno", 0)
            cls_end = getattr(node, "end_lineno", 0)
            if cls_start <= assertion_lineno <= cls_end:
                enclosing_class = node
                break

    class_header_lines = []
    setup_lines = []
    if enclosing_class:
        # Class header (class ClassName(Base):)
        cls_start = getattr(enclosing_class, "lineno", 1) - 1
        # Just the class definition line + decorators
        class_header_lines = [lines[cls_start]]

        # setUp / setUpClass / class-level attributes
        for item in enclosing_class.body:
            item_start = getattr(item, "lineno", 0)
            if item_start >= assertion_lineno:
                break
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in ("setUp", "setUpClass", "tearDown", "setup_method",
                                 "setup", "fixture", "__init__"):
                    setup_lines.append(_get_node_lines(lines, item))
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                setup_lines.append(_get_node_lines(lines, item))

    # --- Section 3: Helper functions/fixtures defined before the test ---
    # Collect non-test functions that appear before the assertion
    helper_lines = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_end = getattr(node, "end_lineno", 0)
            if func_end >= assertion_lineno:
                continue
            # Skip other test functions -- they're noise
            if node.name.startswith("test_") and node.name != test_function_name:
                continue
            # Keep fixtures (decorated with @pytest.fixture etc.) and helpers
            is_fixture = any(
                (isinstance(d, ast.Attribute) and d.attr == "fixture") or
                (isinstance(d, ast.Name) and d.id == "fixture") or
                (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "fixture")
                for d in node.decorator_list
            )
            is_helper = not node.name.startswith("test_")
            if is_fixture or is_helper:
                helper_lines.append(_get_node_lines(lines, node))

    # --- Section 4: The test function body up to the assertion ---
    test_func_lines = []
    # Find the test function node
    test_func_node = None
    all_funcs = list(ast.walk(tree))
    for node in all_funcs:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == test_function_name:
                func_start = getattr(node, "lineno", 0)
                func_end = getattr(node, "end_lineno", 0)
                if func_start <= assertion_lineno <= func_end:
                    test_func_node = node
                    break

    if test_func_node:
        func_start = getattr(test_func_node, "lineno", 1) - 1
        # Include decorators
        for dec in test_func_node.decorator_list:
            dec_line = getattr(dec, "lineno", func_start + 1) - 1
            func_start = min(func_start, dec_line)
        # Lines from function start up to (but not including) the assertion line
        assertion_line_idx = assertion_lineno - 1
        last_line = lines[assertion_line_idx] if assertion_line_idx < len(lines) else ""
        last_line_prefix = last_line[:assertion_col_offset] if assertion_col_offset <= len(last_line) else last_line
        test_func_lines = lines[func_start:assertion_line_idx]
        test_func_lines.append(last_line_prefix.rstrip())
    else:
        # Fallback: just take lines before assertion
        start = max(0, assertion_lineno - 1 - 100)
        test_func_lines = lines[start:assertion_lineno - 1]
        last_line = lines[assertion_lineno - 1] if assertion_lineno - 1 < len(lines) else ""
        last_line_prefix = last_line[:assertion_col_offset] if assertion_col_offset <= len(last_line) else last_line
        test_func_lines.append(last_line_prefix.rstrip())

    # --- Assemble ---
    sections = []
    if import_section_lines:
        sections.append("\n".join(import_section_lines))
    if class_header_lines:
        sections.append("\n".join(class_header_lines))
    if setup_lines:
        sections.append("\n".join(setup_lines))
    if helper_lines:
        sections.append("\n".join(helper_lines))
    if test_func_lines:
        sections.append("\n".join(test_func_lines))

    prefix = "\n\n".join(sections) + "\n"

    # Truncate from the top if still too long (preserve test function body)
    prefix_lines = prefix.splitlines()
    if len(prefix_lines) > max_prefix_lines:
        prefix_lines = prefix_lines[-max_prefix_lines:]
        prefix = "\n".join(prefix_lines) + "\n"

    return prefix


def main():
    default_splits = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )

    ap = argparse.ArgumentParser(description="Rebuild QnA prefixes with structured approach")
    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--repos-root", type=str, default=default_repos)
    ap.add_argument("--splits", type=str, nargs="+",
                    default=["cr_test", "cr_val", "ir_test", "ir_val", "train"])
    ap.add_argument("--max-prefix-lines", type=int, default=500)
    ap.add_argument("--suffix", type=str, default="_structured",
                    help="Suffix for output files (e.g. cr_test_structured.json)")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = Path(args.repos_root).expanduser().resolve()

    for split_name in args.splits:
        path = splits_dir / f"{split_name}.json"
        if not path.exists():
            print(f"Skip {split_name}: not found")
            continue

        print(f"\nProcessing {split_name}...")
        data = json.loads(path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})

        total = 0
        rebuilt = 0
        fallback = 0

        # Cache test file sources per repo
        test_file_cache: dict[str, str] = {}

        for repo_name in tqdm(sorted(repos.keys()), desc=split_name):
            author, rname = repo_name.split("/", 1)
            repo_dir = repos_root / author / rname
            test_dir = repo_dir / TEST_HYPERNET

            r = repos[repo_name]
            pairs = r.get("qna_pairs", [])

            for p in pairs:
                total += 1
                metadata = p.get("metadata", {})
                test_file_rel = metadata.get("file", "")
                lineno = metadata.get("lineno", 0)
                col_offset = metadata.get("col_offset", 0)
                test_function = metadata.get("test_function", "")

                if not test_file_rel or lineno <= 0:
                    fallback += 1
                    continue

                # Read full test file
                cache_key = f"{repo_name}::{test_file_rel}"
                if cache_key not in test_file_cache:
                    test_path = test_dir / test_file_rel
                    if test_path.exists():
                        try:
                            test_file_cache[cache_key] = test_path.read_text(
                                encoding="utf-8", errors="ignore")
                        except Exception:
                            test_file_cache[cache_key] = ""
                    else:
                        test_file_cache[cache_key] = ""

                source = test_file_cache[cache_key]
                if not source:
                    fallback += 1
                    continue

                new_prefix = build_structured_prefix(
                    source, lineno, col_offset, test_function,
                    max_prefix_lines=args.max_prefix_lines,
                )

                if new_prefix:
                    # Rebuild full prefix: structured context + assertion prefix
                    assertion_prefix = p.get("prefix", "").split("\n")[-1] if p.get("prefix") else ""
                    if assertion_prefix and not new_prefix.rstrip().endswith(assertion_prefix.rstrip()):
                        new_prefix = new_prefix.rstrip() + "\n" + assertion_prefix.rstrip() + "\n"
                    p["prefix"] = new_prefix
                    p["metadata"]["prefix_type"] = "structured"
                    rebuilt += 1
                else:
                    p["metadata"]["prefix_type"] = "original"
                    fallback += 1

        out_name = f"{split_name}{args.suffix}.json"
        out_path = splits_dir / out_name
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  Total: {total}, Rebuilt: {rebuilt} ({100 * rebuilt / max(1, total):.1f}%), "
              f"Fallback: {fallback}")
        print(f"  Saved: {out_path}")

    print("\nDone. Original splits unchanged.")


if __name__ == "__main__":
    main()
