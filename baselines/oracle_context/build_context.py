#!/usr/bin/env python3
"""
Pre-build oracle context for each QnA pair by resolving imports in the test
file and extracting the exact source code of functions/classes being tested.

Run once, then test_oracle_context.py loads the cached contexts instantly.

Usage:
    python baselines/oracle_context/build_context.py
    python baselines/oracle_context/build_context.py --repos-root $SCRATCH/REPO_DATASET/repositories
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".nox", ".hg", ".svn", "site-packages", "TEST_HYPERNET",
}

# --------------------------------------------------------------------------
# Step A: Parse imports from test prefix
# --------------------------------------------------------------------------

def parse_imports(source: str) -> list[dict]:
    """
    Parse all import statements from Python source.
    Returns list of {type, module, names, level} dicts.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _parse_imports_fallback(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "asname": alias.asname,
                    "names": [alias.name.split(".")[-1]],
                    "level": 0,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names] if node.names else []
            imports.append({
                "type": "from",
                "module": module,
                "names": names,
                "level": node.level or 0,
            })
    return imports


def _parse_imports_fallback(source: str) -> list[dict]:
    """Regex fallback for unparseable prefixes (truncated code)."""
    import re
    imports = []
    for line in source.splitlines():
        line = line.strip()
        m = re.match(r"^from\s+(\.+)?(\S*)\s+import\s+(.+)$", line)
        if m:
            dots = m.group(1) or ""
            module = m.group(2) or ""
            names = [n.strip().split(" as ")[0].strip() for n in m.group(3).split(",")]
            imports.append({
                "type": "from",
                "module": module,
                "names": names,
                "level": len(dots),
            })
            continue
        m = re.match(r"^import\s+(\S+)", line)
        if m:
            mod = m.group(1).split(" as ")[0].strip()
            imports.append({
                "type": "import",
                "module": mod,
                "names": [mod.split(".")[-1]],
                "level": 0,
            })
    return imports


# --------------------------------------------------------------------------
# Step B: Resolve module paths to repo files
# --------------------------------------------------------------------------

_SOURCE_ROOTS = {"src", "lib", "app", "python", "py"}


def _find_source_roots(repo_dir: Path) -> list[Path]:
    """
    Detect source root directories inside a repo.

    Returns a list of directories that should be treated as additional
    import roots beyond the repo root itself.  Handles:
      - Well-known names: src/, lib/, app/, python/, py/
      - Any top-level directory that contains a Python package
        (i.e. has a subdirectory with __init__.py) but is NOT itself
        a package (no __init__.py of its own).  This catches monorepo
        layouts like  packages/mylib/__init__.py.
    """
    roots: list[Path] = []

    for child in repo_dir.iterdir():
        if not child.is_dir() or child.name in SKIP_DIRS or child.name == "TEST_HYPERNET":
            continue

        if child.name in _SOURCE_ROOTS:
            roots.append(child)
            continue

        # Skip if this directory is already a Python package itself
        if (child / "__init__.py").exists():
            continue

        # Monorepo heuristic: directory contains at least one Python
        # sub-package (child_dir/sub/__init__.py) and is not a known
        # non-source directory.
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "__init__.py").exists():
                roots.append(child)
                break

    return roots


def _build_repo_file_index(repo_dir: Path) -> dict[str, Path]:
    """
    Build a map of dotted module paths to file paths.
    e.g. "mypackage.utils" -> repo_dir / "mypackage" / "utils.py"

    Files are indexed relative to the repo root AND relative to each
    detected source root (src/, lib/, monorepo dirs, ...).  The source-root
    entries take precedence when there is a collision so that
    ``from mypackage import X`` resolves even when mypackage lives
    under ``src/mypackage/``.
    """
    index: dict[str, Path] = {}

    # Collect all Python files once
    all_py_files: list[Path] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                all_py_files.append(Path(root) / fn)

    # Index relative to repo root (baseline)
    for fp in all_py_files:
        dotted = _path_to_dotted(fp, repo_dir)
        if dotted:
            index[dotted] = fp

    # Index relative to each source root (higher priority)
    source_roots = _find_source_roots(repo_dir)
    for src_root in source_roots:
        for fp in all_py_files:
            try:
                fp.relative_to(src_root)
            except ValueError:
                continue
            dotted = _path_to_dotted(fp, src_root)
            if dotted:
                index[dotted] = fp  # overwrites repo-root entry on collision

    return index


def _path_to_dotted(filepath: Path, base: Path) -> str:
    """Convert a .py file path to a dotted module name relative to *base*."""
    rel = filepath.relative_to(base)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        dotted = ".".join(parts[:-1])
    else:
        dotted = ".".join(parts)[:-3]  # strip .py
    return dotted


def _module_exists_in_repo(top_level: str, repo_dir: Path, source_roots: list[Path]) -> bool:
    """Check whether *top_level* corresponds to a directory or .py file
    in the repo root or any detected source root."""
    for base in [repo_dir] + source_roots:
        if (base / top_level).exists() or (base / f"{top_level}.py").exists():
            return True
    return False


def resolve_import_to_files(
    imp: dict,
    repo_dir: Path,
    file_index: dict[str, Path],
    test_file_rel: str,
    source_roots: list[Path] | None = None,
) -> list[tuple[Path, list[str]]]:
    """
    Resolve one import dict to list of (file_path, [names_to_extract]).
    Returns empty list if it's third-party / stdlib.
    """
    module = imp["module"]
    level = imp["level"]
    names = imp["names"]

    # Handle relative imports
    if level > 0:
        test_parts = Path(test_file_rel).parts
        # Go up `level` directories from test file's directory
        if len(test_parts) > level:
            base_parts = test_parts[:-(level)]
        else:
            base_parts = ()
        if module:
            full_module = ".".join(base_parts) + "." + module if base_parts else module
        else:
            full_module = ".".join(base_parts) if base_parts else ""
    else:
        full_module = module

    if not full_module:
        return []

    results = []

    # Check if top-level module exists in repo (skip stdlib / third-party)
    top_level = full_module.split(".")[0]
    if not _module_exists_in_repo(top_level, repo_dir, source_roots or []):
        # Also check conftest.py specially
        if top_level == "conftest":
            conftest = _find_conftest(repo_dir, test_file_rel)
            if conftest:
                return [(conftest, names)]
        # Last resort: the module might still be in file_index if it was
        # indexed from a source root (e.g. src/mypackage -> "mypackage").
        if full_module not in file_index:
            return []

    if imp["type"] == "from":
        # from mypackage.utils import foo, bar
        # -> find mypackage/utils.py, extract foo, bar
        if full_module in file_index:
            results.append((file_index[full_module], names))
        # Maybe it's a package: from mypackage import submodule
        for name in names:
            sub_module = f"{full_module}.{name}"
            if sub_module in file_index:
                results.append((file_index[sub_module], []))
        # Check __init__.py for re-exports
        if not results and full_module in file_index:
            init_path = file_index[full_module]
            reexports = _trace_init_reexports(init_path, names, file_index, full_module)
            results.extend(reexports)
    else:
        # import mypackage.core
        if full_module in file_index:
            results.append((file_index[full_module], names))

    return results


def _find_conftest(repo_dir: Path, test_file_rel: str) -> Optional[Path]:
    """Find conftest.py walking up from the test file's directory."""
    test_dir = repo_dir / Path(test_file_rel).parent
    current = test_dir
    while current != repo_dir.parent:
        conftest = current / "conftest.py"
        if conftest.exists():
            return conftest
        current = current.parent
    return None


def _trace_init_reexports(
    init_path: Path,
    names: list[str],
    file_index: dict[str, Path],
    package_module: str,
) -> list[tuple[Path, list[str]]]:
    """
    If __init__.py re-exports names (from .submodule import Foo),
    follow the chain to the actual source file.
    """
    try:
        source = init_path.read_text(encoding="utf-8", errors="ignore")
        init_imports = parse_imports(source)
    except Exception:
        return []

    results = []
    for imp in init_imports:
        if imp["type"] != "from" or imp["level"] == 0:
            continue
        overlap = set(imp["names"]) & set(names)
        if not overlap:
            continue
        sub_module = imp["module"]
        if sub_module:
            full_sub = f"{package_module}.{sub_module}"
        else:
            continue
        if full_sub in file_index:
            results.append((file_index[full_sub], list(overlap)))
    return results


# --------------------------------------------------------------------------
# Step C: Find names the test actually uses
# --------------------------------------------------------------------------

def find_used_names_in_prefix(source: str, imported_names: set[str]) -> set[str]:
    """
    Walk the AST of the test prefix and find which imported names are
    actually referenced (called, accessed, used as arguments, etc.).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Prefix is truncated code -- use simple string matching
        return {n for n in imported_names if n in source}

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in imported_names:
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in imported_names:
            used.add(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in imported_names:
                used.add(node.func.id)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in imported_names:
                used.add(node.func.attr)
    return used


# --------------------------------------------------------------------------
# Step D: Extract definitions from source files
# --------------------------------------------------------------------------

def extract_definitions(source_path: Path, names: list[str]) -> str:
    """
    Extract function/class definitions from a source file that match
    the given names. If names is empty, return a summary of all top-level
    definitions (first 100 lines).
    """
    try:
        source = source_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    if not names:
        lines = source.splitlines()
        return "\n".join(lines[:150])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse -- return lines containing the names
        return _grep_names(source, names)

    name_set = set(names)
    extracted = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in name_set:
                seg = _get_node_source(source, node)
                if seg:
                    extracted.append(seg)
        elif isinstance(node, ast.ClassDef):
            if node.name in name_set:
                seg = _get_node_source(source, node)
                if seg:
                    extracted.append(seg)
            else:
                # Check if any method names match
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name in name_set:
                            # Include full class
                            seg = _get_node_source(source, node)
                            if seg:
                                extracted.append(seg)
                            break
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in name_set:
                    seg = _get_node_source(source, node)
                    if seg:
                        extracted.append(seg)

    if not extracted:
        return _grep_names(source, names)

    return "\n\n".join(extracted)


def _get_node_source(source: str, node: ast.AST) -> Optional[str]:
    """Extract source for an AST node."""
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(source, node)
        if seg:
            return seg
    start = getattr(node, "lineno", 1) - 1
    end_line = getattr(node, "end_lineno", start + 1)
    lines = source.splitlines()
    if start < len(lines):
        return "\n".join(lines[start:min(end_line, len(lines))])
    return None


def _grep_names(source: str, names: list[str]) -> str:
    """Fallback: find lines containing any of the names."""
    lines = source.splitlines()
    matched = []
    for i, line in enumerate(lines):
        if any(n in line for n in names):
            start = max(0, i - 2)
            end = min(len(lines), i + 10)
            matched.append("\n".join(lines[start:end]))
    return "\n\n".join(matched[:5])


# --------------------------------------------------------------------------
# Main: build context for all QnA pairs
# --------------------------------------------------------------------------

def build_context_for_pair(
    prefix: str,
    metadata: dict,
    repo_dir: Path,
    file_index: dict[str, Path],
    test_file_cache: dict[str, str],
    source_roots: list[Path] | None = None,
) -> dict:
    """Build oracle context for a single QnA pair."""
    test_file_rel = metadata.get("file", "")

    # Step A: parse imports from the FULL test file (not truncated prefix)
    full_test_source = test_file_cache.get(test_file_rel)
    if full_test_source is None:
        test_path = repo_dir / "TEST_HYPERNET" / test_file_rel
        if test_path.exists():
            try:
                full_test_source = test_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                full_test_source = ""
        else:
            full_test_source = ""
        test_file_cache[test_file_rel] = full_test_source

    imports = parse_imports(full_test_source) if full_test_source else parse_imports(prefix)

    # Collect all imported names
    all_imported_names = set()
    for imp in imports:
        all_imported_names.update(imp["names"])

    # Step C: find which names the test actually uses (check both full file and prefix)
    used_names = find_used_names_in_prefix(full_test_source or prefix, all_imported_names)

    # Step B + D: resolve imports and extract code
    resolved_files = []
    extracted_parts = []
    seen_files = set()

    for imp in imports:
        file_matches = resolve_import_to_files(
            imp, repo_dir, file_index, test_file_rel,
            source_roots=source_roots,
        )
        for fpath, imp_names in file_matches:
            if fpath in seen_files:
                continue
            seen_files.add(fpath)

            rel = str(fpath.relative_to(repo_dir))
            resolved_files.append(rel)

            # Filter to names actually used
            if imp_names:
                relevant_names = [n for n in imp_names if n in used_names] or imp_names
            else:
                relevant_names = []

            code = extract_definitions(fpath, relevant_names)
            if code.strip():
                header = f"# Source: {rel}"
                extracted_parts.append(f"{header}\n{code}")

    extracted_code = "\n\n\n".join(extracted_parts)

    return {
        "resolved_imports": resolved_files,
        "used_names": sorted(used_names),
        "extracted_code": extracted_code,
        "n_imports_parsed": len(imports),
        "n_files_resolved": len(resolved_files),
        "n_chars_extracted": len(extracted_code),
    }


def main():
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )
    default_splits = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_cache = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "ORACLE_CONTEXT_CACHE",
    )

    ap = argparse.ArgumentParser(description="Pre-build oracle context for import-aware baseline")
    ap.add_argument("--repos-root", type=str, default=default_repos)
    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--cache-dir", type=str, default=default_cache)
    args = ap.parse_args()

    repos_root = Path(args.repos_root).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all QnA pairs grouped by repo from all splits
    repo_pairs: dict[str, list[dict]] = {}
    for split_name in ["train", "cr_val", "cr_test", "ir_val", "ir_test"]:
        path = splits_dir / f"{split_name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for repo_name, r in data.get("repositories", {}).items():
            if repo_name not in repo_pairs:
                repo_pairs[repo_name] = []
            for p in r.get("qna_pairs", []):
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                metadata = p.get("metadata", {})
                if prefix and target and not target.lstrip().startswith(","):
                    repo_pairs[repo_name].append({
                        "prefix": prefix,
                        "metadata": metadata,
                        "key": f"{metadata.get('file', '')}::{metadata.get('lineno', 0)}",
                    })

    print(f"Found {len(repo_pairs)} repos, {sum(len(v) for v in repo_pairs.values())} total pairs")

    built = 0
    skipped = 0
    stats = {"total_pairs": 0, "pairs_with_context": 0, "avg_chars": 0.0}

    for repo_name in tqdm(sorted(repo_pairs.keys()), desc="Building oracle context"):
        safe_name = repo_name.replace("/", "__")
        cache_path = cache_dir / f"{safe_name}.json"

        if cache_path.exists():
            skipped += 1
            continue

        author, rname = repo_name.split("/", 1)
        repo_dir = repos_root / author / rname
        if not repo_dir.exists():
            continue

        file_index = _build_repo_file_index(repo_dir)
        source_roots = _find_source_roots(repo_dir)
        test_file_cache: dict[str, str] = {}

        contexts = {}
        pairs = repo_pairs[repo_name]
        total_chars = 0

        for pair in pairs:
            ctx = build_context_for_pair(
                pair["prefix"], pair["metadata"], repo_dir, file_index, test_file_cache,
                source_roots=source_roots,
            )
            contexts[pair["key"]] = ctx
            stats["total_pairs"] += 1
            if ctx["n_chars_extracted"] > 0:
                stats["pairs_with_context"] += 1
                total_chars += ctx["n_chars_extracted"]

        cache_data = {
            "repo": repo_name,
            "n_pairs": len(contexts),
            "contexts": contexts,
        }
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
        built += 1

    if stats["pairs_with_context"] > 0:
        stats["avg_chars"] = stats["avg_chars"] / stats["pairs_with_context"] if stats["avg_chars"] else 0

    print(f"\nDone. Built: {built}, Skipped (cached): {skipped}")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Pairs with extracted context: {stats['pairs_with_context']} "
          f"({100 * stats['pairs_with_context'] / max(1, stats['total_pairs']):.1f}%)")
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
