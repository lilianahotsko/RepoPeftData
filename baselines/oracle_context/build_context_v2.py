#!/usr/bin/env python3
"""
Function-scoped oracle context builder (v2).

Improvements over v1 (build_context.py):
  1. Scopes to the enclosing function -- only imports/names used in the
     function containing the assertion are included.
  2. Always follows __init__.py re-exports to actual source files instead
     of dumping re-export lines.
  3. No fallback to "include everything" when no names match.

Writes to ORACLE_CONTEXT_CACHE_V2 by default.

Usage:
    python baselines/oracle_context/build_context_v2.py
    python baselines/oracle_context/build_context_v2.py --cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V2
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

_SOURCE_ROOTS = {"src", "lib", "app", "python", "py"}


# --------------------------------------------------------------------------
# Reused from v1: file indexing, source roots, module resolution helpers
# --------------------------------------------------------------------------

def _find_source_roots(repo_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for child in repo_dir.iterdir():
        if not child.is_dir() or child.name in SKIP_DIRS or child.name == "TEST_HYPERNET":
            continue
        if child.name in _SOURCE_ROOTS:
            roots.append(child)
            continue
        if (child / "__init__.py").exists():
            continue
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "__init__.py").exists():
                roots.append(child)
                break
    return roots


def _build_repo_file_index(repo_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    all_py_files: list[Path] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                all_py_files.append(Path(root) / fn)
    for fp in all_py_files:
        dotted = _path_to_dotted(fp, repo_dir)
        if dotted:
            index[dotted] = fp
    source_roots = _find_source_roots(repo_dir)
    for src_root in source_roots:
        for fp in all_py_files:
            try:
                fp.relative_to(src_root)
            except ValueError:
                continue
            dotted = _path_to_dotted(fp, src_root)
            if dotted:
                index[dotted] = fp
    return index


def _path_to_dotted(filepath: Path, base: Path) -> str:
    rel = filepath.relative_to(base)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        dotted = ".".join(parts[:-1])
    else:
        dotted = ".".join(parts)[:-3]
    return dotted


def _module_exists_in_repo(top_level: str, repo_dir: Path, source_roots: list[Path]) -> bool:
    for base in [repo_dir] + source_roots:
        if (base / top_level).exists() or (base / f"{top_level}.py").exists():
            return True
    return False


def _find_conftest(repo_dir: Path, test_file_rel: str) -> Optional[Path]:
    test_dir = repo_dir / Path(test_file_rel).parent
    current = test_dir
    while current != repo_dir.parent:
        conftest = current / "conftest.py"
        if conftest.exists():
            return conftest
        current = current.parent
    return None


def _get_node_source(source: str, node: ast.AST) -> Optional[str]:
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
    lines = source.splitlines()
    matched = []
    for i, line in enumerate(lines):
        if any(n in line for n in names):
            start = max(0, i - 2)
            end = min(len(lines), i + 10)
            matched.append("\n".join(lines[start:end]))
    return "\n\n".join(matched[:5])


# --------------------------------------------------------------------------
# NEW Step 1: Find the enclosing function for a given line number
# --------------------------------------------------------------------------

def find_enclosing_function(
    source: str,
    lineno: int,
    tree: Optional[ast.Module] = None,
) -> Optional[ast.AST]:
    """
    Return the innermost FunctionDef/AsyncFunctionDef whose body contains
    *lineno*.  Returns None if the assertion is at module level.

    Pass a pre-parsed *tree* to avoid re-parsing the source.
    """
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

    best: Optional[ast.AST] = None
    best_start = -1

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", 0)
        if start <= lineno <= end and start > best_start:
            best = node
            best_start = start

    return best


# --------------------------------------------------------------------------
# NEW Step 2: Collect scoped imports (module-level + function-local)
# --------------------------------------------------------------------------

def _is_import_node(node: ast.AST) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def _import_node_to_dict(node: ast.AST) -> list[dict]:
    """Convert an AST Import/ImportFrom node to the same dict format as parse_imports."""
    results = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            results.append({
                "type": "import",
                "module": alias.name,
                "asname": alias.asname,
                "names": [alias.name.split(".")[-1]],
                "level": 0,
            })
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = [a.name for a in node.names] if node.names else []
        results.append({
            "type": "from",
            "module": module,
            "names": names,
            "level": node.level or 0,
        })
    return results


def collect_scoped_imports(
    tree: ast.AST,
    func_node: Optional[ast.AST],
) -> list[dict]:
    """
    Collect imports that are in scope for *func_node*:
      - Module-level imports (direct children of the module)
      - Imports inside *func_node*'s body (local imports)

    If *func_node* is None (assertion at module level), collect all
    module-level imports.
    """
    imports: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if _is_import_node(node):
            imports.extend(_import_node_to_dict(node))

    if func_node is not None:
        for node in ast.walk(func_node):
            if node is func_node:
                continue
            if _is_import_node(node):
                imports.extend(_import_node_to_dict(node))

    return imports


# --------------------------------------------------------------------------
# NEW Step 3: Find used names scoped to function body only
# --------------------------------------------------------------------------

def find_used_names_in_scope(
    scope_node: Optional[ast.AST],
    full_source: str,
    imported_names: set[str],
) -> set[str]:
    """
    Walk only *scope_node*'s AST subtree to find which imported names
    are referenced.  If *scope_node* is None (module-level assertion or
    unparseable source), fall back to string matching against the prefix.
    """
    if scope_node is None:
        return {n for n in imported_names if n in full_source}

    used: set[str] = set()
    for node in ast.walk(scope_node):
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
# Fallback import parser for truncated / unparseable prefixes
# --------------------------------------------------------------------------

def _parse_imports_fallback(source: str) -> list[dict]:
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
# IMPROVED Step 4: Resolve imports with __init__.py follow-through
# --------------------------------------------------------------------------

def _trace_init_reexports(
    init_path: Path,
    names: list[str],
    file_index: dict[str, Path],
    package_module: str,
) -> list[tuple[Path, list[str]]]:
    try:
        source = init_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return []

    init_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imp_names = [a.name for a in node.names] if node.names else []
            init_imports.append({
                "type": "from",
                "module": module,
                "names": imp_names,
                "level": node.level or 0,
            })

    results = []
    for imp in init_imports:
        if imp["type"] != "from" or imp["level"] == 0:
            continue
        overlap = set(imp["names"]) & set(names)
        if not overlap:
            continue
        sub_module = imp["module"]
        if not sub_module:
            continue
        full_sub = f"{package_module}.{sub_module}"
        if full_sub in file_index:
            results.append((file_index[full_sub], list(overlap)))
    return results


def _is_init_file(fpath: Path) -> bool:
    return fpath.name == "__init__.py"


def resolve_import_to_files(
    imp: dict,
    repo_dir: Path,
    file_index: dict[str, Path],
    test_file_rel: str,
    source_roots: list[Path] | None = None,
) -> list[tuple[Path, list[str]]]:
    """
    Resolve one import dict to (file_path, [names_to_extract]).
    Always follows __init__.py re-exports to actual source files.
    """
    module = imp["module"]
    level = imp["level"]
    names = imp["names"]

    if level > 0:
        test_parts = Path(test_file_rel).parts
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

    top_level = full_module.split(".")[0]
    if not _module_exists_in_repo(top_level, repo_dir, source_roots or []):
        if top_level == "conftest":
            conftest = _find_conftest(repo_dir, test_file_rel)
            if conftest:
                return [(conftest, names)]
        if full_module not in file_index:
            return []

    results = []

    if imp["type"] == "from":
        resolved_path = file_index.get(full_module)
        if resolved_path is not None:
            if _is_init_file(resolved_path) and names:
                reexports = _trace_init_reexports(resolved_path, names, file_index, full_module)
                if reexports:
                    results.extend(reexports)
                    resolved_names = set()
                    for _, rnames in reexports:
                        resolved_names.update(rnames)
                    remaining = [n for n in names if n not in resolved_names]
                    if remaining:
                        results.append((resolved_path, remaining))
                else:
                    results.append((resolved_path, names))
            else:
                results.append((resolved_path, names))

        for name in names:
            sub_module = f"{full_module}.{name}"
            if sub_module in file_index:
                already = any(file_index[sub_module] == fp for fp, _ in results)
                if not already:
                    results.append((file_index[sub_module], []))
    else:
        if full_module in file_index:
            results.append((file_index[full_module], names))

    return results


# --------------------------------------------------------------------------
# IMPROVED: Extract definitions (unchanged logic, but no empty-names dump)
# --------------------------------------------------------------------------

def extract_definitions(source_path: Path, names: list[str]) -> str:
    """
    Extract function/class definitions matching *names*.
    If *names* is empty, return empty string (v2: no dump of first 150 lines).
    """
    if not names:
        return ""

    try:
        source = source_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError:
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
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name in name_set:
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


# --------------------------------------------------------------------------
# Main: build context for a single pair (v2 function-scoped)
# --------------------------------------------------------------------------

def build_context_for_pair(
    prefix: str,
    metadata: dict,
    repo_dir: Path,
    file_index: dict[str, Path],
    test_file_cache: dict[str, str],
    source_roots: list[Path] | None = None,
) -> dict:
    """Build function-scoped oracle context for a single QnA pair."""
    test_file_rel = metadata.get("file", "")
    lineno = metadata.get("lineno", 0)
    if isinstance(lineno, str):
        try:
            lineno = int(lineno)
        except ValueError:
            lineno = 0

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

    # Step 1: find enclosing function
    func_node = None
    tree = None
    if full_test_source and lineno > 0:
        try:
            tree = ast.parse(full_test_source)
            func_node = find_enclosing_function(full_test_source, lineno, tree=tree)
        except SyntaxError:
            tree = None

    # Step 2: collect scoped imports
    if tree is not None:
        imports = collect_scoped_imports(tree, func_node)
    elif full_test_source:
        imports = _parse_imports_fallback(full_test_source)
    else:
        imports = _parse_imports_fallback(prefix)

    all_imported_names: set[str] = set()
    for imp in imports:
        all_imported_names.update(imp["names"])

    # Step 3: find used names in function scope only
    used_names = find_used_names_in_scope(func_node, prefix, all_imported_names)

    # Step 4 + 5: resolve and extract (no fallback)
    resolved_files: list[str] = []
    extracted_parts: list[str] = []
    seen_files: set[Path] = set()

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

            if imp_names:
                relevant_names = [n for n in imp_names if n in used_names]
            else:
                relevant_names = []

            if not relevant_names:
                continue

            code = extract_definitions(fpath, relevant_names)
            if code.strip():
                header = f"# Source: {rel}"
                extracted_parts.append(f"{header}\n{code}")

    extracted_code = "\n\n\n".join(extracted_parts)

    return {
        "resolved_imports": resolved_files,
        "used_names": sorted(used_names),
        "enclosing_function": getattr(func_node, "name", None),
        "extracted_code": extracted_code,
        "n_imports_parsed": len(imports),
        "n_files_resolved": len(resolved_files),
        "n_chars_extracted": len(extracted_code),
    }


# --------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------

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
        "ORACLE_CONTEXT_CACHE_V2",
    )

    ap = argparse.ArgumentParser(description="Function-scoped oracle context builder (v2)")
    ap.add_argument("--repos-root", type=str, default=default_repos)
    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--cache-dir", type=str, default=default_cache)
    ap.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = ap.parse_args()

    repos_root = Path(args.repos_root).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    total_pairs = sum(len(v) for v in repo_pairs.values())
    print(f"Found {len(repo_pairs)} repos, {total_pairs} total pairs")

    built = 0
    skipped = 0
    total_processed = 0
    pairs_with_context = 0
    pairs_with_function_scope = 0
    total_chars = 0

    for repo_name in tqdm(sorted(repo_pairs.keys()), desc="Building v2 oracle"):
        safe_name = repo_name.replace("/", "__")
        cache_path = cache_dir / f"{safe_name}.json"

        if cache_path.exists() and not args.force:
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
        for pair in repo_pairs[repo_name]:
            ctx = build_context_for_pair(
                pair["prefix"], pair["metadata"], repo_dir, file_index, test_file_cache,
                source_roots=source_roots,
            )
            contexts[pair["key"]] = ctx
            total_processed += 1
            if ctx["n_chars_extracted"] > 0:
                pairs_with_context += 1
                total_chars += ctx["n_chars_extracted"]
            if ctx.get("enclosing_function"):
                pairs_with_function_scope += 1

        cache_data = {
            "repo": repo_name,
            "n_pairs": len(contexts),
            "version": "v2_function_scoped",
            "contexts": contexts,
        }
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
        built += 1

    avg_chars = total_chars / max(1, pairs_with_context)
    print(f"\nDone. Built: {built}, Skipped (cached): {skipped}")
    print(f"Total pairs processed: {total_processed}")
    print(f"Pairs with context: {pairs_with_context} ({100 * pairs_with_context / max(1, total_processed):.1f}%)")
    print(f"Pairs with function scope: {pairs_with_function_scope} ({100 * pairs_with_function_scope / max(1, total_processed):.1f}%)")
    print(f"Avg context size: {avg_chars:.0f} chars")
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
