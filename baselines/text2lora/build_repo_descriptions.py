#!/usr/bin/env python3
"""
Build short task descriptions for each training repository, in the format
expected by Text2LoRA (tasks/{repo_slug}/metadata.yaml with `descriptions` list).

For each repo we generate 5 description variants (~40-60 tokens each) by combining:
  - Package description (from pyproject.toml / setup.cfg / setup.py)
  - Top-N public class names (ranked by number of methods)
  - Top-N public function names (from non-test, non-private modules)

Output: creates text2lora/tasks/{repo_slug}/metadata.yaml for every training repo.

Usage:
    python baselines/text2lora/build_repo_descriptions.py \
        --splits-dir $SCRATCH/REPO_DATASET \
        --repos-dir  $SCRATCH/REPO_DATASET/repositories \
        --text2lora-dir text2lora
"""

import ast
import json
import os
import re
import sys
import tomllib
import argparse
from pathlib import Path

import yaml


# ── Repo slug: owner__repo (safe filename) ────────────────────────────────────

def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


# ── Package description extraction ───────────────────────────────────────────

def get_pkg_description(repo_dir: Path) -> str | None:
    """Extract one-line package description from pyproject.toml / setup.cfg / setup.py."""
    # pyproject.toml
    pp = repo_dir / "pyproject.toml"
    if pp.exists():
        try:
            data = tomllib.loads(pp.read_text(errors="ignore"))
            desc = data.get("project", {}).get("description")
            if not desc:
                desc = data.get("tool", {}).get("poetry", {}).get("description")
            if desc and len(desc) > 10:
                return desc.strip()
        except Exception:
            pass

    # setup.cfg
    cfg = repo_dir / "setup.cfg"
    if cfg.exists():
        try:
            for line in cfg.read_text(errors="ignore").splitlines():
                if re.match(r"^\s*description\s*=\s*", line):
                    desc = line.split("=", 1)[1].strip()
                    if len(desc) > 10:
                        return desc
        except Exception:
            pass

    # setup.py
    spy = repo_dir / "setup.py"
    if spy.exists():
        try:
            text = spy.read_text(errors="ignore")
            m = re.search(r'description\s*=\s*["\']([^"\']{10,})["\']', text)
            if m:
                return m.group(1).strip()
        except Exception:
            pass

    return None


# ── AST-based API surface extraction ─────────────────────────────────────────

def _is_test_file(path: Path) -> bool:
    return "test" in path.stem.lower() or "test" in str(path.parent).lower()


def _is_private(name: str) -> bool:
    return name.startswith("_")


def extract_api_surface(repo_dir: Path) -> tuple[list[str], list[str]]:
    """
    Returns (class_names, function_names) sorted by prominence.
    Classes are ranked by number of methods (descending).
    Functions are from __init__.py first, then other modules.
    """
    class_info: dict[str, int] = {}  # name → method count
    func_names: list[str] = []
    init_exports: set[str] = set()

    # Collect __all__ from __init__.py files to prioritize exported symbols
    for init_file in repo_dir.rglob("__init__.py"):
        if _is_test_file(init_file):
            continue
        try:
            tree = ast.parse(init_file.read_text(errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        init_exports.add(elt.value)
        except Exception:
            pass

    # Parse all non-test .py files
    py_files = [
        f for f in repo_dir.rglob("*.py")
        if not _is_test_file(f)
        and not any(p in f.parts for p in [".git", "__pycache__", ".tox", "build", "dist"])
    ]

    seen_funcs: set[str] = set()

    for py_file in sorted(py_files, key=lambda f: (0 if f.stem == "__init__" else 1, f.name)):
        try:
            tree = ast.parse(py_file.read_text(errors="ignore"))
        except Exception:
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not _is_private(node.name):
                methods = sum(
                    1 for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not _is_private(n.name)
                )
                # Keep max method count if seen in multiple files
                class_info[node.name] = max(class_info.get(node.name, 0), methods)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_private(node.name) and node.name not in seen_funcs:
                    seen_funcs.add(node.name)
                    func_names.append(node.name)

    # Rank classes: __all__ exports first, then by method count (desc)
    sorted_classes = sorted(
        class_info.items(),
        key=lambda kv: (0 if kv[0] in init_exports else 1, -kv[1], kv[0])
    )
    # Rank functions: __all__ exports first, then original order
    sorted_funcs_exp = [f for f in func_names if f in init_exports]
    sorted_funcs_rest = [f for f in func_names if f not in init_exports]
    sorted_funcs = sorted_funcs_exp + sorted_funcs_rest

    return (
        [name for name, _ in sorted_classes[:8]],
        sorted_funcs[:10],
    )


# ── Description templates ─────────────────────────────────────────────────────

def make_descriptions(
    repo_name: str,
    pkg_desc: str | None,
    classes: list[str],
    functions: list[str],
) -> list[str]:
    """
    Generate 5 short description variants (~40-60 tokens each).
    Mirrors the style of Text2LoRA's original NLP task descriptions.
    """
    short_name = repo_name.split("/")[-1].replace("-", " ").replace("_", " ")
    cls_str  = ", ".join(classes[:5])  if classes  else ""
    fn_str   = ", ".join(functions[:6]) if functions else ""
    cls_str2 = ", ".join(classes[2:6]) if len(classes) > 2 else cls_str
    fn_str2  = ", ".join(functions[3:8]) if len(functions) > 3 else fn_str

    base = pkg_desc if pkg_desc else f"Python repository {short_name}"
    # Ensure base ends without trailing period for clean joining
    base_clean = base.rstrip(".")

    descs = []

    # Variant 1 — full (desc + classes + functions)
    parts = [base_clean]
    if cls_str:
        parts.append(f"Python repository with classes: {cls_str}")
    if fn_str:
        parts.append(f"Key functions: {fn_str}")
    descs.append(". ".join(parts) + ".")

    # Variant 2 — description + classes only
    if cls_str:
        descs.append(f"{base_clean}. Main API classes include: {cls_str}.")
    else:
        descs.append(f"{base_clean}.")

    # Variant 3 — description + functions only
    if fn_str:
        descs.append(f"{base_clean}. Provides functions such as: {fn_str}.")
    else:
        descs.append(f"Python package {short_name}. {base_clean}.")

    # Variant 4 — reversed order (functions then classes)
    parts4 = [base_clean]
    if fn_str:
        parts4.append(f"Key functions: {fn_str}")
    if cls_str:
        parts4.append(f"Main classes: {cls_str}")
    descs.append(". ".join(parts4) + ".")

    # Variant 5 — second slice of symbols (diversity)
    parts5 = [base_clean]
    if cls_str2:
        parts5.append(f"Classes include: {cls_str2}")
    if fn_str2:
        parts5.append(f"Functions include: {fn_str2}")
    descs.append(". ".join(parts5) + ".")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in descs:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    # Always return at least 1
    if not unique:
        unique = [f"{base_clean}."]

    return unique


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir",    required=True, type=Path)
    ap.add_argument("--repos-dir",     required=True, type=Path)
    ap.add_argument("--text2lora-dir", required=True, type=Path,
                    help="Path to text2lora/ repo root (tasks/ will be written here)")
    ap.add_argument("--splits",        nargs="+",
                    default=["train", "cr_val", "cr_test", "ir_val", "ir_test"])
    args = ap.parse_args()

    # Collect all repo names across splits
    all_repos: set[str] = set()
    for split in args.splits:
        split_file = args.splits_dir / f"{split}.json"
        if not split_file.exists():
            print(f"  SKIP {split}.json (not found)")
            continue
        data = json.loads(split_file.read_text())
        all_repos.update(data.get("repositories", {}).keys())

    print(f"Total unique repos: {len(all_repos)}")

    tasks_dir = args.text2lora_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    stats = {"found_desc": 0, "no_desc": 0, "no_repo_dir": 0}

    for repo_name in sorted(all_repos):
        repo_dir = args.repos_dir / repo_name
        repo_slug = slug(repo_name)
        task_dir = tasks_dir / repo_slug
        task_dir.mkdir(exist_ok=True)

        if not repo_dir.exists():
            stats["no_repo_dir"] += 1
            pkg_desc = None
            classes, functions = [], []
        else:
            pkg_desc = get_pkg_description(repo_dir)
            classes, functions = extract_api_surface(repo_dir)
            if pkg_desc:
                stats["found_desc"] += 1
            else:
                stats["no_desc"] += 1

        descriptions = make_descriptions(repo_name, pkg_desc, classes, functions)

        metadata = {
            "descriptions": descriptions,
            # Store for inspection / debugging
            "_repo": repo_name,
            "_pkg_desc": pkg_desc,
            "_top_classes": classes[:8],
            "_top_functions": functions[:10],
        }

        out_file = task_dir / "metadata.yaml"
        out_file.write_text(yaml.dump(metadata, allow_unicode=True, sort_keys=False))

    print(f"\nDone.")
    print(f"  With package description: {stats['found_desc']}")
    print(f"  No package description:   {stats['no_desc']}")
    print(f"  Repo dir missing:         {stats['no_repo_dir']}")
    print(f"  Task metadata written to: {tasks_dir}/")

    # Print a few examples
    print("\n--- Sample descriptions ---")
    for repo_name in sorted(all_repos)[:5]:
        meta_file = tasks_dir / slug(repo_name) / "metadata.yaml"
        if meta_file.exists():
            m = yaml.safe_load(meta_file.read_text())
            print(f"\n[{repo_name}]")
            for d in m["descriptions"]:
                print(f"  • {d}")


if __name__ == "__main__":
    main()
