#!/usr/bin/env python3
"""Re-extract train QnAs at each repo's anchor commit (static-commit pipeline).

For every train repo in the static-commit manifest (role==train_snapshot),
this script:

  1. Reads each .py file as it exists at the anchor commit via ``git show``
     (no checkout, no worktree, no on-disk perturbation).
  2. Splits files into "test" vs "regular" using the SAME heuristics as
     ``create_dataset/2_separate_tests.py``::
         test_file := contains pytest/unittest/test_* OR has 'test' in path.
  3. Runs the SAME v2 QnA extractor used for the original Code2LoRA training
     data (``create_qnas_v2.extract_from_file``) against every test file's
     anchor-snapshot contents.
  4. Selects a balanced subset via ``select_balanced_pairs`` -- using v2's
     own caps (max-per-repo / max-per-function / max-per-file). These caps
     are higher than the smart-cap caps used for GRU training, so each repo
     gets ~10-200 high-quality assertions even when many tests exist.
  5. Emits a fresh splits-dir entry::
         <out_dir>/train.json   {"repositories": {repo_id: {embedding, qna_pairs, metadata}}}
     that is drop-in compatible with ``hypernetwork/hypernetwork_sampled.py``.

The eval suites (cr_val.json, cr_test.json, ir_test.json, ood_test.json)
are NOT rebuilt here -- they should continue to use the canonical parquet
QnAs (so GRU and static eval against the exact same QnA set per commit).
Run ``build_static_commit_train_jsonl.py`` for those splits.

Why re-extract instead of reusing parquet QnAs at the anchor commit?
  - Parquet QnAs are tied to the commit where each assertion was first
    INTRODUCED, so most assertions are absent at any given anchor commit.
  - Re-extracting at the anchor commit guarantees: (a) more samples (we
    capture every assertion alive at that snapshot, not just newly-added
    ones); (b) zero stale assertions (anything since deleted is gone).

Usage::

    python create_dataset/build_train_qnas_at_anchor.py \\
        --manifest /scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv \\
        --snapshot-embeddings /scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json \\
        --repos-roots /scratch/lhotsko/REPO_DATASET/repositories /scratch/lhotsko/REPO_DATASET/repositories_ood \\
        --out-dir /scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse the v2 extractor and test-file heuristics verbatim.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from create_qnas_v2 import extract_from_file, select_balanced_pairs  # noqa: E402
from importlib import import_module                                  # noqa: E402

# `2_separate_tests` starts with a digit so it can't be imported normally.
_sep_tests = import_module("2_separate_tests")
contains_test_content = _sep_tests.contains_test_content
is_test_by_path = _sep_tests.is_test_by_path
SKIP_DIRS = _sep_tests.SKIP_DIRS


def _resolve_repo_path(repo_id: str, roots: List[Path]) -> Path | None:
    for root in roots:
        p = root / repo_id
        if p.is_dir() and (p / ".git").exists():
            return p
    return None


def _git(repo_path: Path, *args: str, binary: bool = False
         ) -> bytes | str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(repo_path),
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        return b"" if binary else ""
    return res.stdout if binary else res.stdout.decode("utf-8", errors="ignore")


def _list_py_files_at(repo_path: Path, commit_sha: str) -> List[str]:
    out = _git(repo_path, "ls-tree", "-r", "--name-only", commit_sha)
    files = []
    for ln in out.splitlines():
        if not ln.endswith(".py"):
            continue
        parts = Path(ln).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        files.append(ln)
    return files


def _read_file_at(repo_path: Path, commit_sha: str, rel_path: str) -> str:
    raw = _git(repo_path, "show", f"{commit_sha}:{rel_path}", binary=True)
    if not raw:
        return ""
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _detect_test_files_at(repo_path: Path, commit_sha: str
                          ) -> List[Tuple[str, str]]:
    """Returns [(rel_path, contents), ...] for test files at the commit."""
    out: List[Tuple[str, str]] = []
    for rel in _list_py_files_at(repo_path, commit_sha):
        # Try cheap path-based check first; fall back to AST-based content check.
        if is_test_by_path(rel):
            txt = _read_file_at(repo_path, commit_sha, rel)
            if txt:
                out.append((rel, txt))
            continue
        txt = _read_file_at(repo_path, commit_sha, rel)
        if not txt:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if contains_test_content(txt):
                out.append((rel, txt))
    return out


def _extract_repo_qnas_at_anchor(
    repo_id: str,
    repo_path: Path,
    anchor_sha: str,
    max_per_repo: int,
    max_per_function: int,
    max_per_file: int,
) -> Tuple[List[dict], Dict[str, int]]:
    test_files = _detect_test_files_at(repo_path, anchor_sha)
    all_pairs: List[dict] = []
    n_files_tried = 0
    for rel, source in test_files:
        n_files_tried += 1
        try:
            pairs = extract_from_file(source, rel)
        except Exception:
            continue
        for p in pairs:
            md = p.setdefault("metadata", {})
            md["repo"] = repo_id
            md["test_file"] = rel
        all_pairs.extend(pairs)
    if not all_pairs:
        return [], {"n_test_files": n_files_tried, "n_extracted": 0, "n_selected": 0}
    selected = select_balanced_pairs(
        all_pairs,
        max_per_repo=max_per_repo,
        max_per_function=max_per_function,
        max_per_file=max_per_file,
    )
    return selected, {
        "n_test_files": n_files_tried,
        "n_extracted": len(all_pairs),
        "n_selected": len(selected),
    }


def _read_train_anchors(manifest_path: Path) -> Dict[str, Tuple[str, int]]:
    train: Dict[str, Tuple[str, int]] = {}
    with manifest_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["role"] == "train_snapshot":
                train[row["repo_id"]] = (
                    row["commit_sha"], int(row["commit_index"]))
    return train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--snapshot-embeddings",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json")
    ap.add_argument("--repos-roots", nargs="+", default=[
        "/scratch/lhotsko/REPO_DATASET/repositories",
        "/scratch/lhotsko/REPO_DATASET/repositories_ood",
    ])
    ap.add_argument("--out-dir",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor")
    ap.add_argument("--max-per-repo", type=int, default=200,
                    help="v2 cap; same as create_qnas_v2 default.")
    ap.add_argument("--max-per-function", type=int, default=5)
    ap.add_argument("--max-per-file", type=int, default=20)
    ap.add_argument("--limit-repos", type=int, default=0,
                    help="0 = all; >0 for smoke-testing.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in args.repos_roots]

    train_anchors = _read_train_anchors(Path(args.manifest))
    repo_ids = sorted(train_anchors.keys())
    if args.shard_total > 1:
        repo_ids = [r for i, r in enumerate(repo_ids)
                    if i % args.shard_total == args.shard_index]
    if args.limit_repos > 0:
        repo_ids = repo_ids[: args.limit_repos]
    print(f"Train repos to process: {len(repo_ids)} "
          f"(shard {args.shard_index+1}/{args.shard_total})", flush=True)

    print("Loading snapshot embeddings ...", flush=True)
    snap_emb = json.loads(Path(args.snapshot_embeddings).read_text(encoding="utf-8"))
    print(f"  {len(snap_emb):,} embeddings", flush=True)

    repositories: Dict[str, dict] = {}
    agg = Counter()
    for i, repo_id in enumerate(repo_ids, 1):
        anc_sha, anc_idx = train_anchors[repo_id]
        repo_path = _resolve_repo_path(repo_id, roots)
        if repo_path is None:
            agg["no_clone"] += 1
            continue
        emb = snap_emb.get(f"{repo_id}@{anc_sha}")
        if emb is None:
            agg["no_emb"] += 1
            continue
        pairs, stats = _extract_repo_qnas_at_anchor(
            repo_id, repo_path, anc_sha,
            max_per_repo=args.max_per_repo,
            max_per_function=args.max_per_function,
            max_per_file=args.max_per_file,
        )
        agg["test_files_total"] += stats["n_test_files"]
        agg["extracted_total"] += stats["n_extracted"]
        agg["selected_total"] += stats["n_selected"]
        if not pairs:
            agg["no_qna"] += 1
            continue
        repositories[repo_id] = {
            "embedding": emb,
            "qna_pairs": pairs,
            "metadata": {
                "repo_id": repo_id,
                "commit_sha": anc_sha,
                "commit_index": anc_idx,
                "anchor_sha": anc_sha,
                "anchor_index": anc_idx,
                "n_commits_after_anchor": 0,
                "in_repo_split": "train",
                "role": "train_snapshot",
                "qna_source": "v2_extractor_at_anchor",
                "n_test_files": stats["n_test_files"],
                "n_extracted_before_select": stats["n_extracted"],
            },
        }
        agg["kept"] += 1
        if i % 25 == 0 or i == len(repo_ids):
            print(f"  [{i}/{len(repo_ids)}] kept={agg['kept']} "
                  f"extracted={agg['extracted_total']} "
                  f"selected={agg['selected_total']}", flush=True)

    # Sharded output gets a suffix.
    fname = "train.json" if args.shard_total == 1 else f"train.shard{args.shard_index}.json"
    out_path = out_dir / fname
    out_path.write_text(
        json.dumps({"repositories": repositories}),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}", flush=True)

    readme_path = out_dir / f"README.shard{args.shard_index}.json" \
        if args.shard_total > 1 else out_dir / "README.json"
    readme = {
        "manifest": args.manifest,
        "snapshot_embeddings": args.snapshot_embeddings,
        "repos_roots": [str(p) for p in roots],
        "caps": {
            "max_per_repo": args.max_per_repo,
            "max_per_function": args.max_per_function,
            "max_per_file": args.max_per_file,
        },
        "stats": dict(agg),
        "shard": [args.shard_index, args.shard_total],
    }
    readme_path.write_text(json.dumps(readme, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    for k, v in readme["stats"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
