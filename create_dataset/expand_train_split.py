#!/usr/bin/env python3
"""
Expand training split by adding repos excluded by min-qnas filter.
Keeps existing cr_val and cr_test repos EXACTLY the same.
All new repos go into the training pool.

Usage:
    python create_dataset/expand_train_split.py --min-qnas 1
    python create_dataset/expand_train_split.py --min-qnas 1 --dry-run
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_METADATA = "REPO_METADATA.json"
QNA_HYPERNET = "QNA_HYPERNET.json"
TEST_HYPERNET = "TEST_HYPERNET"


def load_all_repos(repos_root: Path, min_qnas: int = 1):
    """Load all repos that pass basic filters."""
    by_repo = {}
    for author_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
        for repo_dir in sorted(p for p in author_dir.iterdir() if p.is_dir()):
            if repo_dir.name == TEST_HYPERNET:
                continue
            test_dir = repo_dir / TEST_HYPERNET
            qna_path = repo_dir / QNA_HYPERNET
            if not test_dir.exists() or not test_dir.is_dir():
                continue
            if not any(test_dir.rglob("*")):
                continue
            if not qna_path.exists():
                continue

            try:
                qna = json.loads(qna_path.read_text(encoding="utf-8"))
                pairs = qna.get("pairs", [])
            except (json.JSONDecodeError, OSError):
                continue
            if len(pairs) < min_qnas:
                continue

            meta_path = repo_dir / REPO_METADATA
            embedding = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    emb = meta.get("embedding")
                    if emb and isinstance(emb, list):
                        embedding = emb
                except (json.JSONDecodeError, OSError):
                    pass
            if embedding is None:
                continue

            file_embeddings = None
            if meta_path.exists():
                try:
                    meta_full = json.loads(meta_path.read_text(encoding="utf-8"))
                    fe = meta_full.get("file_embeddings")
                    if fe and isinstance(fe, list):
                        file_embeddings = fe
                except (json.JSONDecodeError, OSError):
                    pass

            name = f"{author_dir.name}/{repo_dir.name}"
            by_repo[name] = {
                "pairs": pairs,
                "embedding": embedding,
                "file_embeddings": file_embeddings,
            }
    return by_repo


def split_pairs(pairs, train_ratio=0.8, val_ratio=0.1, seed=3407):
    p = pairs.copy()
    random.seed(seed)
    random.shuffle(p)
    n = len(p)
    n_train = max(0, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    return p[:n_train], p[n_train:n_train + n_val], p[n_train + n_val:]


def main():
    ap = argparse.ArgumentParser()
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--repos-root", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Output dir for expanded splits (default: SPLITS_DIR/expanded)")
    ap.add_argument("--min-qnas", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repos_root = Path(args.repos_root or os.path.join(default_root, "repositories"))
    splits_dir = Path(args.splits_dir or default_root)
    out_dir = Path(args.out_dir or os.path.join(default_root, "expanded"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing val/test repos (these stay FIXED)
    cr_val = json.loads((splits_dir / "cr_val.json").read_text())
    cr_test = json.loads((splits_dir / "cr_test.json").read_text())
    val_repos = set(cr_val["repositories"].keys())
    test_repos = set(cr_test["repositories"].keys())
    print(f"Existing cr_val: {len(val_repos)} repos")
    print(f"Existing cr_test: {len(test_repos)} repos")

    # Load ALL available repos with min_qnas threshold
    by_repo = load_all_repos(repos_root, min_qnas=args.min_qnas)
    print(f"Total repos passing filters (min_qnas={args.min_qnas}): {len(by_repo)}")

    # Everything not in val/test goes to training
    train_repo_names = sorted(r for r in by_repo if r not in val_repos and r not in test_repos)
    print(f"Training repos: {len(train_repo_names)} (was 409, +{len(train_repo_names) - 409} new)")
    total_train_pairs = sum(len(by_repo[r]["pairs"]) for r in train_repo_names)
    print(f"Total training pairs: {total_train_pairs}")

    if args.dry_run:
        print("\n[DRY RUN] Would write expanded splits. Exiting.")
        return

    def _repo_entry(repo, pairs):
        entry = {"qna_pairs": pairs, "embedding": by_repo[repo]["embedding"]}
        if by_repo[repo].get("file_embeddings") is not None:
            entry["file_embeddings"] = by_repo[repo]["file_embeddings"]
        return entry

    # Build training split (with in-repo 80/10/10)
    train_data = {"repositories": {}}
    ir_val_data = {"repositories": {}}
    ir_test_data = {"repositories": {}}

    for repo in train_repo_names:
        pairs = by_repo[repo]["pairs"]
        tr, iv, it = split_pairs(pairs, seed=args.seed)
        if tr:
            train_data["repositories"][repo] = _repo_entry(repo, tr)
        if iv:
            ir_val_data["repositories"][repo] = _repo_entry(repo, iv)
        if it:
            ir_test_data["repositories"][repo] = _repo_entry(repo, it)

    # Write expanded training files
    for name, data in [("train.json", train_data), ("ir_val.json", ir_val_data), ("ir_test.json", ir_test_data)]:
        path = out_dir / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        n_repos = len(data["repositories"])
        n_pairs = sum(len(r["qna_pairs"]) for r in data["repositories"].values())
        print(f"  Wrote {path} ({n_repos} repos, {n_pairs} pairs)")

    # Copy val/test unchanged
    for name in ["cr_val.json", "cr_test.json"]:
        src = splits_dir / name
        dst = out_dir / name
        dst.write_text(src.read_text())
        print(f"  Copied {name} unchanged")

    # Also copy structured versions if they exist
    for name in ["cr_val_structured.json", "cr_test_structured.json"]:
        src = splits_dir / name
        if src.exists():
            dst = out_dir / name
            dst.write_text(src.read_text())
            print(f"  Copied {name}")

    # Summary
    all_train_pairs = sum(len(r["qna_pairs"]) for r in train_data["repositories"].values())
    print(f"\n[Summary]")
    print(f"  Training repos: {len(train_data['repositories'])} (was 409)")
    print(f"  Training pairs: {all_train_pairs} (was 39,612)")
    print(f"  Val repos: {len(val_repos)} (unchanged)")
    print(f"  Test repos: {len(test_repos)} (unchanged)")


if __name__ == "__main__":
    main()
