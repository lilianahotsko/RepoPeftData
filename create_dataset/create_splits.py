#!/usr/bin/env python3
"""
Create dataset splits for hypernetwork training.

Outputs in REPO_DATASET/:
  train.json    - 80% of QNAs from 80% of repos (in-repo train)
  ir_val.json   - 10% of QNAs from train repos (in-repo validation)
  ir_test.json  - 10% of QNAs from train repos (in-repo test)
  cr_val.json   - all QNAs from 10% of repos (cross-repo validation)
  cr_test.json  - all QNAs from 10% of repos (cross-repo test)

Structure:
  {
    "repositories": {
      "owner/repo": { "qna_pairs": [...], "embedding": [...] },
      ...
    }
  }
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_METADATA = "REPO_METADATA.json"
QNA_HYPERNET = "QNA_HYPERNET.json"
TEST_HYPERNET = "TEST_HYPERNET"


def iter_repos(repos_root: Path):
    """Yield (author, repo_name, repo_path) for repos with TEST_HYPERNET and QNA_HYPERNET."""
    for author_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
        author = author_dir.name
        for repo_dir in sorted(p for p in author_dir.iterdir() if p.is_dir()):
            repo_name = repo_dir.name
            if repo_name == TEST_HYPERNET:
                continue
            test_dir = repo_dir / TEST_HYPERNET
            qna_path = repo_dir / QNA_HYPERNET
            if not test_dir.exists() or not test_dir.is_dir():
                continue
            if not any(test_dir.rglob("*")):
                continue
            if not qna_path.exists():
                continue
            yield author, repo_name, repo_dir


def load_repo_data(repos_root: Path, min_qnas: int = 1) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Load all repos with TEST_HYPERNET and embedding. Returns (by_repo, skipped_no_embedding)."""
    by_repo = {}
    skipped_no_embedding = 0
    for author, repo_name, repo_path in iter_repos(repos_root):
        repo_full = f"{author}/{repo_name}"
        qna_path = repo_path / QNA_HYPERNET
        meta_path = repo_path / REPO_METADATA
        try:
            qna = json.loads(qna_path.read_text(encoding="utf-8"))
            pairs = qna.get("pairs", [])
        except (json.JSONDecodeError, OSError):
            continue
        if len(pairs) < min_qnas:
            continue
        size_bytes = 0
        embedding = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                size_bytes = meta.get("repo_size_bytes", 0)
                emb = meta.get("embedding")
                if emb and isinstance(emb, list):
                    embedding = emb
            except (json.JSONDecodeError, OSError):
                pass
        if embedding is None:
            skipped_no_embedding += 1
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

        by_repo[repo_full] = {
            "pairs": pairs,
            "size_bytes": size_bytes,
            "n_qnas": len(pairs),
            "embedding": embedding,
            "file_embeddings": file_embeddings,
        }
    return by_repo, skipped_no_embedding


def get_distribution_stats(by_repo: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute distribution stats for stratification."""
    n_qnas = [r["n_qnas"] for r in by_repo.values()]
    sizes = [r["size_bytes"] for r in by_repo.values()]
    return {
        "n_repos": len(by_repo),
        "total_qnas": sum(n_qnas),
        "n_qnas": {"min": min(n_qnas), "max": max(n_qnas), "mean": sum(n_qnas) / len(n_qnas)},
        "size_bytes": {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)},
    }


def stratify_repos(
    by_repo: Dict[str, Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[List[str], List[str], List[str]]:
    """Split repos 80-10-10, stratified by n_qnas (bins)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    repo_names = list(by_repo.keys())
    random.shuffle(repo_names)

    # Stratify by n_qnas: small (<p33), medium (p33-p66), large (>p66)
    n_qnas = [by_repo[r]["n_qnas"] for r in repo_names]
    sorted_n = sorted(n_qnas)
    n = len(sorted_n)
    p33 = sorted_n[n // 3] if n >= 3 else sorted_n[0]
    p66 = sorted_n[2 * n // 3] if n >= 3 else sorted_n[-1]

    def bin_repo(r):
        q = by_repo[r]["n_qnas"]
        if q <= p33:
            return 0
        if q <= p66:
            return 1
        return 2

    by_bin = defaultdict(list)
    for r in repo_names:
        by_bin[bin_repo(r)].append(r)

    train_repos = []
    val_repos = []
    test_repos = []
    for bin_idx in sorted(by_bin.keys()):
        repos = by_bin[bin_idx]
        random.shuffle(repos)
        n = len(repos)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        train_repos.extend(repos[:n_train])
        val_repos.extend(repos[n_train : n_train + n_val])
        test_repos.extend(repos[n_train + n_val :])

    return train_repos, val_repos, test_repos


def split_pairs(pairs: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 3407,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split pairs 80-10-10 for in-repo split."""
    p = pairs.copy()
    random.seed(seed)
    random.shuffle(p)
    n = len(p)
    n_train = max(0, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    return p[:n_train], p[n_train : n_train + n_val], p[n_train + n_val :]


def main():
    ap = argparse.ArgumentParser(description="Create dataset splits for hypernetwork training")
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--repos-root", type=str, default=None,
                    help=f"Root of repositories (default: {default_root}/repositories)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help=f"Output directory (default: {default_root})")
    ap.add_argument("--min-qnas", type=int, default=30,
                    help="Min QNAs per repo to include")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    repos_root = Path(args.repos_root or os.path.join(default_root, "repositories")).expanduser().resolve()
    out_dir = Path(args.out_dir or default_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading repos from {repos_root} (min_qnas={args.min_qnas})...")
    by_repo, skipped = load_repo_data(repos_root, min_qnas=args.min_qnas)
    print(f"Loaded {len(by_repo)} repos" + (f" (skipped {skipped} without embedding)" if skipped else ""))

    stats = get_distribution_stats(by_repo)
    print("\n[Distribution]")
    print(f"  total_repos: {stats['n_repos']}")
    print(f"  total_qnas: {stats['total_qnas']}")
    print(f"  n_qnas: min={stats['n_qnas']['min']} max={stats['n_qnas']['max']} mean={stats['n_qnas']['mean']:.1f}")
    print(f"  size_bytes: min={stats['size_bytes']['min']} max={stats['size_bytes']['max']} mean={stats['size_bytes']['mean']:.0f}")

    train_repos, val_repos, test_repos = stratify_repos(
        by_repo,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"\n[Repo split] train={len(train_repos)} val={len(val_repos)} test={len(test_repos)}")

    # In-repo split: for train repos, split each repo's QNAs 80-10-10
    train_data = {"repositories": {}}
    ir_val_data = {"repositories": {}}
    ir_test_data = {"repositories": {}}
    def _repo_entry(repo, pairs):
        entry = {"qna_pairs": pairs, "embedding": by_repo[repo]["embedding"]}
        if by_repo[repo].get("file_embeddings") is not None:
            entry["file_embeddings"] = by_repo[repo]["file_embeddings"]
        return entry

    for repo in train_repos:
        pairs = by_repo[repo]["pairs"]
        tr, iv, it = split_pairs(pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
        if tr:
            train_data["repositories"][repo] = _repo_entry(repo, tr)
        if iv:
            ir_val_data["repositories"][repo] = _repo_entry(repo, iv)
        if it:
            ir_test_data["repositories"][repo] = _repo_entry(repo, it)

    # Cross-repo: val/test repos -> cr_val, cr_test
    cr_val_data = {"repositories": {}}
    cr_test_data = {"repositories": {}}
    for repo in val_repos:
        cr_val_data["repositories"][repo] = _repo_entry(repo, by_repo[repo]["pairs"])
    for repo in test_repos:
        cr_test_data["repositories"][repo] = _repo_entry(repo, by_repo[repo]["pairs"])

    out_files = [
        ("train.json", train_data),
        ("ir_val.json", ir_val_data),
        ("ir_test.json", ir_test_data),
        ("cr_val.json", cr_val_data),
        ("cr_test.json", cr_test_data),
    ]
    for name, data in out_files:
        path = out_dir / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        n_repos = len(data["repositories"])
        n_pairs = sum(len(r["qna_pairs"]) for r in data["repositories"].values())
        print(f"  Wrote {path} ({n_repos} repos, {n_pairs} pairs)")

    # Statistics summary
    def split_stats(split_name: str, data: Dict) -> None:
        repos = data["repositories"]
        n_repos = len(repos)
        n_qnas = sum(len(r["qna_pairs"]) for r in repos.values())
        if n_repos == 0:
            print(f"  {split_name}: 0 repos, 0 QNAs")
            return
        qnas_per_repo = [len(r["qna_pairs"]) for r in repos.values()]
        print(f"  {split_name}: {n_repos} repos, {n_qnas} QNAs  "
              f"(min={min(qnas_per_repo)} max={max(qnas_per_repo)} mean={sum(qnas_per_repo)/n_repos:.1f} per repo)")

    print("\n[Split statistics]")
    split_stats("train.json", train_data)
    split_stats("ir_val.json", ir_val_data)
    split_stats("ir_test.json", ir_test_data)
    split_stats("cr_val.json", cr_val_data)
    split_stats("cr_test.json", cr_test_data)
    total_qnas = (
        sum(len(r["qna_pairs"]) for r in train_data["repositories"].values())
        + sum(len(r["qna_pairs"]) for r in ir_val_data["repositories"].values())
        + sum(len(r["qna_pairs"]) for r in ir_test_data["repositories"].values())
        + sum(len(r["qna_pairs"]) for r in cr_val_data["repositories"].values())
        + sum(len(r["qna_pairs"]) for r in cr_test_data["repositories"].values())
    )
    print(f"  Total QNAs across all splits: {total_qnas}")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
