#!/usr/bin/env python3
"""Per-repo LoRA OOD split that matches the Table 3 eval qnas exactly.

The standard Table 3 OOD eval (run_baselines_v2.py / run_code2lora_gru_v2_eval.py
with --suite ood_test --qnas-per-commit-limit 8) picks the **first 8 qnas per
(repo, commit_sha) group** from $SCRATCH/REPO_DATASET/commit_parquet_ood/
qna_pairs.parquet, preserving parquet row order. That yields the canonical
14,813-qna subset every other method (sLoRA / FFT / Text2LoRA / C2L-direct /
C2L-GRU) is scored on.

This script reproduces that selection exactly and emits two JSON splits in
the per_repo_lora-compatible layout:

  <out-dir>/
    ir_test.json    # the 14,813 eval qnas, identical to what sLoRA scored
    train.json      # the remaining 166,687 qnas (capped per repo), same commits
    manifest.json   # per-repo bookkeeping

The training partition uses qnas from the **same commits** as eval but
**different qnas** (positions 9, 10, ... in each (repo, commit) group), so
per-repo LoRA learns per-repo patterns without ever seeing the 14,813 test
qnas at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    default_parquet = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
        "commit_parquet_ood",
        "qna_pairs.parquet",
    )
    default_output = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET_GRU_OOD_FULL",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=str, default=default_parquet)
    ap.add_argument("--output", type=str, default=default_output)
    ap.add_argument("--qnas-per-commit-limit", type=int, default=8,
                    help="Cap matching the V2 OOD eval (default: 8)")
    ap.add_argument("--max-train-per-repo", type=int, default=1500,
                    help="Cap on training qnas per repo (default 1500, "
                         "random subsample with --seed if exceeded)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parquet_path = Path(args.parquet).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {parquet_path}")
    table = pq.read_table(str(parquet_path))
    n_total = table.num_rows
    print(f"  {n_total:,} qnas")

    cols_needed = ["repo_id", "commit_sha", "commit_index", "in_repo_split",
                   "test_file", "test_function", "lineno", "col_offset",
                   "assertion_type", "prefix", "target"]
    arrs = {c: table.column(c).to_pylist() for c in cols_needed
            if c in table.column_names}

    repo_train: dict[str, list[dict]] = defaultdict(list)
    repo_eval: dict[str, list[dict]] = defaultdict(list)
    group_idx: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(n_total):
        repo = arrs["repo_id"][i]
        sha = arrs["commit_sha"][i]
        key = (repo, sha)
        pos = group_idx[key]
        group_idx[key] = pos + 1
        pair = {
            "prefix": arrs["prefix"][i] or "",
            "target": arrs["target"][i] or "",
            "commit_idx": int(arrs.get("commit_index", [-1] * n_total)[i] or -1),
            "metadata": {
                "commit_sha": sha,
                "in_repo_split": arrs.get("in_repo_split", [""] * n_total)[i],
                "test_file": arrs.get("test_file", [""] * n_total)[i],
                "test_function": arrs.get("test_function", [""] * n_total)[i],
                "lineno": int(arrs.get("lineno", [0] * n_total)[i] or 0),
                "col_offset": int(arrs.get("col_offset", [0] * n_total)[i] or 0),
                "assertion_type": arrs.get("assertion_type", [""] * n_total)[i],
                "group_pos": pos,
                "repo": repo,
            },
        }
        if pos < args.qnas_per_commit_limit:
            repo_eval[repo].append(pair)
        else:
            repo_train[repo].append(pair)

    rng = random.Random(args.seed)

    train_root: dict = {"repositories": {}}
    test_root: dict = {"split": "ood_test_full", "repositories": {}}
    manifest: dict = {}
    skipped: list[str] = []

    for repo in sorted(set(repo_train) | set(repo_eval)):
        train_pairs = repo_train.get(repo, [])
        eval_pairs = repo_eval.get(repo, [])
        if not train_pairs:
            skipped.append(repo)
            continue
        if not eval_pairs:
            skipped.append(repo)
            continue

        n_full_train = len(train_pairs)
        capped = False
        if args.max_train_per_repo and n_full_train > args.max_train_per_repo:
            local_rng = random.Random(f"{args.seed}:{repo}")
            train_pairs = local_rng.sample(train_pairs, args.max_train_per_repo)
            capped = True

        train_root["repositories"][repo] = {"qna_pairs": train_pairs}
        test_root["repositories"][repo] = {"qna_pairs": eval_pairs}
        manifest[repo] = {
            "n_train_pairs_available": n_full_train,
            "n_train_pairs_kept": len(train_pairs),
            "n_eval_pairs": len(eval_pairs),
            "train_capped": capped,
        }

    train_path = output_dir / "train.json"
    test_path = output_dir / "ir_test.json"
    manifest_path = output_dir / "manifest.json"

    n_train_total = sum(m["n_train_pairs_kept"] for m in manifest.values())
    n_eval_total = sum(m["n_eval_pairs"] for m in manifest.values())
    n_full = sum(m["n_train_pairs_available"] for m in manifest.values())

    print(f"  kept   : {len(manifest)} repos  ({n_train_total:,} train qnas, "
          f"{n_eval_total:,} eval qnas)")
    if skipped:
        print(f"  skipped: {len(skipped)} repos (no train and/or eval qnas)")
        print(f"           {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    if any(m["train_capped"] for m in manifest.values()):
        n_cap = sum(1 for m in manifest.values() if m["train_capped"])
        print(f"  capped : {n_cap} repos hit max_train_per_repo={args.max_train_per_repo}; "
              f"dropped {n_full - n_train_total:,} surplus qnas")

    print(f"Writing {train_path}")
    train_path.write_text(json.dumps(train_root), encoding="utf-8")
    print(f"Writing {test_path}")
    test_path.write_text(json.dumps(test_root), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "parquet": str(parquet_path),
                "qnas_per_commit_limit": args.qnas_per_commit_limit,
                "max_train_per_repo": args.max_train_per_repo,
                "seed": args.seed,
                "n_repos_kept": len(manifest),
                "n_repos_skipped": len(skipped),
                "skipped": skipped,
                "n_train_pairs_total": n_train_total,
                "n_eval_pairs_total": n_eval_total,
                "n_train_pairs_available_total": n_full,
                "per_repo": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  -> {manifest_path}")


if __name__ == "__main__":
    main()
