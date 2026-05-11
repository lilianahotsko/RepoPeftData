#!/usr/bin/env python3
"""Extend the static-commit manifest with ir_val and cr_val rows.

The original ``build_static_commit_manifest.py`` only enumerated:
  * train_snapshot (anchor commit, 400 rows)
  * cr_test_{train,val,test}  (full history of 51 cr_test repos)
  * ir_test                    (in_repo_split=='test' commits of 400 train repos)
  * ood_test                   (every OOD commit)

For paper-grade validation we also need:
  * ir_val   : in_repo_split=='val' commits of the 400 train repos
  * cr_val_* : full history of the 49 cr_val repos

This script appends those rows to the existing manifest IN PLACE (preserving
order; appending only new rows by (repo_id, commit_sha, role) key). It also
emits ``manifest_extras.tsv`` with just the new rows so the embedding job
can target them without re-embedding what's already cached.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import pyarrow.parquet as pq
import pyarrow.compute as pc


def _last_train_commit_per_repo(commits_path: Path) -> Dict[str, Tuple[str, int]]:
    t = pq.read_table(commits_path,
                      columns=["repo_id", "commit_sha", "commit_index", "in_repo_split"])
    t = t.filter(pc.equal(t.column("in_repo_split"), "train"))
    df = t.to_pandas()
    last = (df.sort_values(["repo_id", "commit_index"])
              .groupby("repo_id", as_index=False).tail(1))
    return {r["repo_id"]: (r["commit_sha"], int(r["commit_index"]))
            for _, r in last.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf")
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--extras-out",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest_extras.tsv")
    args = ap.parse_args()

    parq = Path(args.parquet_dir)
    manifest_path = Path(args.manifest)
    extras_path = Path(args.extras_out)

    print("Reading existing manifest ...")
    existing: set = set()
    with manifest_path.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            existing.add((row["repo_id"], row["commit_sha"]))
    print(f"  rows in manifest: {len(existing):,}")

    new_rows = []

    # ---- ir_val: in_repo_split=='val' commits of cross_repo=='train' repos ----
    print("\nPulling ir_val rows (train repos, in_repo_split=='val') ...")
    train_anchors = _last_train_commit_per_repo(parq / "commits" / "train.parquet")
    t = pq.read_table(parq / "commits" / "train.parquet",
                      columns=["repo_id", "commit_sha", "commit_index", "in_repo_split"])
    t = t.filter(pc.equal(t.column("in_repo_split"), "val"))
    df = t.to_pandas().drop_duplicates(["repo_id", "commit_sha"])
    n_added = 0
    for r in df.itertuples(index=False):
        if (r.repo_id, r.commit_sha) in existing:
            continue
        anc = train_anchors.get(r.repo_id)
        if anc is None:
            continue
        anc_sha, anc_idx = anc
        new_rows.append({
            "repo_id": r.repo_id,
            "commit_sha": r.commit_sha,
            "commit_index": int(r.commit_index),
            "role": "ir_val",
            "anchor_sha": anc_sha,
            "anchor_index": anc_idx,
            "n_commits_after_anchor": int(r.commit_index) - anc_idx,
        })
        n_added += 1
    print(f"  ir_val rows added: {n_added}")

    # ---- cr_val_*: full history of cr_val repos -------------------------------
    print("\nPulling cr_val_{train,val,test} rows ...")
    cr_val_anchors = _last_train_commit_per_repo(parq / "commits" / "cr_val.parquet")
    t = pq.read_table(parq / "commits" / "cr_val.parquet",
                      columns=["repo_id", "commit_sha", "commit_index", "in_repo_split"])
    df = t.to_pandas().drop_duplicates(["repo_id", "commit_sha"])
    # Also need at least one QnA at the commit (matches manifest semantics)
    qna = pq.read_table(parq / "qna" / "cr_val.parquet",
                        columns=["repo_id", "commit_sha"]).to_pandas().drop_duplicates()
    has_qna = set(zip(qna["repo_id"], qna["commit_sha"]))
    n_added_cr = 0
    for r in df.itertuples(index=False):
        if (r.repo_id, r.commit_sha) in existing:
            continue
        if (r.repo_id, r.commit_sha) not in has_qna:
            continue
        anc = cr_val_anchors.get(r.repo_id)
        if anc is None:
            anc_sha, anc_idx = "", -1
            delta = 0
        else:
            anc_sha, anc_idx = anc
            delta = int(r.commit_index) - anc_idx
        role = f"cr_val_{r.in_repo_split}"
        new_rows.append({
            "repo_id": r.repo_id,
            "commit_sha": r.commit_sha,
            "commit_index": int(r.commit_index),
            "role": role,
            "anchor_sha": anc_sha,
            "anchor_index": anc_idx,
            "n_commits_after_anchor": delta,
        })
        n_added_cr += 1
    print(f"  cr_val_* rows added: {n_added_cr}")

    print(f"\nTotal new rows: {len(new_rows):,}")
    # Append to manifest (preserve original).
    print("\nAppending to manifest ...")
    fields = ["repo_id", "commit_sha", "commit_index", "role",
              "anchor_sha", "anchor_index", "n_commits_after_anchor"]
    with manifest_path.open("a") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        for row in new_rows:
            wr.writerow(row)

    # Also write extras for embedding job.
    with extras_path.open("w") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        wr.writeheader()
        for row in new_rows:
            wr.writerow(row)
    print(f"  -> {manifest_path} (appended)")
    print(f"  -> {extras_path} (extras only, for embedding)")

    # Final stats.
    print("\n=== Updated manifest stats ===")
    from collections import Counter
    by_role = Counter()
    with manifest_path.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            by_role[row["role"]] += 1
    for k, v in sorted(by_role.items()):
        print(f"  {k}: {v:,}")
    print(f"  TOTAL: {sum(by_role.values()):,}")


if __name__ == "__main__":
    main()
