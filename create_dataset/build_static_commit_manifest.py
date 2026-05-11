#!/usr/bin/env python3
"""Build the (repo_id, commit_sha, role) manifest for the static-commit dataset.

Step 1 of the ``Code2LoRA-direct, commit-aware`` pipeline.

* For every train repo: pick the LAST commit whose ``in_repo_split == 'train'``.
  This is the single ``train_snapshot`` we will (a) embed and (b) attach training
  QnAs to. Anchor = same commit. n_commits_after_anchor = 0.
* For every test repo (cr_test, ir_test, ood_test): pick EVERY commit that has
  >= 1 QnA in the corresponding bench parquet. Anchor for cr_test repos =
  last ``in_repo_split=='train'`` commit in their cr_test parquet (= boundary
  between training cut and held-out window). Anchor for ir_test = the SAME train
  repo's last training commit (since we evaluate on in_repo_split=='test'
  commits of the train cross-repo split). Anchor for ood_test = first commit
  in the OOD parquet (OOD has no in_repo_split=='train' history).

n_commits_after_anchor = commit_index - anchor_index. Used downstream to plot
the decay curve of EM% as the static checkpoint drifts further from its
training snapshot.

Output: ``$SPLITS_DIR/static_commit/manifest.tsv``.

This script is pure metadata; no git, no embedding. Runs in <1 min CPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import pyarrow.parquet as pq
import pyarrow.compute as pc


def _last_train_commit_per_repo(commits_path: Path) -> Dict[str, Tuple[str, int]]:
    """Return {repo_id: (commit_sha, commit_index)} for the last in_repo_split=='train' commit."""
    t = pq.read_table(commits_path,
                      columns=["repo_id", "commit_sha", "commit_index", "in_repo_split"])
    t = t.filter(pc.equal(t.column("in_repo_split"), "train"))
    df = t.to_pandas()
    last = (df.sort_values(["repo_id", "commit_index"])
              .groupby("repo_id", as_index=False).tail(1))
    return {r["repo_id"]: (r["commit_sha"], int(r["commit_index"]))
            for _, r in last.iterrows()}


def _commits_with_qna(qna_path: Path,
                      commits_path: Path,
                      in_repo_split_filter: str | None = None) -> list:
    """Return [(repo_id, commit_sha, commit_index, in_repo_split)] sorted by (repo, idx)."""
    qna_cols = ["repo_id", "commit_sha", "in_repo_split"]
    qna = pq.read_table(qna_path, columns=qna_cols)
    if in_repo_split_filter is not None:
        qna = qna.filter(pc.equal(qna.column("in_repo_split"), in_repo_split_filter))
    pairs_df = (qna.select(["repo_id", "commit_sha"])
                   .to_pandas().drop_duplicates())
    commits = pq.read_table(commits_path,
                            columns=["repo_id", "commit_sha", "commit_index", "in_repo_split"])
    commits_df = commits.to_pandas()
    merged = pairs_df.merge(commits_df, on=["repo_id", "commit_sha"], how="left")
    merged = merged.sort_values(["repo_id", "commit_index"])
    return [
        (r["repo_id"], r["commit_sha"], int(r["commit_index"]), str(r["in_repo_split"]))
        for _, r in merged.iterrows()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir",
                    default=os.environ.get(
                        "PARQUET_DIR",
                        "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf"))
    ap.add_argument("--ood-parquet-dir",
                    default=os.environ.get(
                        "OOD_PARQUET_DIR",
                        "/scratch/lhotsko/REPO_DATASET/commit_parquet_ood"))
    ap.add_argument("--output-dir",
                    default=os.environ.get(
                        "STATIC_COMMIT_DIR",
                        "/scratch/lhotsko/REPO_DATASET/static_commit"))
    args = ap.parse_args()

    parq = Path(args.parquet_dir)
    ood = Path(args.ood_parquet_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_snaps = _last_train_commit_per_repo(parq / "commits" / "train.parquet")
    cr_test_anchor = _last_train_commit_per_repo(parq / "commits" / "cr_test.parquet")

    ood_commits = pq.read_table(ood / "commits.parquet",
                                columns=["repo_id", "commit_sha", "commit_index"])
    ood_df = ood_commits.to_pandas().sort_values(["repo_id", "commit_index"])
    ood_first = ood_df.groupby("repo_id", as_index=False).head(1)
    ood_anchor = {r["repo_id"]: (r["commit_sha"], int(r["commit_index"]))
                  for _, r in ood_first.iterrows()}

    anchors: Dict[str, Tuple[str, int]] = {}
    anchors.update(train_snaps)
    anchors.update(cr_test_anchor)
    anchors.update(ood_anchor)

    rows = []

    def _emit(repo_id, commit_sha, commit_index, role):
        anc = anchors.get(repo_id)
        if anc is None:
            rows.append((repo_id, commit_sha, commit_index, role, "", -1, 0))
            return
        anc_sha, anc_idx = anc
        rows.append((repo_id, commit_sha, commit_index, role,
                     anc_sha, anc_idx, commit_index - anc_idx))

    for repo_id, (sha, idx) in train_snaps.items():
        _emit(repo_id, sha, idx, "train_snapshot")

    for repo_id, sha, idx, irs in _commits_with_qna(
        parq / "qna" / "cr_test.parquet",
        parq / "commits" / "cr_test.parquet",
    ):
        _emit(repo_id, sha, idx, f"cr_test_{irs}")

    for repo_id, sha, idx, irs in _commits_with_qna(
        parq / "qna" / "train.parquet",
        parq / "commits" / "train.parquet",
        in_repo_split_filter="test",
    ):
        _emit(repo_id, sha, idx, "ir_test")

    ood_qna = pq.read_table(ood / "qna_pairs.parquet",
                            columns=["repo_id", "commit_sha"]).to_pandas().drop_duplicates()
    ood_join = ood_qna.merge(ood_df, on=["repo_id", "commit_sha"], how="left")
    ood_join = ood_join.sort_values(["repo_id", "commit_index"])
    for _, r in ood_join.iterrows():
        _emit(r["repo_id"], r["commit_sha"], int(r["commit_index"]), "ood_test")

    manifest_path = out_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("repo_id\tcommit_sha\tcommit_index\trole\tanchor_sha\tanchor_index\tn_commits_after_anchor\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    by_role: Dict[str, int] = {}
    by_role_unique_repos: Dict[str, set] = {}
    for repo_id, _, _, role, *_ in rows:
        by_role[role] = by_role.get(role, 0) + 1
        by_role_unique_repos.setdefault(role, set()).add(repo_id)

    print(f"Wrote {manifest_path} ({len(rows):,} rows)")
    print(f"\n{'role':<24} {'rows':>10} {'repos':>8}")
    print("-" * 46)
    for role in sorted(by_role):
        print(f"{role:<24} {by_role[role]:>10,} {len(by_role_unique_repos[role]):>8,}")
    n_unique_pairs = len({(r[0], r[1]) for r in rows})
    print(f"\nUnique (repo_id, commit_sha) pairs to embed: {n_unique_pairs:,}")


if __name__ == "__main__":
    main()
