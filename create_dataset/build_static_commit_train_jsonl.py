#!/usr/bin/env python3
"""Build the training-side splits dir for Code2LoRA-direct, commit-aware.

Step 3 of the static-commit pipeline. Reads:

  * ``manifest.tsv`` (step 1)
  * ``snapshot_embeddings.json`` (step 2)
  * smart-cap GRU train parquet (the 215k-row subset GRU_commit was trained on)

Writes a *splits dir* in the canonical RepoPeftBench layout that
``hypernetwork/hypernetwork_sampled.py --splits-dir`` consumes verbatim::

    <out_dir>/
      train.json    repositories[<repo_id>]                  -- one entry / train repo
      cr_val.json   repositories[<repo_id>@@<commit_sha>]    -- one entry / val (repo, commit) pair
      cr_test.json  repositories[<repo_id>@@<commit_sha>]    -- one entry / cr_test (repo, commit) pair
      STATIC_COMMIT_README.json
        provenance + counts

Each repositories[*] entry has:
    {"embedding": [2048],
     "qna_pairs": [{"prefix": str, "target": str, "assertion_type": ..., ...}],
     "metadata": {"repo_id", "commit_sha", "commit_index",
                  "anchor_sha", "anchor_index", "n_commits_after_anchor",
                  "in_repo_split", "role"}}

Train semantics
---------------
* Anchor per train repo = its last in_repo_split=='train' commit (manifest
  role=train_snapshot).
* Embedding = snapshot embedding at the anchor.
* QnAs = smart-cap GRU train rows for that repo with commit_index <= anchor_index
  AND (optionally) test_file still tracked at the anchor commit
  (``--strict-file-validity``, default ON). Deduped by
  (test_file, test_function, target).

Val / test semantics
--------------------
* One entry per (repo, commit) with QnAs in the corresponding parquet.
* Embedding = snapshot embedding at that commit.
* QnAs = the parquet's QnAs at that (repo, commit), filtered for non-empty
  / non-leading-comma targets (matches ``run_repopeft_bench.load_bench``).
* metadata.n_commits_after_anchor is the drift distance from the train cut.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import pyarrow.compute as pc


REPOS_ROOTS = (
    "/scratch/lhotsko/REPO_DATASET/repositories",
    "/scratch/lhotsko/REPO_DATASET/repositories_ood",
)


def _resolve_repo_path(repo_id: str) -> Path | None:
    for root in REPOS_ROOTS:
        p = Path(root) / repo_id
        if p.is_dir() and (p / ".git").exists():
            return p
    return None


def _list_py_files_at(repo_path: Path, commit_sha: str) -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit_sha],
            cwd=repo_path, capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return set()
    return {ln for ln in out.splitlines() if ln.endswith(".py")}


def _read_anchors(manifest_path: Path) -> Tuple[Dict[str, Tuple[str, int]],
                                                  Dict[str, Tuple[str, int]]]:
    """Returns (train_snapshots, anchor_for_every_repo).

    * train_snapshots: {repo_id: (sha, idx)} for role==train_snapshot
    * anchor_for_every_repo: {repo_id: (anchor_sha, anchor_idx)} — the anchor
      column from the manifest (already populated for all rows).
    """
    train_snaps: Dict[str, Tuple[str, int]] = {}
    all_anchors: Dict[str, Tuple[str, int]] = {}
    with manifest_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["role"] == "train_snapshot":
                train_snaps[row["repo_id"]] = (row["commit_sha"], int(row["commit_index"]))
            if row["repo_id"] not in all_anchors:
                all_anchors[row["repo_id"]] = (row["anchor_sha"], int(row["anchor_index"]))
    return train_snaps, all_anchors


def _build_train_split(train_snaps: Dict[str, Tuple[str, int]],
                       snap_emb: Dict[str, List[float]],
                       smartcap_parquet: Path,
                       strict_file_validity: bool
                       ) -> Tuple[Dict[str, dict], Dict[str, int]]:
    print(f"Loading smart-cap QnA parquet ({smartcap_parquet}) ...", flush=True)
    tbl = pq.read_table(
        smartcap_parquet,
        columns=["repo_id", "commit_sha", "commit_index", "in_repo_split",
                 "test_file", "test_function", "prefix", "target",
                 "assertion_type"],
    )
    tbl = tbl.filter(pc.equal(tbl.column("in_repo_split"), "train"))
    print(f"  {tbl.num_rows:,} smart-cap train QnAs", flush=True)
    df = tbl.to_pandas()
    by_repo: Dict[str, list] = defaultdict(list)
    for row in df.itertuples(index=False):
        by_repo[row.repo_id].append(row)

    repositories: Dict[str, dict] = {}
    stats = defaultdict(int)
    for i, (repo_id, (anc_sha, anc_idx)) in enumerate(sorted(train_snaps.items()), 1):
        stats["n_repos_seen"] += 1
        emb = snap_emb.get(f"{repo_id}@{anc_sha}")
        if emb is None:
            stats["n_repos_no_emb"] += 1
            continue
        candidates = by_repo.get(repo_id, [])
        if not candidates:
            stats["n_repos_no_qna_in_smartcap"] += 1
            continue

        # commit_index <= anc_idx (i.e. only QnAs from the training cut)
        before = [q for q in candidates if q.commit_index <= anc_idx]
        stats["n_qna_dropped_after_anchor"] += len(candidates) - len(before)

        if strict_file_validity:
            repo_path = _resolve_repo_path(repo_id)
            if repo_path is None:
                stats["n_repos_no_clone"] += 1
                continue
            files_at_anc = _list_py_files_at(repo_path, anc_sha)
            kept = [q for q in before if q.test_file in files_at_anc]
            stats["n_qna_dropped_missing_file"] += len(before) - len(kept)
            before = kept

        if not before:
            stats["n_repos_no_qna_after_filter"] += 1
            continue

        # Dedup by (test_file, test_function, target).
        seen = set()
        deduped = []
        for q in before:
            k = (q.test_file, q.test_function, q.target)
            if k in seen: continue
            seen.add(k)
            deduped.append(q)
        stats["n_qna_dropped_dedup"] += len(before) - len(deduped)

        pairs = []
        for q in deduped:
            pairs.append({
                "prefix": q.prefix,
                "target": q.target,
                "assertion_type": q.assertion_type,
                "metadata": {
                    "test_file": q.test_file,
                    "test_function": q.test_function,
                    "commit_sha": q.commit_sha,
                    "commit_index": int(q.commit_index),
                },
            })

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
            },
        }
        stats["n_repos_kept"] += 1
        stats["n_qna_kept"] += len(pairs)
        if i % 50 == 0 or i == len(train_snaps):
            print(f"  [{i}/{len(train_snaps)}] kept={stats['n_repos_kept']} "
                  f"qna={stats['n_qna_kept']:,}", flush=True)
    return repositories, dict(stats)


def _build_eval_split(qna_parquet: Path,
                      commits_parquet: Path,
                      anchors: Dict[str, Tuple[str, int]],
                      snap_emb: Dict[str, List[float]],
                      in_repo_split_filter: str | None = None,
                      ) -> Tuple[Dict[str, dict], Dict[str, int]]:
    qna = pq.read_table(qna_parquet)
    if in_repo_split_filter is not None:
        qna = qna.filter(pc.equal(qna.column("in_repo_split"), in_repo_split_filter))
    qdf = qna.to_pandas()

    cdf = pq.read_table(
        commits_parquet,
        columns=[c for c in ["repo_id", "commit_sha", "commit_index", "in_repo_split"]
                 if c in pq.ParquetFile(commits_parquet).schema.names],
    ).to_pandas().drop_duplicates(["repo_id", "commit_sha"])
    cmeta: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for _, r in cdf.iterrows():
        cmeta[(r["repo_id"], r["commit_sha"])] = (
            int(r["commit_index"]),
            str(r["in_repo_split"]) if "in_repo_split" in cdf.columns else "",
        )

    by_pair: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in qdf.itertuples(index=False):
        target = (row.target or "").lstrip()
        if not row.prefix or not target or target.startswith(","):
            continue
        by_pair[(row.repo_id, row.commit_sha)].append({
            "prefix": row.prefix,
            "target": row.target,
            "assertion_type": row.assertion_type,
            "metadata": {
                "test_file": row.test_file,
                "test_function": row.test_function,
            },
        })

    repositories: Dict[str, dict] = {}
    stats = defaultdict(int)
    for (repo_id, sha), pairs in by_pair.items():
        stats["n_pairs_seen"] += 1
        emb = snap_emb.get(f"{repo_id}@{sha}")
        if emb is None:
            stats["n_pairs_no_emb"] += 1
            continue
        ci, irs = cmeta.get((repo_id, sha), (-1, ""))
        anc_sha, anc_idx = anchors.get(repo_id, ("", -1))
        n_after = (ci - anc_idx) if (anc_idx >= 0 and ci >= 0) else 0
        repositories[f"{repo_id}@@{sha}"] = {
            "embedding": emb,
            "qna_pairs": pairs,
            "metadata": {
                "repo_id": repo_id, "commit_sha": sha,
                "commit_index": ci, "in_repo_split": irs,
                "anchor_sha": anc_sha, "anchor_index": anc_idx,
                "n_commits_after_anchor": n_after,
            },
        }
        stats["n_pairs_kept"] += 1
        stats["n_qna_kept"] += len(pairs)
    return repositories, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--snapshot-embeddings",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json")
    ap.add_argument("--smartcap-parquet",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet")
    ap.add_argument("--parquet-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf")
    ap.add_argument("--out-dir",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/splits")
    ap.add_argument("--strict-file-validity", action="store_true", default=True)
    ap.add_argument("--no-strict-file-validity",
                    dest="strict_file_validity", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_snaps, anchors = _read_anchors(Path(args.manifest))
    print(f"Train snapshots: {len(train_snaps)}; total anchors: {len(anchors)}",
          flush=True)

    print("Loading snapshot embeddings ...", flush=True)
    snap_emb = json.loads(Path(args.snapshot_embeddings).read_text(encoding="utf-8"))
    print(f"  {len(snap_emb):,} snapshot embeddings", flush=True)

    print("\n[1/3] Building train.json ...", flush=True)
    train_repos, train_stats = _build_train_split(
        train_snaps, snap_emb, Path(args.smartcap_parquet),
        strict_file_validity=args.strict_file_validity)
    (out_dir / "train.json").write_text(
        json.dumps({"repositories": train_repos}), encoding="utf-8")
    print(f"  -> {out_dir / 'train.json'}", flush=True)

    print("\n[2/3] Building cr_val.json ...", flush=True)
    parq = Path(args.parquet_dir)
    cr_val_repos, cr_val_stats = _build_eval_split(
        parq / "qna" / "cr_val.parquet",
        parq / "commits" / "cr_val.parquet",
        anchors, snap_emb)
    (out_dir / "cr_val.json").write_text(
        json.dumps({"repositories": cr_val_repos}), encoding="utf-8")
    print(f"  -> {out_dir / 'cr_val.json'}", flush=True)

    print("\n[3/3] Building cr_test.json ...", flush=True)
    cr_test_repos, cr_test_stats = _build_eval_split(
        parq / "qna" / "cr_test.parquet",
        parq / "commits" / "cr_test.parquet",
        anchors, snap_emb)
    (out_dir / "cr_test.json").write_text(
        json.dumps({"repositories": cr_test_repos}), encoding="utf-8")
    print(f"  -> {out_dir / 'cr_test.json'}", flush=True)

    readme = {
        "manifest": args.manifest,
        "snapshot_embeddings": args.snapshot_embeddings,
        "smartcap_parquet": args.smartcap_parquet,
        "parquet_dir": args.parquet_dir,
        "strict_file_validity": args.strict_file_validity,
        "train_stats": train_stats,
        "cr_val_stats": cr_val_stats,
        "cr_test_stats": cr_test_stats,
    }
    (out_dir / "STATIC_COMMIT_README.json").write_text(
        json.dumps(readme, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    for k, v in readme.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
