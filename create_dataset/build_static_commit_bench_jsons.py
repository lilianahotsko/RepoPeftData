#!/usr/bin/env python3
"""Build IR-test and OOD-test bench JSONs for the static-commit eval.

Step 4 of the static-commit pipeline (cr_val / cr_test are produced by
``build_static_commit_train_jsonl.py`` because the trainer wants them in its
splits dir for end-of-epoch eval).

Each output JSON has the canonical RepoPeftBench layout consumed by
``evaluation/run_repopeft_bench.py --bench-json``::

    {"repositories": {"<author>/<repo>@@<commit_sha>": {
        "embedding": [2048],
        "qna_pairs": [{"prefix", "target", "assertion_type", "metadata"}],
        "metadata": {"repo_id", "commit_sha", "commit_index", "in_repo_split",
                     "anchor_sha", "anchor_index", "n_commits_after_anchor",
                     "suite"}}}}

Used as ``run_repopeft_bench.py --method direct --bench-json <here>``; the
direct-method install fn already reads ``repo_item['embedding']`` straight
off each item, so per-(repo, commit) eval falls out for free.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import pyarrow.compute as pc


SUITES = {
    "ir_test": (
        "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/qna/train.parquet",
        "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/commits/train.parquet",
        "test",  # held-out in-repo test commits
    ),
    "ood_test": (
        "/scratch/lhotsko/REPO_DATASET/commit_parquet_ood/qna_pairs.parquet",
        "/scratch/lhotsko/REPO_DATASET/commit_parquet_ood/commits.parquet",
        None,
    ),
}


def _read_anchors(manifest_path: Path) -> Dict[str, Tuple[str, int]]:
    anchors: Dict[str, Tuple[str, int]] = {}
    with manifest_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["repo_id"] in anchors:
                continue
            anchors[row["repo_id"]] = (row["anchor_sha"], int(row["anchor_index"]))
    return anchors


def build_suite(name: str, qna_path: Path, commits_path: Path,
                in_repo_split: str | None,
                anchors: Dict[str, Tuple[str, int]],
                snap_emb: Dict[str, List[float]],
                out_path: Path) -> dict:
    qna = pq.read_table(qna_path)
    if in_repo_split is not None:
        qna = qna.filter(pc.equal(qna.column("in_repo_split"), in_repo_split))
    print(f"[{name}] {qna.num_rows:,} QnA rows", flush=True)

    commits_pf = pq.ParquetFile(commits_path)
    cols = [c for c in ["repo_id", "commit_sha", "commit_index", "in_repo_split"]
            if c in commits_pf.schema.names]
    cdf = pq.read_table(commits_path, columns=cols).to_pandas().drop_duplicates(
        ["repo_id", "commit_sha"])
    cmeta = {(r["repo_id"], r["commit_sha"]): (
        int(r["commit_index"]),
        str(r["in_repo_split"]) if "in_repo_split" in cdf.columns else "")
        for _, r in cdf.iterrows()}

    by_pair: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in qna.to_pandas().itertuples(index=False):
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
    n_missing = 0
    for (repo_id, sha), pairs in by_pair.items():
        emb = snap_emb.get(f"{repo_id}@{sha}")
        if emb is None:
            n_missing += 1
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
                "suite": name,
            },
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"repositories": repositories}),
                        encoding="utf-8")
    summary = {
        "suite": name, "out_path": str(out_path),
        "n_items": len(repositories),
        "n_qna_kept": sum(len(v["qna_pairs"]) for v in repositories.values()),
        "n_missing_emb": n_missing,
    }
    print(f"[{name}] {summary}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--snapshot-embeddings",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json")
    ap.add_argument("--out-dir",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/bench")
    ap.add_argument("--suites", nargs="*", default=list(SUITES))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = _read_anchors(Path(args.manifest))
    snap_emb = json.loads(Path(args.snapshot_embeddings).read_text(encoding="utf-8"))
    summaries = []
    for name in args.suites:
        qna_p, commits_p, irs = SUITES[name]
        if not Path(qna_p).exists() or not Path(commits_p).exists():
            print(f"[skip] {name}: parquet missing"); continue
        summaries.append(build_suite(
            name, Path(qna_p), Path(commits_p), irs,
            anchors, snap_emb,
            out_dir / f"{name}_static_commit.json"))
    (out_dir / "BUILD_SUMMARY.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
