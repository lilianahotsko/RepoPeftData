#!/usr/bin/env python3
"""
Flatten the OOD commit-parquet into a static ``ood_test.json`` (and
``gru_ood_test.json``) compatible with the existing baseline eval scripts.

Output format mirrors ``cr_test.json`` so any script that takes
``--split ood_test`` and looks up ``repositories[repo_id].qna_pairs`` will work
without modification.

What this writes
----------------

* ``$SPLITS_DIR/ood_test.json`` -- minimal shape, no embeddings, used by
  pretrained / FFT / sLoRA / RAG / ICL / DRC / T2L-text.
* ``$SPLITS_DIR/gru_ood_test.json`` -- includes a ``commit_history`` block
  (file order taken at HEAD) so the file-level GRU evaluator can also load it
  if file embeddings are later backfilled.

What this does NOT write
------------------------

* Per-repo static ``embedding`` (the 2048-dim mean+max pool used by direct
  \method and T2L-code). Computing it requires re-running the Qwen3-Embedding
  pipeline over the OOD repos at HEAD; do that separately via
  ``embed_repos/4_construct_embeddings.py`` if you want those rows.
* Per-file ``file_embeddings`` (used by Code2LoRA-GRU\textsubscript{file}).

We deduplicate QnAs by ``(test_file, target_lineno, target)`` so each unique
assertion only appears once even if it was added/modified across multiple
commits in the OOD parquet. This matches the static RepoPeftBench protocol
where we score each assertion in its current form once.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qna-parquet", type=Path,
        default=Path(os.environ.get("SCRATCH", "."))
        / "REPO_DATASET" / "commit_parquet_ood" / "qna_pairs.parquet",
    )
    ap.add_argument(
        "--commits-parquet", type=Path,
        default=Path(os.environ.get("SCRATCH", "."))
        / "REPO_DATASET" / "commit_parquet_ood" / "commits.parquet",
    )
    ap.add_argument(
        "--splits-dir", type=Path,
        default=Path(os.environ.get("SCRATCH", "."))
        / "REPO_DATASET",
    )
    ap.add_argument("--bench-name", type=str, default="ood_test")
    ap.add_argument("--gru-bench-name", type=str, default="gru_ood_test")
    ap.add_argument(
        "--max-pairs-per-repo", type=int, default=120,
        help=(
            "Cap pairs per repo (default 120 to match cr_test.json scale; "
            "0 = no cap)."
        ),
    )
    ap.add_argument(
        "--max-prefix-chars", type=int, default=32768,
        help=(
            "Truncate prefix to the last N characters (default 32K, ample "
            "for a 16K-token max-input-tokens window). 0 = no truncation. "
            "The OOD parquet builds prefixes from full test-file context so "
            "untrimmed JSONs blow up to GB scale."
        ),
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    if not args.qna_parquet.exists():
        raise SystemExit(f"missing {args.qna_parquet}")
    print(f"Reading QnAs from {args.qna_parquet} ...")
    qna_tbl = pq.read_table(args.qna_parquet)
    qna_df = qna_tbl.to_pandas()
    print(f"  {len(qna_df):,} rows, columns={list(qna_df.columns)}")

    # Normalize the commit-index column name; the OOD parquet uses
    # ``commit_index`` while older schemas used ``commit_idx``.
    if "commit_idx" not in qna_df.columns:
        for cand in ("commit_index", "step_idx", "commit_position"):
            if cand in qna_df.columns:
                qna_df = qna_df.rename(columns={cand: "commit_idx"})
                break
    if "commit_idx" not in qna_df.columns:
        qna_df["commit_idx"] = 0

    # Deduplicate to most-recent assertion per (test_file, lineno). We keep the
    # row with the largest ``commit_idx`` so the snapshot reflects the latest
    # state of the assertion.
    qna_df = qna_df.sort_values(
        ["repo_id", "test_file", "lineno", "commit_idx"],
        ascending=[True, True, True, True],
    )
    qna_df = qna_df.drop_duplicates(
        subset=["repo_id", "test_file", "lineno"], keep="last",
    )
    qna_df = qna_df[
        qna_df["target"].astype(str).str.lstrip().str.startswith(",") == False  # noqa: E712
    ]
    qna_df = qna_df[qna_df["prefix"].astype(str).str.len() > 0]
    qna_df = qna_df[qna_df["target"].astype(str).str.len() > 0]
    print(f"  after dedup + filter: {len(qna_df):,} rows")

    # Group into the cr_test.json shape. The OOD parquet has the columns:
    #   repo_id, cross_repo_split, commit_idx, commit_sha, in_repo_split,
    #   test_file, lineno, col_offset, assertion_type, test_function,
    #   prefix, target
    repos: Dict[str, Dict] = defaultdict(lambda: {"qna_pairs": []})
    max_pref = int(args.max_prefix_chars)
    for _, row in qna_df.iterrows():
        rid = str(row["repo_id"])
        col_offset = int(row.get("col_offset") or 0)
        prefix = str(row["prefix"])
        if max_pref > 0 and len(prefix) > max_pref:
            prefix = prefix[-max_pref:]
        pair = {
            "prefix": prefix,
            "target": str(row["target"]),
            "assertion_type": str(row.get("assertion_type") or ""),
            "metadata": {
                "file": str(row.get("test_file") or ""),
                "lineno": int(row.get("lineno") or 0),
                "col_offset": col_offset,
                # The OOD parquet does not record ``target_col_offset``; use
                # ``col_offset`` as a best-effort fallback for the exec pilot.
                "target_col_offset": col_offset,
                "test_function": str(row.get("test_function") or ""),
                "test_class": "",
                "repo": rid,
                "source_file": str(row.get("test_file") or ""),
                "commit_idx": int(row.get("commit_idx") or 0),
            },
            "commit_idx": int(row.get("commit_idx") or 0),
        }
        repos[rid]["qna_pairs"].append(pair)

    if args.max_pairs_per_repo > 0:
        for rid in repos:
            repos[rid]["qna_pairs"] = repos[rid]["qna_pairs"][: args.max_pairs_per_repo]

    out = {"split": args.bench_name, "repositories": dict(repos)}
    out_path = args.splits_dir / f"{args.bench_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_pairs = sum(len(r["qna_pairs"]) for r in repos.values())
    print(f"Wrote {out_path} : {len(repos)} repos, {n_pairs:,} pairs")

    # ----- gru_ood_test.json (adds an empty embedding + commit_history stub
    # so the gru_file eval can run once file_embeddings are backfilled) -----
    print(f"Reading commits from {args.commits_parquet} ...")
    commits_df = None
    if args.commits_parquet.exists():
        commits_tbl = pq.read_table(args.commits_parquet)
        commits_df = commits_tbl.to_pandas()
        print(f"  {len(commits_df):,} commit rows, "
              f"columns={list(commits_df.columns)}")
    gru_repos: Dict[str, Dict] = {}
    for rid, info in repos.items():
        gru_repos[rid] = {
            "qna_pairs": info["qna_pairs"],
            "embedding": None,
            "commit_history": None,
        }
    if commits_df is not None:
        # Normalize commit-index column name.
        if "commit_idx" not in commits_df.columns and "commit_index" in commits_df.columns:
            commits_df = commits_df.rename(columns={"commit_index": "commit_idx"})
        for rid in gru_repos:
            sub = commits_df[commits_df["repo_id"] == rid]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("commit_idx")
            preamble_cutoff = max(1, int(0.1 * len(sub)))
            # The OOD parquet only stores ``production_code_diff`` blobs, not
            # per-commit changed-path lists. We therefore record a minimal
            # commit_history stub: counts and a preamble cutoff, but no
            # ``file_order`` (the gru_file evaluator requires per-file
            # embeddings anyway, which are not available for OOD without an
            # additional embedding pass).
            gru_repos[rid]["commit_history"] = {
                "total_commits_in_repo": int(len(sub)),
                "sampled_commit_count": int(len(sub)),
                "preamble_commit_cutoff": preamble_cutoff,
                "preamble_files": [],
                "file_order": [],
                "commits": [],
            }
    out2 = {"split": args.gru_bench_name, "repositories": gru_repos}
    out_path2 = args.splits_dir / f"{args.gru_bench_name}.json"
    out_path2.write_text(json.dumps(out2, indent=2), encoding="utf-8")
    print(f"Wrote {out_path2} (no per-repo embeddings; backfill if needed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
