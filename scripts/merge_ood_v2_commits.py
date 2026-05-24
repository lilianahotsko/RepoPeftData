#!/usr/bin/env python3
"""Merge OOD commits + repo_state_embedding shards + diff_embedding shards
into a single ``commit_parquet_hf_v2/commits/ood_test.parquet`` that matches
the schema of ``cr_test.parquet`` so ``run_code2lora_gru_v2_eval.py`` can
ingest OOD with no further changes.

Sources
-------
- ``$SCRATCH/REPO_DATASET/commit_parquet_ood/commits.parquet``
    (repo_id, commit_sha, commit_index, cross_repo_split, commit_timestamp,
     in_repo_split, production_code_diff, n_new_assertions)
- ``$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_ood/repo_state/ood_test/
    shard_*_of_04.parquet``
    (repo_id, commit_sha, commit_index, cross_repo_split, repo_state_embedding)
- ``$SCRATCH/REPO_DATASET/commit_parquet_hf_v2_shards/diff/ood_test/
    shard_*_of_04.parquet`` (built by ``build_diff_embeddings_ood.sh``)
    (repo_id, commit_sha, commit_index, cross_repo_split, diff_embedding)

Output
------
``$SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits/ood_test.parquet``
with the exact column order/type of ``cr_test.parquet``. The
``n_added_assertions`` / ``n_modified_assertions`` columns -- absent from
OOD -- are filled with 0.

Notes
-----
- A row is emitted only if it has BOTH a ``diff_embedding`` and a
  ``repo_state_embedding`` (the GRU needs both); rows missing either are
  dropped with a warning.
- Inner-join key is ``(repo_id, commit_sha)`` which is unique across the
  three source families.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")

DEFAULT_COMMITS = f"{SCRATCH}/REPO_DATASET/commit_parquet_ood/commits.parquet"
DEFAULT_REPO_STATE_GLOB = (
    f"{SCRATCH}/REPO_DATASET/commit_parquet_hf_v2_ood/repo_state/ood_test/"
    "shard_*_of_*.parquet"
)
DEFAULT_DIFF_GLOB = (
    f"{SCRATCH}/REPO_DATASET/commit_parquet_hf_v2_shards/diff/ood_test/"
    "shard_*_of_*.parquet"
)
DEFAULT_OUT = (
    f"{SCRATCH}/REPO_DATASET/commit_parquet_hf_v2/commits/ood_test.parquet"
)
TARGET_SCHEMA_REF = (
    f"{SCRATCH}/REPO_DATASET/commit_parquet_hf_v2/commits/cr_test.parquet"
)


def _read_concat(glob_pat: str) -> pa.Table:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"No parquets found at {glob_pat}")
    print(f"[read] {glob_pat}: {len(paths)} shard(s)")
    tables = [pq.read_table(p) for p in paths]
    return pa.concat_tables(tables, promote_options="default")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits", default=DEFAULT_COMMITS)
    ap.add_argument("--repo-state-glob", default=DEFAULT_REPO_STATE_GLOB)
    ap.add_argument("--diff-glob", default=DEFAULT_DIFF_GLOB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--schema-ref", default=TARGET_SCHEMA_REF)
    args = ap.parse_args()

    # 1) Load each source.
    commits_t = pq.read_table(args.commits)
    print(f"[commits] {commits_t.num_rows} rows, cols={commits_t.column_names}")
    rs_t = _read_concat(args.repo_state_glob)
    print(f"[repo_state] {rs_t.num_rows} rows, cols={rs_t.column_names}")
    diff_t = _read_concat(args.diff_glob)
    print(f"[diff] {diff_t.num_rows} rows, cols={diff_t.column_names}")

    # 2) Build lookups by (repo_id, commit_sha).
    def _key_to_row(t: pa.Table) -> Dict[Tuple[str, str], int]:
        repos = t.column("repo_id").to_pylist()
        shas = t.column("commit_sha").to_pylist()
        return {(r, s): i for i, (r, s) in enumerate(zip(repos, shas))}

    rs_keys = _key_to_row(rs_t)
    diff_keys = _key_to_row(diff_t)
    print(f"[keys] repo_state has {len(rs_keys)} unique (repo,sha); "
          f"diff has {len(diff_keys)} unique (repo,sha)")

    # 3) Reference schema to mirror exactly.
    ref_schema = pq.read_schema(args.schema_ref)
    print(f"[schema_ref] {len(ref_schema)} columns:")
    for f in ref_schema:
        print(f"    {f.name:30s} {f.type}")

    # 4) Walk commits, emit rows that have both embeddings.
    commits_repos = commits_t.column("repo_id").to_pylist()
    commits_shas = commits_t.column("commit_sha").to_pylist()
    commits_indices = commits_t.column("commit_index").to_pylist()
    commits_split = commits_t.column("cross_repo_split").to_pylist()
    commits_ts = commits_t.column("commit_timestamp").to_pylist()
    commits_in_repo = commits_t.column("in_repo_split").to_pylist()
    commits_diff = commits_t.column("production_code_diff").to_pylist()
    commits_n_new = commits_t.column("n_new_assertions").to_pylist()

    rs_emb_col = rs_t.column("repo_state_embedding")
    diff_emb_col = diff_t.column("diff_embedding")

    keep_rows: list[int] = []
    rs_idxs: list[int] = []
    diff_idxs: list[int] = []
    n_missing_diff = 0
    n_missing_rs = 0
    for i in range(commits_t.num_rows):
        k = (commits_repos[i], commits_shas[i])
        ri = rs_keys.get(k)
        di = diff_keys.get(k)
        if ri is None:
            n_missing_rs += 1
            continue
        if di is None:
            n_missing_diff += 1
            continue
        keep_rows.append(i)
        rs_idxs.append(ri)
        diff_idxs.append(di)
    print(f"[merge] {len(keep_rows)} rows survive; "
          f"missing repo_state for {n_missing_rs}, missing diff for {n_missing_diff}")
    if not keep_rows:
        raise SystemExit("No rows survived the join -- aborting.")

    # 5) Materialize columns in the target order.
    def _take_strings(src: list, idxs: list[int]) -> list:
        return [src[i] for i in idxs]

    repo_id = pa.array(_take_strings(commits_repos, keep_rows), type=pa.string())
    cross_repo_split = pa.array(_take_strings(commits_split, keep_rows), type=pa.string())
    commit_index = pa.array(_take_strings(commits_indices, keep_rows), type=pa.int32())
    commit_sha = pa.array(_take_strings(commits_shas, keep_rows), type=pa.string())

    # commit_timestamp may be either str (ISO 8601) or datetime; cr_test uses string.
    ts_list = _take_strings(commits_ts, keep_rows)
    commit_timestamp = pa.array(
        ["" if v is None else (str(v) if not isinstance(v, str) else v)
         for v in ts_list],
        type=pa.string(),
    )

    in_repo_split = pa.array(_take_strings(commits_in_repo, keep_rows), type=pa.string())
    production_code_diff = pa.array(
        _take_strings(commits_diff, keep_rows), type=pa.large_string()
    )
    n_new_assertions = pa.array(_take_strings(commits_n_new, keep_rows), type=pa.int32())
    # n_added / n_modified are absent in OOD; fill with 0 to satisfy schema.
    zeros = pa.array([0] * len(keep_rows), type=pa.int32())

    rs_taken = rs_emb_col.take(pa.array(rs_idxs, type=pa.int64())).combine_chunks()
    diff_taken = diff_emb_col.take(pa.array(diff_idxs, type=pa.int64())).combine_chunks()

    fields = list(ref_schema)
    col_by_name = {
        "repo_id": repo_id,
        "cross_repo_split": cross_repo_split,
        "commit_index": commit_index,
        "commit_sha": commit_sha,
        "commit_timestamp": commit_timestamp,
        "in_repo_split": in_repo_split,
        "production_code_diff": production_code_diff,
        "n_new_assertions": n_new_assertions,
        "n_added_assertions": zeros,
        "n_modified_assertions": zeros,
        "diff_embedding": diff_taken,
        "repo_state_embedding": rs_taken,
    }
    columns = [col_by_name[f.name] for f in fields]
    out_t = pa.Table.from_arrays(columns, schema=ref_schema)
    print(f"[out] {out_t.num_rows} rows, schema OK")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_t, str(out_path), compression="zstd")
    print(f"[done] wrote {out_path} "
          f"({out_path.stat().st_size / (1024**2):.1f} MiB)")


if __name__ == "__main__":
    main()
