#!/usr/bin/env python3
"""Materialize a snapshot-format ood_test.parquet for Code2LoRA-static eval.

Reads:
  - $SCRATCH/REPO_DATASET/commit_parquet_ood/commits.parquet
    (canonical OOD commit manifest: 1,950 commits, 92 repos)
  - $SCRATCH/REPO_DATASET/commit_parquet_hf_v2_ood/repo_state/ood_test/shard_*.parquet
    (per-commit repo_state_embedding shards produced by
     build_repo_state_embeddings_shard.py)

Writes:
  - $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/ood_test.parquet
    All OOD commits annotated with repo_state_embedding, kept in
    snapshot-style columns matching the existing cr_test.parquet schema:
        repo_id, cross_repo_split, commit_index, commit_sha,
        commit_timestamp, in_repo_split, n_new_assertions,
        n_added_assertions (set to 0 if absent), n_modified_assertions
        (set to 0 if absent), element (set to '' if absent),
        repo_state_embedding (list<float16>[2048])
  - $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/ood_test.parquet
    Symlink to the existing OOD qna parquet (same schema as a v2 qna parquet).

The C2L-static eval (run_code2lora_static_v2_eval.py) groups by repo and
iterates over all (repo, snapshot) rows, so retaining all 1,950 OOD commits
maximizes the matched evaluation set with Code2LoRA-GRU on OOD.
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


EMBED_DIM = 2048
DEFAULT_COMMITS_PATH = "/scratch/lhotsko/REPO_DATASET/commit_parquet_ood/commits.parquet"
DEFAULT_REPO_STATE_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2_ood/repo_state/ood_test"
DEFAULT_OUT_COMMITS = "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/commits/ood_test.parquet"
DEFAULT_OUT_QNA = "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/qna/ood_test.parquet"
DEFAULT_OOD_QNA = "/scratch/lhotsko/REPO_DATASET/commit_parquet_ood/qna_pairs.parquet"


def _read_emb_shards(shards_dir: Path) -> Dict[Tuple[str, str], np.ndarray]:
    """Return {(repo_id, commit_sha): emb_vec[fp16]} merged across shards."""
    shards = sorted(glob.glob(str(shards_dir / "shard_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {shards_dir}")
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for sh in shards:
        t = pq.read_table(sh, memory_map=True)
        repos = t.column("repo_id").to_pylist()
        shas = t.column("commit_sha").to_pylist()
        embs = t.column("repo_state_embedding").to_pylist()
        for r, s, e in zip(repos, shas, embs):
            out[(r, s)] = np.asarray(e, dtype=np.float16)
        print(f"  read {sh}: +{t.num_rows} rows (total {len(out)})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits-path", default=DEFAULT_COMMITS_PATH)
    ap.add_argument("--repo-state-dir", default=DEFAULT_REPO_STATE_DIR)
    ap.add_argument("--out-commits", default=DEFAULT_OUT_COMMITS)
    ap.add_argument("--out-qna", default=DEFAULT_OUT_QNA)
    ap.add_argument("--ood-qna", default=DEFAULT_OOD_QNA)
    args = ap.parse_args()

    commits_path = Path(args.commits_path)
    shards_dir = Path(args.repo_state_dir)
    out_commits = Path(args.out_commits)
    out_qna = Path(args.out_qna)
    out_commits.parent.mkdir(parents=True, exist_ok=True)
    out_qna.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] reading OOD commits manifest {commits_path}", flush=True)
    t = pq.read_table(commits_path, memory_map=True)
    n = t.num_rows
    repo_col = t.column("repo_id").to_pylist()
    sha_col = t.column("commit_sha").to_pylist()
    idx_col = t.column("commit_index").to_pylist()
    # commit_timestamp is a string (ISO 8601) in both the v2 snapshot parquets
    # and the OOD commits parquet -- keep it as a string for schema parity.
    ts_col = (t.column("commit_timestamp").to_pylist()
              if "commit_timestamp" in t.column_names else [""] * n)
    ts_col = [str(x) if x is not None else "" for x in ts_col]
    irs_col = (t.column("in_repo_split").to_pylist()
               if "in_repo_split" in t.column_names else [""] * n)
    crs_col = (t.column("cross_repo_split").to_pylist()
               if "cross_repo_split" in t.column_names else ["ood_test"] * n)
    n_new = (t.column("n_new_assertions").to_pylist()
             if "n_new_assertions" in t.column_names else [0] * n)
    print(f"  rows={n:,}  unique repos={len(set(repo_col)):,}", flush=True)

    print(f"[2/3] reading per-commit repo_state embeddings from {shards_dir}",
          flush=True)
    emb_map = _read_emb_shards(shards_dir)
    print(f"  emb_map covers {len(emb_map):,} (repo, commit) keys", flush=True)

    emb_arr = np.zeros((n, EMBED_DIM), dtype=np.float16)
    n_hit = 0
    n_miss = 0
    miss_examples = []
    for i, (r, s) in enumerate(zip(repo_col, sha_col)):
        k = (r, s)
        if k in emb_map:
            emb_arr[i] = emb_map[k]
            n_hit += 1
        else:
            n_miss += 1
            if len(miss_examples) < 5:
                miss_examples.append(k)
    print(f"  hits: {n_hit:,}/{n:,}  misses: {n_miss:,}", flush=True)
    if miss_examples:
        print(f"  miss examples: {miss_examples}", flush=True)

    print(f"[3/3] writing snapshot parquet -> {out_commits}", flush=True)
    # Schema mirrors $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/cr_test.parquet:
    #   commit_timestamp is a string (ISO 8601), there is no "element" column,
    #   and repo_state_embedding is a fixed_size_list<float16>[2048].
    table = pa.table({
        "repo_id":              pa.array(repo_col, type=pa.string()),
        "cross_repo_split":     pa.array(crs_col, type=pa.string()),
        "commit_index":         pa.array(idx_col, type=pa.int32()),
        "commit_sha":           pa.array(sha_col, type=pa.string()),
        "commit_timestamp":     pa.array(ts_col, type=pa.string()),
        "in_repo_split":        pa.array(irs_col, type=pa.string()),
        "n_new_assertions":     pa.array(n_new, type=pa.int32()),
        "n_added_assertions":   pa.array([0] * n, type=pa.int32()),
        "n_modified_assertions": pa.array([0] * n, type=pa.int32()),
        "repo_state_embedding": pa.array(emb_arr.tolist(),
                                         type=pa.list_(pa.float16(), EMBED_DIM)),
    })
    tmp = out_commits.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_commits)
    print(f"  wrote {out_commits}  ({n} rows)", flush=True)

    # Symlink qna parquet (schema-compatible with v2 layout)
    if out_qna.exists() or out_qna.is_symlink():
        out_qna.unlink()
    os.symlink(args.ood_qna, out_qna)
    print(f"  symlinked qna -> {out_qna} -> {args.ood_qna}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
