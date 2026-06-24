#!/usr/bin/env python3
"""Merge diff/repo-state embedding shards into the v2 commits parquet.

Inputs:
    --commits-dir   <base>/commits/{train,cr_val,cr_test}.parquet
                    (the original GRU dataset; columns are unchanged)
    --diff-dir      <base>/diff/<split>/shard_*.parquet
    --repo-dir      <base>/repo_state/<split>/shard_*.parquet

Output:
    --out-dir       <out>/commits/{train,cr_val,cr_test}.parquet
                    same schema as the input + two new columns:
                      diff_embedding       list<float16>[2048]
                      repo_state_embedding list<float16>[2048]
    --out-dir/EMBEDDINGS_README.json
                    Provenance: model name, chunk_tokens, pooling order,
                    L2 normalization flag, original shard manifest, sha sums.

Memory: we read one split at a time. For ~58 k commits x 2048 fp16 x 2 cols,
that's ~480 MB plus the original 425 MB diff column, well within 32 GB.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_COMMITS_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/commits"
DEFAULT_SHARDS_BASE = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2_shards"
DEFAULT_OUT_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2"
EMBED_DIM = 2048


def _file_sha256(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_embedding_shards(shards_dir: Path, embedding_col: str
                           ) -> Tuple[Dict[tuple, np.ndarray], int]:
    """Read all shard parquets under shards_dir.

    Returns ``({(repo, sha): emb_vec[fp16]}, embed_dim)`` where embed_dim is
    inferred from the first non-empty vector (0 if no rows)."""
    shards = sorted(glob.glob(str(shards_dir / "shard_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {shards_dir}")
    out: Dict[tuple, np.ndarray] = {}
    dim = 0
    for sh in shards:
        t = pq.read_table(sh, memory_map=True)
        repo = t.column("repo_id").to_pylist()
        sha = t.column("commit_sha").to_pylist()
        emb = t.column(embedding_col).to_pylist()
        # to_pylist() yields python lists; convert to numpy fp16 lazily.
        for r, s, e in zip(repo, sha, emb):
            v = np.asarray(e, dtype=np.float16)
            out[(r, s)] = v
            if dim == 0 and v.size:
                dim = int(v.size)
        print(f"  read {sh}: +{t.num_rows} rows (running total {len(out)})",
              flush=True)
    return out, dim


def _merge_split(commits_path: Path,
                 diff_map: Dict[tuple, np.ndarray],
                 repo_map: Dict[tuple, np.ndarray],
                 out_path: Path,
                 diff_dim: int,
                 repo_dim: int) -> Dict[str, int]:
    """Append diff_embedding + repo_state_embedding to the commits table."""
    t = pq.read_table(commits_path, memory_map=True)
    n = t.num_rows
    repo_col = t.column("repo_id").to_pylist()
    sha_col = t.column("commit_sha").to_pylist()

    diff_arr = np.zeros((n, diff_dim), dtype=np.float16)
    repo_arr = np.zeros((n, repo_dim), dtype=np.float16)
    n_diff_hit = 0
    n_repo_hit = 0
    for i, (r, s) in enumerate(zip(repo_col, sha_col)):
        k = (r, s)
        if k in diff_map:
            diff_arr[i] = diff_map[k]
            n_diff_hit += 1
        if k in repo_map:
            repo_arr[i] = repo_map[k]
            n_repo_hit += 1

    diff_col = pa.array(diff_arr.tolist(),
                        type=pa.list_(pa.float16(), diff_dim))
    repo_col_arr = pa.array(repo_arr.tolist(),
                            type=pa.list_(pa.float16(), repo_dim))
    t2 = t.append_column("diff_embedding", diff_col)
    t2 = t2.append_column("repo_state_embedding", repo_col_arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(t2, tmp, compression="zstd")
    os.replace(tmp, out_path)
    return {"rows": n, "diff_hits": n_diff_hit, "repo_hits": n_repo_hit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits-dir", default=DEFAULT_COMMITS_DIR)
    ap.add_argument("--shards-base", default=DEFAULT_SHARDS_BASE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "cr_val", "cr_test"])
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B",
                    help="Encoder used to produce the shards (provenance only).")
    ap.add_argument("--pooling", default="maxmean",
                    choices=["maxmean", "lasttoken"],
                    help="Pooling used to produce the shards (provenance only).")
    args = ap.parse_args()

    commits_dir = Path(args.commits_dir)
    shards_base = Path(args.shards_base)
    out_dir = Path(args.out_dir)
    out_commits_dir = out_dir / "commits"
    out_commits_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    for split in args.splits:
        commits_path = commits_dir / f"{split}.parquet"
        diff_dir = shards_base / "diff" / split
        repo_dir = shards_base / "repo_state" / split
        out_path = out_commits_dir / f"{split}.parquet"
        print(f"\n=== {split} ===", flush=True)
        print(f"  commits: {commits_path}", flush=True)
        print(f"  diff   : {diff_dir}", flush=True)
        print(f"  repo   : {repo_dir}", flush=True)

        t0 = time.time()
        diff_map, diff_dim = _read_embedding_shards(diff_dir, "diff_embedding")
        print(f"  diff_map: {len(diff_map)} keys, dim={diff_dim} "
              f"({(time.time()-t0):.1f}s)", flush=True)
        t1 = time.time()
        repo_map, repo_dim = _read_embedding_shards(repo_dir, "repo_state_embedding")
        print(f"  repo_map: {len(repo_map)} keys, dim={repo_dim} "
              f"({(time.time()-t1):.1f}s)", flush=True)
        diff_dim = diff_dim or EMBED_DIM
        repo_dim = repo_dim or EMBED_DIM

        stats = _merge_split(commits_path, diff_map, repo_map, out_path,
                             diff_dim=diff_dim, repo_dim=repo_dim)
        stats["diff_dim"] = diff_dim
        stats["repo_dim"] = repo_dim
        sha = _file_sha256(out_path)
        stats["sha256"] = sha
        stats["out_path"] = str(out_path)
        summary[split] = stats
        print(f"  -> {out_path}  rows={stats['rows']}  "
              f"diff_hits={stats['diff_hits']}  repo_hits={stats['repo_hits']}  "
              f"sha256={sha[:16]}...", flush=True)

    # Also copy / symlink qna + splits + provenance from the v1 dataset so the
    # output dir is self-contained for the push step. We just write a small
    # README describing the new columns; the push script will copy v1 qna/.
    _any = next(iter(summary.values()), {}) if summary else {}
    diff_dim = _any.get("diff_dim", EMBED_DIM)
    repo_dim = _any.get("repo_dim", EMBED_DIM)
    if args.pooling == "lasttoken":
        diff_pooling = "last-token + L2-norm per chunk -> mean over chunks + L2-norm"
        file_pooling = "last-token + L2-norm per chunk -> mean over chunks + L2-norm"
        repo_pooling = "mean over file vectors then L2-normalize"
        diff_norm = repo_norm = "L2"
    else:
        diff_pooling = "concat(MaxPool, MeanPool)"
        file_pooling = "attention_mean -> mean over chunks"
        repo_pooling = "concat(mean_files, max_files) then L2-normalize"
        diff_norm, repo_norm = "none", "L2"
    readme = {
        "model": args.model_name,
        "pooling": args.pooling,
        "diff_embedding": {
            "dim": diff_dim,
            "dtype": "float16",
            "chunk_tokens": 512,
            "chunk_overlap": 64,
            "max_length": 512,
            "pooling": diff_pooling,
            "normalization": diff_norm,
            "source": "production_code_diff (filtered, test hunks removed)",
        },
        "repo_state_embedding": {
            "dim": repo_dim,
            "dtype": "float16",
            "chunk_tokens": 2048,
            "chunk_overlap": 256,
            "max_file_bytes": 2_000_000,
            "min_window_tokens": 8,
            "file_pooling": file_pooling,
            "repo_pooling": repo_pooling,
            "normalization": repo_norm,
            "file_filter": "tracked .py blobs <= 2 MB",
        },
        "splits": summary,
    }
    (out_dir / "EMBEDDINGS_README.json").write_text(
        json.dumps(readme, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'EMBEDDINGS_README.json'}", flush=True)


if __name__ == "__main__":
    main()
