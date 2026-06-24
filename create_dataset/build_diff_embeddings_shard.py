#!/usr/bin/env python3
"""Compute 2048-d diff embeddings for one shard of the commits parquet.

Reuses the exact ``DiffEmbedder`` recipe used by Code2LoRA-GRU training:

    model        : Qwen/Qwen3-Embedding-0.6B
    chunk_tokens : 512
    overlap      : 64
    max_length   : 512
    pooling      : per-diff MaxPool || MeanPool over chunk embeddings  -> 2048-d
                   (NO L2 normalization; matches `pool_file_chunks_maxmean`)

Input parquets (read-only):

    $SCRATCH/REPO_DATASET/commit_parquet_hf/commits/{train,cr_val,cr_test}.parquet
        columns: repo_id, commit_sha, commit_index, cross_repo_split,
                 production_code_diff, ...

Output (one parquet per (split, shard)):

    <out-dir>/<split>/shard_<idx>_<total>.parquet
        columns: repo_id (string), commit_sha (string),
                 commit_index (int32), cross_repo_split (string),
                 diff_embedding (list<float16>[2048])

Sharding: repos are sorted globally across the three splits, then assigned to
shards via ``stride = i % shard_total == shard_index``. This balances both
repo count and total diff bytes between shards.

Typical wall on a single H100 with chunk_tokens=512 and a balanced shard
(~9 k commits, ~12 k chunks): 25-50 min per shard.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

# Reuse the canonical DiffEmbedder verbatim.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hypernetwork.train_code2lora_gru_commits import (  # noqa: E402
    DIFF_POOLING_MODES,
    DiffEmbedder,
)


DEFAULT_INPUT_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/commits"
DEFAULT_OUT_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2_shards/diff"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _list_all_repos(input_dir: Path,
                    splits: List[str]) -> Dict[str, List[str]]:
    """Return {repo_id: [splits it appears in]} across the requested splits."""
    out: Dict[str, List[str]] = {}
    for s in splits:
        path = input_dir / f"{s}.parquet"
        if not path.exists():
            print(f"  [warn] missing {path}", flush=True)
            continue
        t = pq.read_table(path, columns=["repo_id"], memory_map=True)
        for r in set(t.column("repo_id").to_pylist()):
            out.setdefault(r, []).append(s)
    return out


def _select_shard_repos(all_repos: Dict[str, List[str]],
                        shard_index: int, shard_total: int) -> List[str]:
    sorted_repos = sorted(all_repos.keys())
    return [r for i, r in enumerate(sorted_repos)
            if i % shard_total == shard_index]


def _load_shard_rows(input_dir: Path, split: str,
                     repo_set: set) -> Tuple[List[str], List[str],
                                              List[int], List[str]]:
    """Read one split's parquet and filter to repos in ``repo_set``.
    Returns parallel arrays (repo_id, commit_sha, commit_index, diff_text)."""
    path = input_dir / f"{split}.parquet"
    t = pq.read_table(
        path,
        columns=["repo_id", "commit_sha", "commit_index", "production_code_diff"],
        memory_map=True,
    )
    repo_col = t.column("repo_id").to_pylist()
    sha_col = t.column("commit_sha").to_pylist()
    idx_col = t.column("commit_index").to_pylist()
    diff_col = t.column("production_code_diff").to_pylist()
    repos, shas, idxs, diffs = [], [], [], []
    for r, s, i, d in zip(repo_col, sha_col, idx_col, diff_col):
        if r in repo_set:
            repos.append(r)
            shas.append(s)
            idxs.append(int(i))
            diffs.append(d if d is not None else "")
    return repos, shas, idxs, diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "cr_val", "cr_test"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--pooling", default="maxmean", choices=list(DIFF_POOLING_MODES),
                    help="Per/across-chunk pooling. 'maxmean' (default, Qwen3 "
                         "recipe -> 2*hidden) or 'lasttoken' (decoder-embedder "
                         "recipe: last-token + L2-norm -> hidden).")
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Tokenizer/model batch size INSIDE embed_diffs_batched.")
    ap.add_argument("--diff-batch", type=int, default=64,
                    help="Number of diffs grouped per embed_diffs_batched() call.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16", "auto"],
                    help="Model load dtype. Keep float32 for the small Qwen3 "
                         "encoder (reproducibility); use bfloat16/auto for "
                         "large models (e.g. harrier-27b) so they fit in GPU mem.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Smoke test: stop after this many diffs across all splits.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.splits:
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    # 1) Determine shard repos.
    all_repos = _list_all_repos(input_dir, args.splits)
    shard_repos = _select_shard_repos(all_repos, args.shard_index, args.shard_total)
    repo_set = set(shard_repos)
    n_total = sum(len(v) for v in all_repos.values())  # (repo, split) pairs
    print(f"Shard {args.shard_index+1}/{args.shard_total}: "
          f"{len(shard_repos)} repos / {len(all_repos)} total "
          f"({n_total} (repo,split) pairs).", flush=True)

    # 2) Load model.
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16, "auto": "auto"}[args.dtype]
    print(f"Loading {args.model_name} on {device} (dtype={args.dtype}, "
          f"pooling={args.pooling}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch_dtype).to(device)
    model.eval()
    embedder = DiffEmbedder(
        model=model,
        tokenizer=tokenizer,
        device=device,
        chunk_tokens=args.chunk_tokens,
        overlap=args.chunk_overlap,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )
    embed_dim = embedder.embed_dim  # 2*D=2048 (Qwen3 maxmean) or D=5376 (harrier lasttoken)
    print(f"  embed_dim = {embed_dim}", flush=True)

    # 3) Per-split processing.
    t_global = time.time()
    n_done = 0
    for split in args.splits:
        out_path = out_dir / split / f"shard_{args.shard_index:02d}_of_{args.shard_total:02d}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path} already exists.", flush=True)
            try:
                n_done += pq.read_metadata(out_path).num_rows
            except Exception:
                pass
            continue

        repos, shas, idxs, diffs = _load_shard_rows(input_dir, split, repo_set)
        print(f"[{split}] {len(diffs)} diffs in this shard.", flush=True)
        if not diffs:
            schema = pa.schema([
                ("repo_id", pa.string()),
                ("commit_sha", pa.string()),
                ("commit_index", pa.int32()),
                ("cross_repo_split", pa.string()),
                ("diff_embedding",
                 pa.list_(pa.float16(), int(embed_dim))),
            ])
            pq.write_table(pa.table({k: [] for k in schema.names}, schema=schema),
                           out_path, compression="zstd")
            continue

        all_embs = np.zeros((len(diffs), embed_dim), dtype=np.float16)
        t0 = time.time()
        B = max(1, int(args.diff_batch))
        for start in range(0, len(diffs), B):
            end = min(start + B, len(diffs))
            chunk_texts = diffs[start:end]
            embs = embedder.embed_diffs_batched(chunk_texts)  # [k, 2*D] fp32 cpu
            all_embs[start:end] = embs.numpy().astype(np.float16)
            n_done += (end - start)
            if args.limit and n_done >= args.limit:
                print(f"  [limit] stopping after {n_done} diffs.", flush=True)
                break
            if (end // B) % 10 == 0 or end == len(diffs):
                rate = end / max(time.time() - t0, 1e-3)
                eta = (len(diffs) - end) / max(rate, 1e-3)
                print(f"  [{split}] {end}/{len(diffs)} ({rate:.1f}/s, "
                      f"eta {eta/60:.1f}m)", flush=True)

        # Truncate if limit hit early.
        if args.limit and n_done >= args.limit:
            keep = min(len(diffs), args.limit)
            repos, shas, idxs = repos[:keep], shas[:keep], idxs[:keep]
            all_embs = all_embs[:keep]

        # Write parquet.
        emb_list = all_embs.tolist()
        table = pa.table({
            "repo_id": pa.array(repos, type=pa.string()),
            "commit_sha": pa.array(shas, type=pa.string()),
            "commit_index": pa.array(idxs, type=pa.int32()),
            "cross_repo_split": pa.array([split] * len(repos), type=pa.string()),
            "diff_embedding": pa.array(
                emb_list,
                type=pa.list_(pa.float16(), int(embed_dim)),
            ),
        })
        tmp_path = out_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)
        print(f"[{split}] wrote {out_path} ({len(repos)} rows)", flush=True)

        if args.limit and n_done >= args.limit:
            break

    print(f"\nDone. {n_done} diffs total in {(time.time()-t_global)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
