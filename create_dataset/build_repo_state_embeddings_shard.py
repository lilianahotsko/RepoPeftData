#!/usr/bin/env python3
"""Compute (or load from cache) whole-repo 2048-d embeddings for one shard.

Output (one parquet per (split, shard)):

    <out-dir>/<split>/shard_<idx>_<total>.parquet
        columns: repo_id (string), commit_sha (string),
                 commit_index (int32), cross_repo_split (string),
                 repo_state_embedding (list<float16>[2048])

Per-snapshot pooling (matches ``build_static_commit_embeddings.py``)::

    file_vec = mean(chunk embeddings)                                      [1024]
    repo_vec = L2_normalize(concat(mean_files, max_files))                 [2048]

Heavy lifting reuses the existing per-repo blob cache at::

    /scratch/lhotsko/REPO_DATASET/static_commit/cache/<author>__<repo>/
        blob_embeddings.f16.npy      one row per blob_sha (already filled)
        blob_shas.txt                row order
        snapshot_embeddings.npz      already-pooled snapshots

Logic per (repo, commit_sha) needed by this shard:

    1. If the NPZ already contains ``commit_sha`` -> use cached.
    2. Else: ``git ls-tree`` at that commit, look up each .py blob in the cache.
       Embed any missing blob with the Qwen3 recipe (rare for train repos --
       the cache was built from the union of all commits in the manifest).
    3. Pool to 2048-d, append to NPZ for future runs.

After processing, every shard emits a single parquet shard with embeddings for
all (repo, commit) pairs in its assigned repos.

The script does NOT honor the ``_done`` marker (unlike the original embedder)
because we are explicitly extending pre-existing caches with new commits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---- Defaults / runtime config (overridable via CLI in main()) ----
# These are module-level so the embedding helpers can read them after main()
# resolves the chosen encoder + pooling. EMBED_DIM (per-file hidden) and
# REPO_DIM (per-snapshot vector) are derived from the model config + pooling.
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_NAME = DEFAULT_MODEL_NAME
POOLING = "maxmean"  # "maxmean" (Qwen3 recipe) | "lasttoken" (decoder embedder)
CHUNK_TOKENS = 2048
CHUNK_OVERLAP = 256
MAX_FILE_BYTES = 2_000_000
MIN_WINDOW_TOKENS = 8
EMBED_DIM = 1024  # per-file hidden (set from model config in main())
REPO_DIM = 2 * EMBED_DIM  # per-snapshot: 2*hidden (maxmean) or hidden (lasttoken)


def _repo_dim_for(hidden: int, pooling: str) -> int:
    """Per-snapshot embedding dim given per-file hidden size and pooling."""
    return hidden if pooling == "lasttoken" else 2 * hidden

REPOS_ROOTS = (
    "/scratch/lhotsko/REPO_DATASET/repositories",
    "/scratch/lhotsko/REPO_DATASET/repositories_ood",
)

DEFAULT_INPUT_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/commits"
DEFAULT_CACHE_ROOT = "/scratch/lhotsko/REPO_DATASET/static_commit/cache"
DEFAULT_OUT_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2_shards/repo_state"


def _resolve_repo_path(repo_id: str) -> Path | None:
    for root in REPOS_ROOTS:
        p = Path(root) / repo_id
        if p.is_dir() and (p / ".git").exists():
            return p
    return None


def _ls_tree_py(repo_path: Path, commit_sha: str
                ) -> List[Tuple[str, str, int]]:
    """Return [(blob_sha, fpath, size)] for tracked .py files <= MAX_FILE_BYTES."""
    res = subprocess.run(
        ["git", "ls-tree", "-r", "-l", commit_sha],
        cwd=repo_path, capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return []
    rows: List[Tuple[str, str, int]] = []
    for ln in res.stdout.splitlines():
        try:
            meta, fpath = ln.split("\t", 1)
            _mode, ftype, blob, size_str = meta.split()
        except ValueError:
            continue
        if ftype != "blob" or not fpath.endswith(".py"):
            continue
        try:
            size = int(size_str)
        except ValueError:
            continue
        if size <= 0 or size > MAX_FILE_BYTES:
            continue
        rows.append((blob, fpath, size))
    return rows


def _cat_file_blob(repo_path: Path, blob_sha: str) -> bytes:
    res = subprocess.run(
        ["git", "cat-file", "blob", blob_sha],
        cwd=repo_path, capture_output=True, check=False,
    )
    return res.stdout if res.returncode == 0 else b""


@torch.inference_mode()
def _embed_blob_text(text: str, tokenizer, model, device,
                     batch_size: int = 8) -> np.ndarray | None:
    """Tokenize -> chunk -> per-chunk pool -> avg chunk vecs -> [hidden] fp32.

    Per-chunk pooling follows ``POOLING``: masked-mean ('maxmean') or
    last-token + L2-norm ('lasttoken', decoder-embedder recipe)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return None
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    windows: List[List[int]] = []
    n = len(ids)
    for start in range(0, n, step):
        end = min(start + CHUNK_TOKENS, n)
        w = ids[start:end]
        if len(w) >= MIN_WINDOW_TOKENS:
            windows.append(w)
        if end >= n:
            break
    if not windows:
        return None
    chunk_vecs = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        decoded = [tokenizer.decode(w, skip_special_tokens=True) for w in batch]
        enc = tokenizer(decoded, padding=True, truncation=True,
                        max_length=CHUNK_TOKENS, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state
        if POOLING == "lasttoken":
            am = enc["attention_mask"]
            T = am.shape[1]
            ar = torch.arange(T, device=am.device)
            pos = torch.where(am.bool(), ar, torch.full_like(ar, -1))
            last_idx = pos.max(dim=1).values.clamp(min=0)
            vec = last[torch.arange(last.shape[0], device=last.device), last_idx]
            vec = F.normalize(vec.float(), dim=-1).detach().cpu()
            chunk_vecs.append(vec)
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
            denom = mask.sum(dim=1).clamp(min=1)
            mean = ((last * mask).sum(dim=1) / denom).detach().to(torch.float32).cpu()
            chunk_vecs.append(mean)
    file_vec = torch.cat(chunk_vecs, dim=0).mean(dim=0)
    if POOLING == "lasttoken":
        file_vec = F.normalize(file_vec, dim=-1)
    return file_vec.numpy()


def _read_blob_cache(cache_dir: Path) -> Tuple[Dict[str, int], np.ndarray | None]:
    blob_shas_file = cache_dir / "blob_shas.txt"
    blob_emb_file = cache_dir / "blob_embeddings.f16.npy"
    cached_idx: Dict[str, int] = {}
    cached_arr: np.ndarray | None = None
    if blob_shas_file.exists() and blob_emb_file.exists():
        cached_arr = np.load(blob_emb_file)
        with blob_shas_file.open() as f:
            for i, ln in enumerate(f):
                cached_idx[ln.strip()] = i
    return cached_idx, cached_arr


def _read_cached_snapshots(cache_dir: Path) -> Dict[str, np.ndarray]:
    snap_npz = cache_dir / "snapshot_embeddings.npz"
    if not snap_npz.exists():
        return {}
    try:
        data = np.load(snap_npz, allow_pickle=False)
        return {k: data[k].astype(np.float32) for k in data.files}
    except Exception:
        return {}


def _process_repo(
    repo_id: str,
    needed_shas: List[str],
    cache_root: Path,
    tokenizer, model, device,
    *,
    blob_batch: int = 8,
    log_every_blobs: int = 200,
) -> Dict[str, np.ndarray]:
    """Return {commit_sha: 2048-d fp32 vec} for every commit in needed_shas.
    Uses / extends the per-repo blob cache, and updates snapshot_embeddings.npz."""
    safe = repo_id.replace("/", "__")
    cache_dir = cache_root / safe
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_snaps = _read_cached_snapshots(cache_dir)
    missing_shas = [s for s in needed_shas if s not in cached_snaps]
    if not missing_shas:
        return {s: cached_snaps[s] for s in needed_shas if s in cached_snaps}

    repo_path = _resolve_repo_path(repo_id)
    if repo_path is None:
        print(f"  [skip] {repo_id}: clone not found on disk; "
              f"{len(missing_shas)} commits left unembedded.", flush=True)
        return {s: cached_snaps[s] for s in needed_shas if s in cached_snaps}

    # Walk each missing commit -> required blobs.
    per_snap_files: Dict[str, List[Tuple[str, str, int]]] = {}
    needed_blobs: Dict[str, int] = {}
    for sha in missing_shas:
        files = _ls_tree_py(repo_path, sha)
        per_snap_files[sha] = files
        for blob, _path, size in files:
            needed_blobs.setdefault(blob, size)

    if not needed_blobs:
        # All snapshots have zero usable files; emit zero vectors.
        for sha in missing_shas:
            cached_snaps[sha] = np.zeros(REPO_DIM, dtype=np.float32)
        # Update NPZ.
        _save_npz(cache_dir, cached_snaps)
        return {s: cached_snaps[s] for s in needed_shas if s in cached_snaps}

    # Embed any blob not already in cache.
    cached_idx, cached_arr = _read_blob_cache(cache_dir)
    todo_blobs = [b for b in needed_blobs if b not in cached_idx]
    n_todo = len(todo_blobs)
    n_uniq = len(needed_blobs)
    print(f"  [{repo_id}] new_commits={len(missing_shas)}, "
          f"unique_blobs={n_uniq}, blob_cache_hit={n_uniq - n_todo}, "
          f"blobs_to_embed={n_todo}", flush=True)

    new_vecs: List[np.ndarray] = []
    new_shas: List[str] = []
    if n_todo:
        t0 = time.time()
        for i, blob in enumerate(todo_blobs):
            try:
                text = _cat_file_blob(repo_path, blob).decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            vec = _embed_blob_text(text, tokenizer, model, device,
                                   batch_size=blob_batch) if text else None
            if vec is None:
                vec = np.zeros(EMBED_DIM, dtype=np.float32)
            new_vecs.append(vec.astype(np.float16))
            new_shas.append(blob)
            if (i + 1) % log_every_blobs == 0 or i + 1 == n_todo:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"    [{repo_id}] blob {i+1}/{n_todo} ({rate:.1f}/s)", flush=True)

        if cached_arr is not None and cached_arr.size > 0:
            all_arr = np.concatenate([cached_arr, np.stack(new_vecs)], axis=0)
            all_shas = list(cached_idx.keys()) + new_shas
        else:
            all_arr = np.stack(new_vecs)
            all_shas = new_shas
        np.save(cache_dir / "blob_embeddings.f16.npy", all_arr)
        (cache_dir / "blob_shas.txt").write_text(
            "\n".join(all_shas) + "\n", encoding="utf-8")
        sha_to_idx = {s: i for i, s in enumerate(all_shas)}
    else:
        all_arr = cached_arr
        sha_to_idx = cached_idx

    # Pool to REPO_DIM-d per missing snapshot.
    for sha, files in per_snap_files.items():
        idxs = [sha_to_idx[b] for b, _p, _sz in files if b in sha_to_idx]
        if not idxs:
            cached_snaps[sha] = np.zeros(REPO_DIM, dtype=np.float32)
            continue
        sub = all_arr[idxs].astype(np.float32)
        if POOLING == "lasttoken":
            # Mean of L2-normalized per-file vectors, renormalized -> [hidden].
            repo_vec = sub.mean(axis=0)
        else:
            # concat(mean_files, max_files) -> [2*hidden].
            repo_vec = np.concatenate([sub.mean(axis=0), sub.max(axis=0)], axis=0)
        norm = np.linalg.norm(repo_vec) + 1e-12
        repo_vec /= norm
        cached_snaps[sha] = repo_vec.astype(np.float32)

    # Update NPZ + meta.
    _save_npz(cache_dir, cached_snaps)
    return {s: cached_snaps[s] for s in needed_shas if s in cached_snaps}


def _save_npz(cache_dir: Path, snaps: Dict[str, np.ndarray]) -> None:
    """Atomic write of snapshot_embeddings.npz.
    Note: np.savez_compressed auto-appends ``.npz`` if the filename doesn't
    already end with it -- so the temp file MUST end with ``.npz`` to be
    written at the path we then os.replace from."""
    snap_npz = cache_dir / "snapshot_embeddings.npz"
    tmp = cache_dir / "snapshot_embeddings.tmp.npz"
    np.savez_compressed(tmp, **snaps)
    os.replace(tmp, snap_npz)


def _list_all_repos(input_dir: Path, splits: List[str]) -> Dict[str, str]:
    """Return {repo_id: split_it_appears_in}. Each repo is in exactly one split
    (cross-repo split is mutually exclusive)."""
    out: Dict[str, str] = {}
    for s in splits:
        path = input_dir / f"{s}.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["repo_id"], memory_map=True)
        for r in set(t.column("repo_id").to_pylist()):
            out.setdefault(r, s)
    return out


def main() -> None:
    # Declared global up front (before the argparser reads CHUNK_TOKENS /
    # CHUNK_OVERLAP as defaults) so we can rebind them from CLI below.
    global MODEL_NAME, POOLING, CHUNK_TOKENS, CHUNK_OVERLAP, EMBED_DIM, REPO_DIM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT,
                    help="Per-blob embedding cache root. Use a DISTINCT dir per "
                         "encoder (the cache stores model-specific vectors).")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--pooling", default="maxmean",
                    choices=["maxmean", "lasttoken"],
                    help="'maxmean' (Qwen3 recipe -> 2*hidden) or 'lasttoken' "
                         "(decoder-embedder recipe: last-token+L2 -> hidden).")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16", "auto"],
                    help="Model load dtype; use bfloat16/auto for large models.")
    ap.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--blob-batch", type=int, default=8,
                    help="Per-forward chunk batch size when embedding blobs. "
                         "Lower it for large models at long chunk lengths.")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "cr_val", "cr_test"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Resolve runtime config into module globals consumed by the embed helpers.
    MODEL_NAME = args.model_name
    POOLING = args.pooling
    CHUNK_TOKENS = args.chunk_tokens
    CHUNK_OVERLAP = args.chunk_overlap
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    EMBED_DIM = int(cfg.hidden_size)
    REPO_DIM = _repo_dim_for(EMBED_DIM, POOLING)
    print(f"Encoder={MODEL_NAME} pooling={POOLING} hidden={EMBED_DIM} "
          f"repo_dim={REPO_DIM}", flush=True)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    cache_root = Path(args.cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.splits:
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    # Determine shard repos.
    repo_to_split = _list_all_repos(input_dir, args.splits)
    sorted_repos = sorted(repo_to_split.keys())
    shard_repos = [r for i, r in enumerate(sorted_repos)
                   if i % args.shard_total == args.shard_index]
    print(f"Shard {args.shard_index+1}/{args.shard_total}: "
          f"{len(shard_repos)} repos / {len(sorted_repos)} total.", flush=True)

    # Group commits per repo per split.
    per_split_rows: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(dict)
    for s in args.splits:
        path = input_dir / f"{s}.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["repo_id", "commit_sha", "commit_index"],
                          memory_map=True)
        rid = t.column("repo_id").to_pylist()
        sha = t.column("commit_sha").to_pylist()
        idx = t.column("commit_index").to_pylist()
        for r, c, i in zip(rid, sha, idx):
            if r not in repo_to_split or repo_to_split[r] != s:
                continue
            per_split_rows[s].setdefault(r, []).append((c, int(i)))

    # Lazy-load the embedder (only spin up GPU if needed).
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    tokenizer = None
    model = None

    t_global = time.time()
    n_done = 0
    for split in args.splits:
        out_path = out_dir / split / f"shard_{args.shard_index:02d}_of_{args.shard_total:02d}.parquet"
        if out_path.exists():
            print(f"[skip] {out_path} already exists.", flush=True)
            continue

        repos_in_split = [r for r in shard_repos if r in per_split_rows.get(split, {})]
        print(f"[{split}] {len(repos_in_split)} repos in this shard.", flush=True)

        out_repos: List[str] = []
        out_shas: List[str] = []
        out_idxs: List[int] = []
        out_embs: List[List[float]] = []

        for ri, repo_id in enumerate(repos_in_split, 1):
            rows = per_split_rows[split][repo_id]
            needed = [sha for sha, _ in rows]
            # Lazy-load model only if we'll need to embed new blobs.
            cache_dir = cache_root / repo_id.replace("/", "__")
            cached_snaps = _read_cached_snapshots(cache_dir)
            missing = [s for s in needed if s not in cached_snaps]
            if missing and model is None:
                torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                               "float16": torch.float16, "auto": "auto"}[args.dtype]
                print(f"Loading {MODEL_NAME} on {device} (dtype={args.dtype}, "
                      f"first missing blob) ...", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModel.from_pretrained(
                    MODEL_NAME, torch_dtype=torch_dtype).to(device)
                model.eval()
            snap_map = _process_repo(
                repo_id, needed, cache_root, tokenizer, model, device,
                blob_batch=args.blob_batch,
            )
            for sha, idx in rows:
                emb = snap_map.get(sha)
                if emb is None:
                    emb = np.zeros(REPO_DIM, dtype=np.float32)
                out_repos.append(repo_id)
                out_shas.append(sha)
                out_idxs.append(idx)
                out_embs.append(emb.astype(np.float16).tolist())
                n_done += 1
            if ri % 5 == 0 or ri == len(repos_in_split):
                elapsed = (time.time() - t_global) / 60
                print(f"  [{split}] {ri}/{len(repos_in_split)} repos done, "
                      f"{n_done} commits embedded, {elapsed:.1f}m elapsed",
                      flush=True)

        # Write parquet.
        table = pa.table({
            "repo_id": pa.array(out_repos, type=pa.string()),
            "commit_sha": pa.array(out_shas, type=pa.string()),
            "commit_index": pa.array(out_idxs, type=pa.int32()),
            "cross_repo_split": pa.array([split] * len(out_repos), type=pa.string()),
            "repo_state_embedding": pa.array(
                out_embs,
                type=pa.list_(pa.float16(), int(REPO_DIM)),
            ),
        })
        tmp_path = out_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)
        print(f"[{split}] wrote {out_path} ({len(out_repos)} rows)", flush=True)

    print(f"\nDone. {n_done} commits embedded in {(time.time()-t_global)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
