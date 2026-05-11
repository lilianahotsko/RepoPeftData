#!/usr/bin/env python3
"""Build per-snapshot 2048-d Qwen3 embeddings for the static-commit dataset.

Step 2 of the ``Code2LoRA-direct, commit-aware`` pipeline.

Reads the manifest produced by ``build_static_commit_manifest.py`` and, for
each unique ``(repo_id, commit_sha)`` pair, produces a 2048-d L2-normalized
``concat(mean_pool, max_pool)`` repo embedding using the EXACT canonical
recipe from ``embed_repos/4_construct_embeddings.py``:

    model:  Qwen/Qwen3-Embedding-0.6B
    pooling per chunk: attention-mean of last_hidden_state (1024-d)
    chunking: 2048 tokens with 256 overlap, drop windows < 8 tokens
    pooling per file: mean of chunk embeddings
    pooling per repo: concat(mean_files, max_files) then L2-normalize -> 2048-d
    file selection: tracked .py blobs <= 2 MB

Key trick: work is keyed on blob SHAs, not (repo, commit) snapshots. Because
Qwen3 file embeddings depend only on file CONTENTS, identical blobs across
commits / paths share an embedding. ~10x speedup over naive re-embedding on
our 543-repo corpus (2.1 M file refs -> 208 k unique blobs).

Per-repo cache layout under ``$STATIC_COMMIT_DIR/cache/<author>__<repo>/``:

    blob_embeddings.f16.npy float16 [N_blobs, 1024], one row per blob_sha
    blob_shas.txt           one blob_sha per row, matches .npy row order
    snapshot_embeddings.npz keys = commit_sha -> float32 [2048]
    snapshot_meta.json      {commit_sha: {"n_files": int}}
    _done                   sentinel file written when this repo is finished

Top-level: ``$STATIC_COMMIT_DIR/snapshot_embeddings.json`` (flat map
``"<repo_id>@<commit_sha>" -> [2048]``). Built incrementally by the rank-0
post-merge pass; safe to run after sharded SLURM jobs all finish.

Resumability: per-repo ``_done`` marker is honored on rerun. Blob cache is
also incremental — already-embedded blobs are skipped.
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
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_TOKENS = 2048
CHUNK_OVERLAP = 256
MAX_FILE_BYTES = 2_000_000
MIN_WINDOW_TOKENS = 8
EMBED_DIM = 1024

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


def _ls_tree_py(repo_path: Path, commit_sha: str,
                max_bytes: int = MAX_FILE_BYTES
                ) -> List[Tuple[str, str, int]]:
    """Return [(blob_sha, fpath, size)] for tracked .py files <= max_bytes."""
    out = subprocess.run(
        ["git", "ls-tree", "-r", "-l", commit_sha],
        cwd=repo_path, capture_output=True, text=True, check=True
    ).stdout
    rows: List[Tuple[str, str, int]] = []
    for ln in out.splitlines():
        try:
            meta, fpath = ln.split("\t", 1)
            _mode, ftype, blob, size_str = meta.split()
        except ValueError:
            continue
        if ftype != "blob":
            continue
        if not fpath.endswith(".py"):
            continue
        try:
            size = int(size_str)
        except ValueError:
            continue
        if size <= 0 or size > max_bytes:
            continue
        rows.append((blob, fpath, size))
    return rows


def _cat_file_blob(repo_path: Path, blob_sha: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", blob_sha],
        cwd=repo_path, capture_output=True, check=True
    ).stdout


@torch.inference_mode()
def _embed_blob_text(text: str, tokenizer, model, device,
                     batch_size: int = 8) -> np.ndarray | None:
    """Canonical recipe: tokenize -> chunk -> attention-mean -> avg chunk vecs.
    Returns float32 [1024] or None if no usable chunks."""
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
    chunk_vecs: List[torch.Tensor] = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        decoded = [tokenizer.decode(w, skip_special_tokens=True) for w in batch]
        enc = tokenizer(decoded, padding=True, truncation=True,
                        max_length=CHUNK_TOKENS, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        mean = ((last * mask).sum(dim=1) / denom).detach().to(torch.float32).cpu()
        chunk_vecs.append(mean)
    return torch.cat(chunk_vecs, dim=0).mean(dim=0).numpy()


def _read_manifest(path: Path) -> Dict[str, List[Tuple[str, int]]]:
    per_repo: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    seen = set()
    with path.open() as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            repo_id, commit_sha = cols[0], cols[1]
            commit_index = int(cols[2])
            key = (repo_id, commit_sha)
            if key in seen:
                continue
            seen.add(key)
            per_repo[repo_id].append((commit_sha, commit_index))
    for repo_id in per_repo:
        per_repo[repo_id].sort(key=lambda x: x[1])
    return per_repo


def process_repo(
    repo_id: str,
    snapshots: List[Tuple[str, int]],
    cache_root: Path,
    tokenizer,
    model,
    device,
    *,
    log_every_blobs: int = 200,
    force: bool = False,
) -> Dict[str, np.ndarray] | None:
    repo_path = _resolve_repo_path(repo_id)
    if repo_path is None:
        print(f"[skip] {repo_id}: not on disk", flush=True)
        return None

    safe = repo_id.replace("/", "__")
    cache_dir = cache_root / safe
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_marker = cache_dir / "_done"
    snap_npz = cache_dir / "snapshot_embeddings.npz"

    if (not force) and done_marker.exists() and snap_npz.exists():
        try:
            data = np.load(snap_npz, allow_pickle=False)
            return {k: data[k].astype(np.float32) for k in data.files}
        except Exception:
            pass

    per_snap_files: Dict[str, List[Tuple[str, str, int]]] = {}
    needed_blobs: Dict[str, int] = {}
    for sha, _ in snapshots:
        try:
            files = _ls_tree_py(repo_path, sha)
        except subprocess.CalledProcessError as e:
            print(f"  [skip-snap] {repo_id}@{sha[:8]}: ls-tree failed: {e}", flush=True)
            continue
        per_snap_files[sha] = files
        for blob, _path, size in files:
            needed_blobs.setdefault(blob, size)

    if not needed_blobs:
        print(f"[skip] {repo_id}: no .py blobs", flush=True)
        return None

    blob_shas_file = cache_dir / "blob_shas.txt"
    blob_emb_file = cache_dir / "blob_embeddings.f16.npy"
    cached_idx: Dict[str, int] = {}
    cached_arr: np.ndarray | None = None
    if blob_shas_file.exists() and blob_emb_file.exists():
        cached_arr = np.load(blob_emb_file)
        with blob_shas_file.open() as f:
            for i, ln in enumerate(f):
                cached_idx[ln.strip()] = i

    todo = [b for b in needed_blobs if b not in cached_idx]
    n_todo = len(todo)
    print(f"[{repo_id}] snapshots={len(snapshots)}, file_refs={sum(len(v) for v in per_snap_files.values())}, "
          f"unique_blobs={len(needed_blobs)}, cached={len(cached_idx)}, to_embed={n_todo}",
          flush=True)

    new_vecs: List[np.ndarray] = []
    new_shas: List[str] = []
    t0 = time.time()
    for i, blob in enumerate(todo):
        try:
            text = _cat_file_blob(repo_path, blob).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError:
            text = ""
        vec = _embed_blob_text(text, tokenizer, model, device) if text else None
        if vec is None:
            vec = np.zeros((EMBED_DIM,), dtype=np.float32)
        new_vecs.append(vec.astype(np.float16))
        new_shas.append(blob)
        if (i + 1) % log_every_blobs == 0 or i + 1 == n_todo:
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            print(f"  [{repo_id}] {i+1}/{n_todo} blobs ({rate:.1f}/s)", flush=True)

    if cached_arr is not None and cached_arr.size > 0:
        if new_vecs:
            all_arr = np.concatenate([cached_arr, np.stack(new_vecs)], axis=0)
        else:
            all_arr = cached_arr
        all_shas = list(cached_idx.keys()) + new_shas
    else:
        all_arr = np.stack(new_vecs) if new_vecs else np.zeros((0, EMBED_DIM), np.float16)
        all_shas = new_shas
    np.save(blob_emb_file, all_arr)
    blob_shas_file.write_text("\n".join(all_shas) + "\n", encoding="utf-8")
    sha_to_idx: Dict[str, int] = {s: i for i, s in enumerate(all_shas)}

    snap_to_emb: Dict[str, np.ndarray] = {}
    snap_meta: Dict[str, Dict[str, int]] = {}
    for sha, files in per_snap_files.items():
        idxs = [sha_to_idx[blob] for blob, _p, _sz in files if blob in sha_to_idx]
        if not idxs:
            snap_meta[sha] = {"n_files": 0}
            continue
        sub = all_arr[idxs].astype(np.float32)
        mean_pool = sub.mean(axis=0)
        max_pool = sub.max(axis=0)
        repo_vec = np.concatenate([mean_pool, max_pool], axis=0)
        norm = np.linalg.norm(repo_vec) + 1e-12
        repo_vec /= norm
        snap_to_emb[sha] = repo_vec.astype(np.float32)
        snap_meta[sha] = {"n_files": len(idxs)}

    np.savez_compressed(snap_npz, **snap_to_emb)
    (cache_dir / "snapshot_meta.json").write_text(
        json.dumps(snap_meta, indent=2), encoding="utf-8")
    done_marker.write_text("ok\n", encoding="utf-8")

    return snap_to_emb


def merge_flat_json(cache_root: Path, out_path: Path) -> int:
    """Walk per-repo caches and merge their snapshot_embeddings.npz into a flat
    JSON ``{<repo_id>@<commit_sha>: [2048]}``. Used as the final aggregation
    pass after all sharded SLURM jobs finish."""
    flat: Dict[str, List[float]] = {}
    for repo_cache in sorted(cache_root.iterdir()):
        if not repo_cache.is_dir():
            continue
        snap_npz = repo_cache / "snapshot_embeddings.npz"
        if not snap_npz.exists():
            continue
        try:
            data = np.load(snap_npz, allow_pickle=False)
        except Exception as e:
            print(f"  [warn] {repo_cache.name}: {e}", flush=True)
            continue
        repo_id = repo_cache.name.replace("__", "/", 1)
        for sha in data.files:
            flat[f"{repo_id}@{sha}"] = data[sha].astype(np.float32).tolist()
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(flat), encoding="utf-8")
    tmp.rename(out_path)
    return len(flat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--out-dir",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-repos", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="Skip first N repos (sharded SLURM).")
    ap.add_argument("--stop", type=int, default=0, help="Stop after this many repos; 0 = no cap.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip embedding; just walk caches and rebuild snapshot_embeddings.json.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    flat_path = out_dir / "snapshot_embeddings.json"

    if args.merge_only:
        n = merge_flat_json(cache_root, flat_path)
        print(f"Merged {n:,} snapshot embeddings -> {flat_path}", flush=True)
        return

    per_repo = _read_manifest(Path(args.manifest))
    repo_ids = sorted(per_repo)
    if args.start:
        repo_ids = repo_ids[args.start:]
    if args.stop:
        repo_ids = repo_ids[: args.stop]
    if args.limit_repos:
        repo_ids = repo_ids[: args.limit_repos]
    print(f"Processing {len(repo_ids)} repos -> {cache_root}", flush=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    t_global = time.time()
    for i, repo_id in enumerate(repo_ids, 1):
        try:
            process_repo(repo_id, per_repo[repo_id], cache_root,
                         tokenizer, model, device)
        except Exception as e:
            print(f"  [error] {repo_id}: {type(e).__name__}: {e}", flush=True)
            continue
        if i % 10 == 0 or i == len(repo_ids):
            elapsed = time.time() - t_global
            eta = elapsed / i * (len(repo_ids) - i)
            print(f"[{i}/{len(repo_ids)} repos | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m]",
                  flush=True)

    n = merge_flat_json(cache_root, flat_path)
    print(f"\nDone. Merged {n:,} snapshot embeddings -> {flat_path}", flush=True)


if __name__ == "__main__":
    main()
