#!/usr/bin/env python3
"""Build per-``(repo_id, commit_sha)`` RAG chunk indices for the v2
commit-derived QnA dataset.

For every unique commit referenced by the v2 QnA parquets we read the
**state of the repository at that commit** via ``git ls-tree`` and
``git show`` (no checkout / no worktree -- avoids polluting the clones).
Each ``.py`` non-test file is tokenized with Qwen3-Embedding-0.6B,
chunked into 512-token windows with 64-token overlap (matching
``baselines/rag/build_indices.py``), embedded, L2-normalized, and saved
to ``$RAG_COMMIT_CACHE_DIR/<author>__<repo>__<sha>.pt``.

This is the per-commit analogue of ``baselines/rag/build_indices.py``,
which only builds one index per repo at HEAD.

Resumable, shardable, and safe to rerun -- the per-commit cache file is
written atomically via ``tmp + rename``.

Usage::

    python evaluation/build_rag_cache_per_commit.py \
        --qna-dir /scratch/lhotsko/REPO_DATASET/commit_parquet_hf/qna \
        --suites cr_val cr_test ir_val ir_test \
        --repos-root /scratch/lhotsko/REPO_DATASET/repositories \
        --out-dir /scratch/lhotsko/RAG_CHUNK_CACHE_COMMITS \
        --num-shards 4 --shard-i 0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Filters for non-test source files (mirrors create_dataset/embed_repos.py).
# ---------------------------------------------------------------------------

_TEST_FILE_RE = re.compile(
    r"(^|/)(test_[^/]*\.py$|[^/]*_test\.py$|tests?/)"
)
_SKIP_DIRS = re.compile(
    r"(^|/)("
    r"\.git/|__pycache__/|node_modules/|\.tox/|\.eggs/|\.mypy_cache/|"
    r"\.pytest_cache/|venv/|\.venv/|env/|\.nox/|\.hg/|\.svn/|site-packages/|"
    r"TEST_HYPERNET/"
    r")"
)


def _is_kept_py_file(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    if _SKIP_DIRS.search("/" + path):
        return False
    if _TEST_FILE_RE.search("/" + path):
        return False
    return True


# ---------------------------------------------------------------------------
# Git helpers (no checkout / no worktree)
# ---------------------------------------------------------------------------

def _git_ls_tree(repo_dir: Path, sha: str) -> List[str]:
    """Return all blob paths at *sha* (recursive, name-only)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", sha],
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return out.decode("utf-8", errors="ignore").splitlines()


def _git_show_blob(repo_dir: Path, sha: str, path: str,
                   max_bytes: int = 2_000_000) -> Optional[str]:
    """Return file contents at ``<sha>:<path>``. Returns None on error or
    if the blob is larger than *max_bytes*."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "show", f"{sha}:{path}"],
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if len(out) > max_bytes:
        return None
    try:
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chunking (lifted from baselines/rag/build_indices.py: 512/64).
# ---------------------------------------------------------------------------

def _chunk_token_ids(token_ids: List[int], chunk_tokens: int,
                     overlap: int) -> List[List[int]]:
    if chunk_tokens <= 0 or overlap >= chunk_tokens:
        return [token_ids[:chunk_tokens]] if token_ids else []
    step = chunk_tokens - overlap
    n = len(token_ids)
    out: List[List[int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        out.append(token_ids[start:end])
        if end == n:
            break
        start += step
    return out


# ---------------------------------------------------------------------------
# Embedding loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _embed_texts(model, tokenizer, texts: List[str], device: str,
                 batch_size: int = 32, max_length: int = 512) -> torch.Tensor:
    """Attention-mean pooled embeddings, returned on CPU as float32."""
    if not texts:
        return torch.empty(0)
    out_chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out_chunks.append(pooled.float().cpu())
    return torch.cat(out_chunks, dim=0)


def _build_one_commit_index(
    *,
    repo_dir: Path,
    repo_id: str,
    sha: str,
    embed_model,
    embed_tokenizer,
    device: str,
    chunk_tokens: int,
    overlap: int,
    max_chunks: int,
    batch_size: int,
) -> Dict[str, object]:
    """Walk the tree at *sha*, chunk and embed non-test ``.py`` files."""
    paths = _git_ls_tree(repo_dir, sha)
    py_paths = [p for p in paths if _is_kept_py_file(p)]
    py_paths.sort()  # deterministic order

    all_chunks: List[str] = []
    for rel in py_paths:
        text = _git_show_blob(repo_dir, sha, rel)
        if not text:
            continue
        ids = embed_tokenizer.encode(text, add_special_tokens=False)
        windows = _chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=overlap)
        for w in windows:
            if not w:
                continue
            chunk_text = embed_tokenizer.decode(w, skip_special_tokens=True)
            all_chunks.append(f"# File: {rel}\n{chunk_text}")
            if len(all_chunks) >= max_chunks:
                break
        if len(all_chunks) >= max_chunks:
            break

    if not all_chunks:
        return {
            "chunks": [],
            "embeddings": None,
            "repo": repo_id,
            "commit_sha": sha,
            "chunk_tokens": chunk_tokens,
            "overlap": overlap,
            "max_chunks": max_chunks,
            "n_py_files_in_tree": len(py_paths),
        }

    embs = _embed_texts(
        embed_model, embed_tokenizer, all_chunks, device=device,
        batch_size=batch_size, max_length=chunk_tokens,
    )
    embs = F.normalize(embs, p=2, dim=-1).to(torch.float16)
    return {
        "chunks": all_chunks,
        "embeddings": embs,
        "repo": repo_id,
        "commit_sha": sha,
        "chunk_tokens": chunk_tokens,
        "overlap": overlap,
        "max_chunks": max_chunks,
        "n_py_files_in_tree": len(py_paths),
    }


# ---------------------------------------------------------------------------
# Suite enumeration
# ---------------------------------------------------------------------------

def _enumerate_commit_keys(qna_dir: Path, suites: List[str]) -> List[Tuple[str, str]]:
    """Collect the union of ``(repo_id, commit_sha)`` keys across suites.

    Loads only ``repo_id`` and ``commit_sha`` columns -- this avoids the
    multi-gigabyte ``prefix`` / ``assertion_anchor`` columns and keeps
    memory under a few hundred MB even when scanning all four suites.

    Prefers a standalone ``<suite>.parquet`` (the default in
    ``code2lora_snapshots_hf/qna/``); falls back to filtering
    ``train.parquet`` on ``in_repo_split`` for ``ir_*`` suites when only
    ``commit_parquet_hf/qna/`` is available.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    fallback_in_split = {"ir_test": "test", "ir_val": "val"}
    keys: Set[Tuple[str, str]] = set()
    for s in suites:
        direct = qna_dir / f"{s}.parquet"
        if direct.exists():
            tbl = pq.read_table(str(direct), columns=["repo_id", "commit_sha"])
        elif s in fallback_in_split:
            train = qna_dir / "train.parquet"
            if not train.exists():
                print(f"[enumerate] WARNING: missing {direct} (and "
                      f"{train} fallback)", flush=True)
                continue
            tbl = pq.read_table(
                str(train), columns=["repo_id", "commit_sha", "in_repo_split"],
            )
            tbl = tbl.filter(pc.equal(tbl["in_repo_split"], fallback_in_split[s]))
        else:
            print(f"[enumerate] WARNING: missing {direct}", flush=True)
            continue
        repos = tbl["repo_id"].to_pylist()
        shas = tbl["commit_sha"].to_pylist()
        for r, sh in zip(repos, shas):
            if r and sh:
                keys.add((r, sh))
        print(f"[enumerate] suite={s:8s} -> {tbl.num_rows:,} qnas, "
              f"running unique-key total: {len(keys):,}", flush=True)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--qna-dir",
        default=f"{scratch}/REPO_DATASET/code2lora_snapshots_hf/qna",
        help="Directory containing v2 commit-derived qna parquets. "
             "Standalone <suite>.parquet files are preferred; for ir_* the "
             "builder falls back to filtering train.parquet on in_repo_split.",
    )
    ap.add_argument(
        "--suites", nargs="+",
        default=["cr_val", "cr_test", "ir_val", "ir_test"],
        choices=["cr_val", "cr_test", "ir_val", "ir_test"],
    )
    ap.add_argument(
        "--repos-root",
        default=f"{scratch}/REPO_DATASET/repositories",
        help="Root that contains <author>/<repo>/.git clones.",
    )
    ap.add_argument(
        "--out-dir",
        default=f"{scratch}/RAG_CHUNK_CACHE_COMMITS",
        help="Where to write per-(repo, commit) .pt indices.",
    )
    ap.add_argument("--embed-model-name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--max-chunks", type=int, default=300,
                    help="Upper bound on chunks per (repo, commit) -- matches "
                         "baselines/rag/build_indices.py.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only process the first N (repo, sha) keys "
                         "after sharding.")
    ap.add_argument("--force", action="store_true",
                    help="Re-embed even if the cache file already exists.")
    args = ap.parse_args()

    qna_dir = Path(args.qna_dir).expanduser().resolve()
    repos_root = Path(args.repos_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[args] qna_dir   = {qna_dir}")
    print(f"[args] repos_root = {repos_root}")
    print(f"[args] out_dir   = {out_dir}")
    print(f"[args] suites    = {args.suites}")
    print(f"[args] shard     = {args.shard_i + 1}/{args.num_shards}")
    print(f"[args] embed     = {args.embed_model_name}", flush=True)

    keys = _enumerate_commit_keys(qna_dir, args.suites)
    print(f"\n[enumerate] total unique (repo, commit) keys: {len(keys):,}",
          flush=True)
    if args.num_shards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_i]
        print(f"[enumerate] after shard {args.shard_i + 1}/{args.num_shards}: "
              f"{len(keys):,} keys", flush=True)
    if args.limit:
        keys = keys[: args.limit]
        print(f"[enumerate] --limit -> {len(keys)} keys", flush=True)

    # Filter to keys whose repo clone exists.
    repo_to_dir: Dict[str, Path] = {}
    missing_repo: Set[str] = set()
    keys_valid: List[Tuple[str, str]] = []
    for repo_id, sha in keys:
        if "/" not in repo_id:
            continue
        rd = repos_root / repo_id
        if not rd.exists():
            missing_repo.add(repo_id)
            continue
        repo_to_dir[repo_id] = rd
        keys_valid.append((repo_id, sha))
    if missing_repo:
        print(f"[enumerate] skipping {len(missing_repo)} keys with missing repo clones: "
              f"{sorted(missing_repo)[:5]}...", flush=True)

    # Drop already-cached keys (unless --force).
    todo: List[Tuple[str, str]] = []
    skipped_cached = 0
    for repo_id, sha in keys_valid:
        safe = repo_id.replace("/", "__")
        cache_path = out_dir / f"{safe}__{sha}.pt"
        if cache_path.exists() and not args.force:
            skipped_cached += 1
            continue
        todo.append((repo_id, sha))
    print(f"[enumerate] {len(todo):,} keys to process "
          f"({skipped_cached:,} already cached)\n", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    from transformers import AutoModel, AutoTokenizer
    print(f"[load] embedder {args.embed_model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        args.embed_model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    t0 = time.time()
    n_done = 0
    n_empty = 0
    for repo_id, sha in todo:
        safe = repo_id.replace("/", "__")
        cache_path = out_dir / f"{safe}__{sha}.pt"
        try:
            index = _build_one_commit_index(
                repo_dir=repo_to_dir[repo_id],
                repo_id=repo_id, sha=sha,
                embed_model=model, embed_tokenizer=tokenizer, device=device,
                chunk_tokens=args.chunk_tokens, overlap=args.overlap,
                max_chunks=args.max_chunks, batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"  [error] {repo_id} @ {sha[:8]}: {type(e).__name__}: {e}",
                  flush=True)
            n_done += 1
            continue
        # Atomic write: tmp + rename.
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(index, tmp)
        os.replace(tmp, cache_path)
        if not index["chunks"]:
            n_empty += 1
        n_done += 1
        if n_done % 25 == 0 or n_done == len(todo):
            elapsed = (time.time() - t0) / 60.0
            rate = n_done / max(elapsed, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{len(todo)}] last={repo_id}@{sha[:8]} "
                  f"chunks={len(index['chunks'])} files_in_tree={index['n_py_files_in_tree']} "
                  f"empty_so_far={n_empty} "
                  f"elapsed={elapsed:.1f}m ETA={eta:.1f}m", flush=True)

    print(f"\nDone. Built {n_done} indices, {n_empty} of which had no chunks. "
          f"Cache dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
