#!/usr/bin/env python3
"""
Pre-build and cache chunk embedding indices for RAG retrieval.
Run this ONCE, then test_rag.py loads the cached indices instantly.

Saves per-repo: {chunks: list[str], embeddings: Tensor[N, D]} as .pt files.

Usage:
    python baselines/rag/build_indices.py
    python baselines/rag/build_indices.py --chunk-tokens 512 --max-chunks 300
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@torch.inference_mode()
def embed_texts(model, tokenizer, texts, device, batch_size=32, max_length=512):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        mean = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_vecs.append(mean.cpu())
    return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0)


def build_repo_index(repo_dir: Path, embed_model, embed_tokenizer, device: str,
                     chunk_tokens: int = 512, max_chunks: int = 300) -> dict:
    from create_dataset.embed_repos import (
        iter_source_files, read_text_file, chunk_token_ids,
        DEFAULT_EXTS, MAX_FILE_BYTES_DEFAULT,
    )

    files = iter_source_files(repo_dir, DEFAULT_EXTS, max_files_per_repo=500, max_file_bytes=MAX_FILE_BYTES_DEFAULT)

    all_chunks = []
    for fp in files:
        text = read_text_file(fp)
        if not text:
            continue
        rel = str(fp.relative_to(repo_dir))
        ids = embed_tokenizer.encode(text, add_special_tokens=False)
        windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=64)
        for w in windows:
            chunk_text = embed_tokenizer.decode(w, skip_special_tokens=True)
            all_chunks.append(f"# File: {rel}\n{chunk_text}")
            if len(all_chunks) >= max_chunks:
                break
        if len(all_chunks) >= max_chunks:
            break

    if not all_chunks:
        return {"chunks": [], "embeddings": None}

    chunk_embs = embed_texts(embed_model, embed_tokenizer, all_chunks, device,
                             batch_size=32, max_length=chunk_tokens)
    chunk_embs = F.normalize(chunk_embs, p=2, dim=-1)

    return {"chunks": all_chunks, "embeddings": chunk_embs}


def main():
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )
    default_splits = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_cache = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "RAG_CHUNK_CACHE",
    )

    ap = argparse.ArgumentParser(description="Pre-build RAG chunk indices")
    ap.add_argument("--repos-root", type=str, default=default_repos)
    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--cache-dir", type=str, default=default_cache)
    ap.add_argument("--embed-model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    repos_root = Path(args.repos_root).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all repo names from all splits
    all_repos = set()
    for split_name in ["train", "cr_val", "cr_test", "ir_val", "ir_test"]:
        path = splits_dir / f"{split_name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            all_repos.update(data.get("repositories", {}).keys())

    print(f"Found {len(all_repos)} repos across all splits")

    print(f"Loading embedding model: {args.embed_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.embed_model_name, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()

    built = 0
    skipped = 0
    for repo_name in tqdm(sorted(all_repos), desc="Building indices"):
        safe_name = repo_name.replace("/", "__")
        cache_path = cache_dir / f"{safe_name}.pt"

        if cache_path.exists():
            skipped += 1
            continue

        author, rname = repo_name.split("/", 1)
        repo_dir = repos_root / author / rname
        if not repo_dir.exists():
            continue

        index = build_repo_index(repo_dir, model, tokenizer, args.device,
                                 chunk_tokens=args.chunk_tokens, max_chunks=args.max_chunks)

        torch.save({
            "chunks": index["chunks"],
            "embeddings": index["embeddings"],
            "repo": repo_name,
            "chunk_tokens": args.chunk_tokens,
            "max_chunks": args.max_chunks,
        }, cache_path)
        built += 1

    print(f"\nDone. Built: {built}, Skipped (cached): {skipped}")
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
