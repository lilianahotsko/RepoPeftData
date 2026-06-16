#!/usr/bin/env python3
"""Dump the RAG retrieval result (top-k chunks + similarity scores) for
every QnA in a V1 split, so the V1 viewer (``visualize_v1_qnas.py``) can
display the actual context that was injected at eval time.

Output layout
-------------

One JSON file per repo at ``<output-dir>/<repo_slug>.json`` with the
shape::

    {
        "repo": "<author/name>",
        "split": "<ir_test|cr_test|ood_test>",
        "cache_dir": "<RAG chunk cache used>",
        "top_k": <int>,
        "embed_model_name": "<embedding model>",
        "retrievals": [
            { "row_within_repo": <int>, "global_row_idx": <int>,
              "top_k_chunks": ["chunk text", ...],
              "scores": [0.412, 0.397, ...] },
            ...
        ]
    }

The ordering inside ``retrievals`` mirrors the V1 evaluator: the same
``evaluation.data_utils.load_split`` order, filtered by the
"target-startswith-comma" rule. ``global_row_idx`` is the absolute index
across all repos in the split, so the V1 viewer can join on either.

This is the same retrieval pipeline as ``baselines/rag/test_rag.py``:
embed ``prefix[-2000:]`` with the configured embedding model, do
``query @ chunk_emb.T`` cosine similarity, take ``top_k``.

Usage (GPU recommended, but CPU works for small ``--max-rows`` test runs)::

    python scripts/dump_v1_rag_context.py \\
        --split ir_test \\
        --cache-dir $SCRATCH/RAG_CHUNK_CACHE_256 \\
        --output-dir $SCRATCH/V1_RAG_CONTEXT_DUMP/ir_test_k10

For a full IR-test (5,222 QnAs) dump on an H100, expect ~3-5 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from evaluation.data_utils import load_split  # noqa: E402


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def load_cached_index(cache_dir: Path, repo_id: str) -> Dict:
    cache_path = cache_dir / f"{_slug(repo_id)}.pt"
    if not cache_path.exists():
        return {"chunks": [], "embeddings": None}
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return {
        "chunks": data.get("chunks", []),
        "embeddings": (data["embeddings"].float()
                       if data.get("embeddings") is not None else None),
    }


@torch.inference_mode()
def embed_query(text: str, model, tokenizer, device: str,
                max_length: int = 512) -> torch.Tensor:
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out.last_hidden_state * mask).sum(dim=1) \
        / mask.sum(dim=1).clamp(min=1)
    return F.normalize(mean.cpu(), p=2, dim=-1)


def retrieve(query_emb: torch.Tensor, index: Dict, top_k: int
             ) -> List[tuple]:
    chunks = index["chunks"]
    emb = index["embeddings"]
    if not chunks or emb is None:
        return []
    if query_emb.dtype != emb.dtype:
        query_emb = query_emb.to(emb.dtype)
    sims = (query_emb @ emb.T).squeeze(0)
    k = min(top_k, len(chunks))
    scores, idxs = sims.topk(k)
    return [(chunks[idxs[i].item()], float(scores[i].item()))
            for i in range(k)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="ir_test",
                   choices=["ir_test", "cr_test", "ood_test"])
    p.add_argument("--splits-dir", type=Path,
                   default=Path(os.environ.get("SCRATCH",
                                               os.path.expanduser("~/scratch"))) /
                           "REPO_DATASET")
    p.add_argument("--cache-dir", type=Path,
                   default=Path(os.environ.get("SCRATCH",
                                               os.path.expanduser("~/scratch"))) /
                           "RAG_CHUNK_CACHE_256",
                   help="Pre-built RAG chunk index dir (one .pt per repo).")
    p.add_argument("--embed-model-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many chunks to keep per QnA. Use the max of all "
                        "RAG variants you may want to inspect (e.g. 10 covers "
                        "k=3, k=5, k=10).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write <repo_slug>.json files.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="If >0, only retrieve for the first N global rows "
                        "(useful for smoke tests).")
    p.add_argument("--limit-repos", type=int, default=None)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                        else "cpu"))
    p.add_argument("--batch-size", type=int, default=64,
                   help="Number of prefixes embedded per forward pass.")
    args = p.parse_args()

    print(f"[v1-rag-dump] split={args.split}  device={args.device}  "
          f"cache_dir={args.cache_dir}  out={args.output_dir}", flush=True)

    items = load_split(args.splits_dir, args.split,
                       limit_repos=args.limit_repos)
    if not items:
        print(f"[error] empty {args.split}.json at {args.splits_dir}",
              flush=True)
        return 1
    if args.max_rows and args.max_rows > 0:
        items = items[:args.max_rows]
    print(f"  loaded {len(items)} items across "
          f"{len({it['repo'] for it in items})} repos", flush=True)

    repo_names = sorted({it["repo"] for it in items})
    print(f"  loading chunk indices for {len(repo_names)} repos ...",
          flush=True)
    repo_indices: Dict[str, Dict] = {}
    n_with_chunks = 0
    for rn in repo_names:
        ix = load_cached_index(args.cache_dir, rn)
        repo_indices[rn] = ix
        if ix["chunks"]:
            n_with_chunks += 1
    print(f"  {n_with_chunks}/{len(repo_names)} repos have non-empty chunk "
          f"caches", flush=True)

    from transformers import AutoModel, AutoTokenizer
    print(f"  loading embedding model {args.embed_model_name} ...",
          flush=True)
    embed_tok = AutoTokenizer.from_pretrained(args.embed_model_name,
                                              use_fast=True)
    embed_model = AutoModel.from_pretrained(args.embed_model_name,
                                            torch_dtype=torch.bfloat16)
    embed_model.to(args.device).eval()

    # Per-repo (in-repo row idx, global row idx).
    per_repo_rows: Dict[str, List[tuple]] = defaultdict(list)
    seen_per_repo: Dict[str, int] = defaultdict(int)
    for gidx, it in enumerate(items):
        rid = it["repo"]
        per_repo_rows[rid].append((seen_per_repo[rid], gidx))
        seen_per_repo[rid] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for rid in repo_names:
        rows = per_repo_rows[rid]
        if not rows:
            continue
        ix = repo_indices[rid]
        if not ix["chunks"]:
            doc = {
                "repo": rid, "split": args.split,
                "cache_dir": str(args.cache_dir),
                "top_k": args.top_k,
                "embed_model_name": args.embed_model_name,
                "warning": "no cached chunks for this repo",
                "retrievals": [
                    {"row_within_repo": r, "global_row_idx": g,
                     "top_k_chunks": [], "scores": []}
                    for (r, g) in rows
                ],
            }
            out_path = args.output_dir / f"{_slug(rid)}.json"
            out_path.write_text(json.dumps(doc), encoding="utf-8")
            n_done += len(rows)
            continue

        # Batch-embed all this repo's prefixes.
        prefixes = [items[g]["prefix"][-2000:] for (_, g) in rows]
        retrievals: List[Dict] = []
        for i in range(0, len(prefixes), args.batch_size):
            chunk_prefixes = prefixes[i:i + args.batch_size]
            with torch.inference_mode():
                enc = embed_tok(chunk_prefixes, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
                enc = {k: v.to(args.device) for k, v in enc.items()}
                out = embed_model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1)
                mean = (out.last_hidden_state * mask).sum(dim=1) \
                    / mask.sum(dim=1).clamp(min=1)
                q = F.normalize(mean.cpu(), p=2, dim=-1).float()
            chunk_emb = ix["embeddings"]
            sims = q @ chunk_emb.T   # [bs, n_chunks]
            k = min(args.top_k, sims.shape[1])
            scores, idxs = sims.topk(k, dim=1)
            for j in range(len(chunk_prefixes)):
                r, g = rows[i + j]
                top_chunks = [ix["chunks"][int(t)] for t in idxs[j].tolist()]
                top_scores = [float(s) for s in scores[j].tolist()]
                retrievals.append({
                    "row_within_repo": r,
                    "global_row_idx": g,
                    "top_k_chunks": top_chunks,
                    "scores": top_scores,
                })

        doc = {
            "repo": rid, "split": args.split,
            "cache_dir": str(args.cache_dir),
            "top_k": args.top_k,
            "embed_model_name": args.embed_model_name,
            "retrievals": retrievals,
        }
        out_path = args.output_dir / f"{_slug(rid)}.json"
        out_path.write_text(json.dumps(doc), encoding="utf-8")
        n_done += len(rows)
        elapsed = time.time() - t0
        if n_done % 500 < args.batch_size or n_done == len(items):
            rate = n_done / max(elapsed, 1e-6)
            print(f"  [{n_done}/{len(items)}]  {rate:.1f} qna/s  "
                  f"({elapsed/60:.1f} min)", flush=True)

    print(f"[done] wrote {len(repo_names)} per-repo JSONs to {args.output_dir} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
