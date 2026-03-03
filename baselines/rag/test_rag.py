#!/usr/bin/env python3
"""
RAG baseline: retrieve top-k code chunks from the repo, prepend as context,
then use Qwen2.5-Coder to complete assertions.

Retrieval uses Qwen3-Embedding-0.6B (same as hypernetwork repo embeddings).
Chunks are retrieved per test-prefix query using cosine similarity.

Usage:
    python baselines/rag/test_rag.py --split cr_test --top-k 5
    python baselines/rag/test_rag.py --split cr_test --top-k 3 --limit 100
    python baselines/rag/test_rag.py --split ir_test --top-k 10 --limit-repos 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments
from evaluation.data_utils import get_default_splits_dir, load_split, get_bos_id


def build_repo_chunk_index(repos_root: Path, repo_name: str, embed_model, embed_tokenizer,
                           device: str, chunk_tokens: int = 512, max_chunks: int = 200) -> dict:
    """
    Build an embedding index of code chunks for a single repo.
    Returns {chunks: list[str], embeddings: Tensor[N, D]}.
    """
    from create_dataset.embed_repos import (
        iter_source_files, read_text_file, chunk_token_ids,
        SKIP_DIRS, DEFAULT_EXTS, MAX_FILE_BYTES_DEFAULT, TEST_HYPERNET,
    )

    author, rname = repo_name.split("/", 1)
    repo_dir = repos_root / author / rname

    if not repo_dir.exists():
        return {"chunks": [], "embeddings": None}

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
            header = f"# File: {rel}\n"
            all_chunks.append(header + chunk_text)
            if len(all_chunks) >= max_chunks:
                break
        if len(all_chunks) >= max_chunks:
            break

    if not all_chunks:
        return {"chunks": [], "embeddings": None}

    chunk_embs = _embed_texts(embed_model, embed_tokenizer, all_chunks, device, batch_size=32, max_length=chunk_tokens)
    chunk_embs = F.normalize(chunk_embs, p=2, dim=-1)

    return {"chunks": all_chunks, "embeddings": chunk_embs}


@torch.inference_mode()
def _embed_texts(model, tokenizer, texts, device, batch_size=32, max_length=512):
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


def retrieve_chunks(query: str, index: dict, embed_model, embed_tokenizer, device: str, top_k: int = 5) -> list[str]:
    """Retrieve top-k chunks most similar to the query prefix."""
    if not index["chunks"] or index["embeddings"] is None:
        return []

    query_emb = _embed_texts(embed_model, embed_tokenizer, [query[-2000:]], device, max_length=512)
    query_emb = F.normalize(query_emb, p=2, dim=-1)

    sims = (query_emb @ index["embeddings"].T).squeeze(0)
    k = min(top_k, len(index["chunks"]))
    _, top_indices = sims.topk(k)

    return [index["chunks"][i] for i in top_indices.tolist()]


def format_rag_prompt(prefix: str, retrieved_chunks: list[str], max_context_chars: int = 6000) -> str:
    """Format the RAG prompt: retrieved context + original prefix."""
    if not retrieved_chunks:
        return prefix

    context_parts = []
    total_len = 0
    for chunk in retrieved_chunks:
        if total_len + len(chunk) > max_context_chars:
            break
        context_parts.append(chunk)
        total_len += len(chunk)

    context = "\n\n".join(context_parts)
    return f"# Retrieved repository context:\n{context}\n\n# Test file:\n{prefix}"


def main():
    ap = argparse.ArgumentParser(description="RAG baseline evaluation")
    default_dataset = get_default_splits_dir()
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )

    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--repos-root", type=str, default=default_repos,
                    help="Root dir with author/repo structure for chunk retrieval")
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ap.add_argument("--chunk-tokens", type=int, default=512, help="Chunk size in tokens")
    ap.add_argument("--max-context-chars", type=int, default=6000, help="Max chars for retrieved context")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = Path(args.repos_root).expanduser().resolve()

    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    print(f"Loading embedding model: {args.embed_model_name}")
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name, use_fast=True)
    embed_model = AutoModel.from_pretrained(args.embed_model_name, torch_dtype=torch.bfloat16)
    embed_model.to(args.device).eval()

    print(f"Loading generation model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": args.device},
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    bos_id = get_bos_id(tok)

    # Build per-repo chunk indices
    repo_names = sorted(set(it["repo"] for it in items))
    print(f"Building chunk indices for {len(repo_names)} repos...")
    repo_indices = {}
    for rn in repo_names:
        repo_indices[rn] = build_repo_chunk_index(
            repos_root, rn, embed_model, embed_tokenizer,
            args.device, chunk_tokens=args.chunk_tokens,
        )
        n_chunks = len(repo_indices[rn]["chunks"])
        print(f"  {rn}: {n_chunks} chunks")

    # Free embedding model to save GPU memory
    del embed_model
    torch.cuda.empty_cache()

    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    n = len(items)

    print(f"\nEvaluating {n} examples (top_k={args.top_k})...")

    # Re-load embedding model for retrieval queries (lighter weight approach: keep on CPU)
    embed_model_cpu = AutoModel.from_pretrained(args.embed_model_name, torch_dtype=torch.float32)
    embed_model_cpu.eval()
    retrieval_device = "cpu"

    # Move chunk embeddings to CPU
    for rn in repo_indices:
        if repo_indices[rn]["embeddings"] is not None:
            repo_indices[rn]["embeddings"] = repo_indices[rn]["embeddings"].float()

    for i, it in enumerate(items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{n}...", flush=True)

        prefix = it["prefix"]
        target = it["target"]
        repo = it["repo"]

        retrieved = retrieve_chunks(
            prefix, repo_indices[repo],
            embed_model_cpu, embed_tokenizer,
            retrieval_device, top_k=args.top_k,
        )

        rag_prompt = format_rag_prompt(prefix, retrieved, max_context_chars=args.max_context_chars)

        prefix_ids = tok.encode(rag_prompt, add_special_tokens=False)
        input_ids = [bos_id] + prefix_ids
        if len(input_ids) > args.max_input_tokens:
            input_ids = input_ids[-args.max_input_tokens:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(
                input_t, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][len(input_ids):].tolist()
        pred = tok.decode(gen_ids, skip_special_tokens=True)

        pred_clean = postprocess_prediction(pred, target)
        target_clean = strip_comments(target)

        em = exact_match(pred_clean, target_clean)
        bleu = code_bleu_score(pred_clean, target_clean)
        edit_sim = edit_similarity(pred_clean, target_clean)
        em_count += 1 if em else 0
        bleu_sum += bleu
        edit_sum += edit_sim

        entries.append({
            "repo": repo, "expected": target_clean, "got": pred_clean,
            "exact_match": em, "code_bleu": bleu, "edit_similarity": edit_sim,
            "n_retrieved": len(retrieved),
        })

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n

    results = {
        "method": f"rag_top{args.top_k}",
        "split": args.split,
        "exact_match_pct": exact_match_pct,
        "exact_match_count": em_count,
        "n": n,
        "code_bleu": code_bleu_avg,
        "edit_similarity": edit_sim_avg,
        "config": {
            "top_k": args.top_k,
            "chunk_tokens": args.chunk_tokens,
            "max_context_chars": args.max_context_chars,
            "model_name": args.model_name,
            "embed_model_name": args.embed_model_name,
        },
        "entries": entries,
    }

    if args.output:
        results_path = Path(args.output).expanduser().resolve()
    else:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
        results_path = Path(scratch) / "BASELINES" / f"rag_top{args.top_k}_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"RAG Baseline (top-{args.top_k}) on {args.split}")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
