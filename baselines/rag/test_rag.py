#!/usr/bin/env python3
"""
RAG baseline: retrieve top-k code chunks from a pre-built index,
prepend as context, then use Qwen2.5-Coder to complete assertions.

Pre-build indices first:
    python baselines/rag/build_indices.py

Then evaluate:
    python baselines/rag/test_rag.py --split cr_test --top-k 5
    python baselines/rag/test_rag.py --split cr_test --top-k 10 --max-input-tokens 16384
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


def load_cached_index(cache_dir: Path, repo_name: str) -> dict:
    """Load pre-built chunk index from cache."""
    safe_name = repo_name.replace("/", "__")
    cache_path = cache_dir / f"{safe_name}.pt"
    if not cache_path.exists():
        return {"chunks": [], "embeddings": None}
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return {
        "chunks": data["chunks"],
        "embeddings": data["embeddings"].float() if data["embeddings"] is not None else None,
    }


@torch.inference_mode()
def embed_query(text: str, model, tokenizer, device: str, max_length: int = 512) -> torch.Tensor:
    """Embed a single query text."""
    enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(mean.cpu(), p=2, dim=-1)


def retrieve_chunks(query_emb: torch.Tensor, index: dict, top_k: int = 5) -> list[str]:
    """Retrieve top-k chunks by cosine similarity."""
    if not index["chunks"] or index["embeddings"] is None:
        return []
    emb = index["embeddings"]
    if query_emb.dtype != emb.dtype:
        query_emb = query_emb.to(emb.dtype)
    sims = (query_emb @ emb.T).squeeze(0)
    k = min(top_k, len(index["chunks"]))
    _, top_indices = sims.topk(k)
    return [index["chunks"][i] for i in top_indices.tolist()]


def format_rag_prompt(prefix: str, retrieved_chunks: list[str]) -> str:
    """
    Present retrieved code as natural preceding context.
    No artificial headers -- the model is a code completion model.
    """
    if not retrieved_chunks:
        return prefix
    context = "\n\n\n".join(retrieved_chunks)
    return f"{context}\n\n\n{prefix}"


def main():
    ap = argparse.ArgumentParser(description="RAG baseline evaluation")
    default_dataset = get_default_splits_dir()
    default_cache = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "RAG_CHUNK_CACHE",
    )

    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--cache-dir", type=str, default=default_cache,
                    help="Dir with pre-built chunk indices (from build_indices.py)")
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384,
                    help="Max input tokens (Qwen2.5-Coder supports 32K)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {cache_dir}\n"
            "Run build_indices.py first: python baselines/rag/build_indices.py"
        )

    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    # Load cached chunk indices
    repo_names = sorted(set(it["repo"] for it in items))
    print(f"Loading cached chunk indices for {len(repo_names)} repos...")
    repo_indices = {}
    for rn in repo_names:
        repo_indices[rn] = load_cached_index(cache_dir, rn)
        n_chunks = len(repo_indices[rn]["chunks"])
        if n_chunks == 0:
            print(f"  WARNING: {rn}: no cached chunks (run build_indices.py)")
        else:
            print(f"  {rn}: {n_chunks} chunks")

    # Load embedding model (for query encoding only -- lightweight)
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

    # Stop generation on FIM tokens — Qwen2.5-Coder can enter Fill-In-the-Middle
    # mode when retrieved code chunks precede the prefix, producing garbage output.
    # Adding FIM tokens as stop tokens prevents this without discarding the whole run.
    _fim_strings = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>"]
    _fim_ids = [tok.encode(t, add_special_tokens=False) for t in _fim_strings]
    fim_stop_ids = [ids[0] for ids in _fim_ids if len(ids) == 1]
    eos_ids = list({tok.eos_token_id} | set(fim_stop_ids))

    bos_id = get_bos_id(tok)

    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    n = len(items)

    print(f"\nEvaluating {n} examples (top_k={args.top_k}, max_input={args.max_input_tokens})...")

    for i, it in enumerate(items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{n}...", flush=True)

        prefix = it["prefix"]
        target = it["target"]
        repo = it["repo"]

        # Embed query (last 2000 chars of prefix for relevance)
        query_emb = embed_query(prefix[-2000:], embed_model, embed_tokenizer, args.device)

        # Move query to CPU for retrieval against cached embeddings
        retrieved = retrieve_chunks(query_emb.cpu(), repo_indices[repo], top_k=args.top_k)

        rag_prompt = format_rag_prompt(prefix, retrieved)

        prefix_ids = tok.encode(rag_prompt, add_special_tokens=False)
        input_ids = [bos_id] + prefix_ids
        if len(input_ids) > args.max_input_tokens:
            input_ids = input_ids[-args.max_input_tokens:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(
                input_t, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
            )
        gen_ids = out[0][len(input_ids):].tolist()
        pred = tok.decode(gen_ids, skip_special_tokens=True)

        pred_raw = pred
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
            "got_raw": pred_raw,
            "exact_match": em, "code_bleu": bleu, "edit_similarity": edit_sim,
            "n_retrieved": len(retrieved),
        })

        # Write after every 50 examples so partial results survive crashes
        if (i + 1) % 50 == 0 or (i + 1) == n:
            n_eval = len(entries)
            results = {
                "method": f"rag_top{args.top_k}",
                "split": args.split,
                "exact_match_pct": 100.0 * em_count / n_eval,
                "exact_match_count": em_count,
                "n": n_eval,
                "n_total": n,
                "code_bleu": bleu_sum / n_eval,
                "edit_similarity": edit_sum / n_eval,
                "config": {
                    "top_k": args.top_k,
                    "max_input_tokens": args.max_input_tokens,
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

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n
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
