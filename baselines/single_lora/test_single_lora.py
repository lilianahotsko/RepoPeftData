#!/usr/bin/env python3
"""
Evaluate single LoRA adapter on split JSON.

Usage:
    python baselines/single_lora/test_single_lora.py --adapter $SCRATCH/TRAINING_CHECKPOINTS/SINGLE_LORA/adapter --split cr_test
    python baselines/single_lora/test_single_lora.py --adapter $SCRATCH/TRAINING_CHECKPOINTS/SINGLE_LORA/adapter --split ir_test
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


def _load_cached_index(cache_dir: Path, repo_name: str) -> dict:
    safe = repo_name.replace("/", "__")
    path = cache_dir / f"{safe}.pt"
    if not path.exists():
        return {"chunks": [], "embeddings": None}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "chunks": data["chunks"],
        "embeddings": data["embeddings"].float() if data["embeddings"] is not None else None,
    }


@torch.inference_mode()
def _embed_query(text: str, model, tokenizer, device: str) -> torch.Tensor:
    enc = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(mean.cpu(), p=2, dim=-1)


def _retrieve_and_prepend(prefix: str, repo: str, cache_dir: Path,
                           repo_indices: dict, embed_model, embed_tok,
                           embed_device: str, top_k: int) -> str:
    if repo not in repo_indices:
        repo_indices[repo] = _load_cached_index(cache_dir, repo)
    index = repo_indices[repo]
    if not index["chunks"]:
        return prefix
    query_emb = _embed_query(prefix, embed_model, embed_tok, embed_device)
    emb = index["embeddings"]
    sims = (query_emb.to(emb.dtype) @ emb.T).squeeze(0)
    k = min(top_k, len(index["chunks"]))
    _, top_idx = sims.topk(k)
    chunks = [index["chunks"][i] for i in top_idx.tolist()]
    return "\n\n\n".join(chunks) + f"\n\n\n{prefix}"

TARGET_MARKER = "### Target:"


def invoke_batch(model, tokenizer, prefixes, bos_id, max_input_tokens, max_new_tokens, device, pad_token_id):
    if not prefixes:
        return []
    all_input_ids = []
    for p in prefixes:
        prompt = p + "\n" + TARGET_MARKER + "\n"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = [bos_id] + ids
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
        all_input_ids.append(ids)

    max_len = max(len(ids) for ids in all_input_ids)
    padded, attn_masks = [], []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded.append([pad_token_id] * pad_len + ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))

    input_t = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_t, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )

    preds = []
    for gen_full in out:
        gen_ids = gen_full[max_len:].tolist()
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return preds


def main():
    ap = argparse.ArgumentParser(description="Evaluate single LoRA on split JSON")
    default_dataset = get_default_splits_dir()

    ap.add_argument("--adapter", type=str, required=True, help="Path to single LoRA adapter")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rag-cache-dir", type=str, default=None,
                    help="Pre-built chunk index dir. If set, top-k chunks prepended at inference.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embed-model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--oracle-cache-dir", type=str, default=None,
                    help="Pre-built oracle context cache dir (ORACLE_CONTEXT_CACHE_V2). If set, DRC context prepended.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    adapter_path = Path(args.adapter).expanduser().resolve()
    print(f"Loading model + single LoRA adapter from {adapter_path} (bf16, no quantization)")

    tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    oracle_cache_dir = None
    oracle_contexts: dict = {}
    if args.oracle_cache_dir:
        from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context, augment_prefix_with_oracle
        oracle_cache_dir = Path(args.oracle_cache_dir).expanduser().resolve()
        print(f"Oracle DRC mode: cache={oracle_cache_dir}")

    rag_cache_dir = None
    embed_model = embed_tok = None
    repo_indices: dict = {}
    if args.rag_cache_dir:
        from transformers import AutoModel
        rag_cache_dir = Path(args.rag_cache_dir).expanduser().resolve()
        print(f"RAG mode: cache={rag_cache_dir}, top_k={args.top_k}")
        embed_tok = __import__("transformers").AutoTokenizer.from_pretrained(
            args.embed_model_name, use_fast=True)
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name, torch_dtype=torch.bfloat16).to(args.device).eval()

    bos_id = get_bos_id(tok)
    device = next(model.parameters()).device
    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    n = len(items)
    batch_size = args.batch_size

    print(f"Evaluating {n} examples (batch_size={batch_size})...")
    for start in range(0, n, batch_size):
        batch_items = items[start:start + batch_size]
        if (start // batch_size + 1) % 10 == 0 or start == 0:
            print(f"  {min(start + batch_size, n)}/{n}...", flush=True)

        if oracle_cache_dir:
            prefixes = []
            for it in batch_items:
                repo = it["repo"]
                if repo not in oracle_contexts:
                    oracle_contexts[repo] = load_oracle_cache(oracle_cache_dir, repo)
                oracle_code = lookup_oracle_context(oracle_contexts[repo], it.get("metadata", {}))
                prefixes.append(augment_prefix_with_oracle(it["prefix"], oracle_code))
        elif rag_cache_dir:
            prefixes = [
                _retrieve_and_prepend(it["prefix"], it["repo"], rag_cache_dir,
                                      repo_indices, embed_model, embed_tok,
                                      args.device, args.top_k)
                for it in batch_items
            ]
        else:
            prefixes = [it["prefix"] for it in batch_items]
        preds = invoke_batch(
            model, tok, prefixes, bos_id=bos_id,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=str(device), pad_token_id=tok.pad_token_id,
        )

        for it, pred in zip(batch_items, preds):
            target = it["target"]
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
                "repo": it["repo"], "expected": target_clean, "got": pred_clean,
                "got_raw": pred_raw,
                "exact_match": em, "code_bleu": bleu, "edit_similarity": edit_sim,
            })

        n_eval = len(entries)
        _res = {
            "method": "single_lora", "split": args.split,
            "exact_match_pct": 100.0 * em_count / n_eval,
            "exact_match_count": em_count,
            "n": n_eval, "n_total": n,
            "code_bleu": bleu_sum / n_eval,
            "edit_similarity": edit_sum / n_eval,
            "entries": entries,
        }
        if args.output:
            results_path = Path(args.output).expanduser().resolve()
        else:
            scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
            results_path = Path(scratch) / "BASELINES" / f"single_lora_{args.split}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(_res, indent=2), encoding="utf-8")

    exact_match_pct = 100.0 * em_count / n if n else 0
    code_bleu_avg = bleu_sum / n if n else 0
    edit_sim_avg = edit_sum / n if n else 0

    print("\n" + "=" * 60)
    print(f"Single LoRA Baseline on {args.split}")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
