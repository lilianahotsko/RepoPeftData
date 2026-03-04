#!/usr/bin/env python3
"""
Oracle Context baseline: use import-resolved source code as context.

Pre-build contexts first:
    python baselines/oracle_context/build_context.py

Then evaluate:
    python baselines/oracle_context/test_oracle_context.py --split cr_test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments
from evaluation.data_utils import get_default_splits_dir, load_split, get_bos_id


def load_cached_context(cache_dir: Path, repo_name: str) -> dict:
    """Load pre-built oracle contexts for a repo."""
    safe_name = repo_name.replace("/", "__")
    cache_path = cache_dir / f"{safe_name}.json"
    if not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return data.get("contexts", {})


def lookup_context(contexts: dict, metadata: dict) -> str:
    """Look up extracted code for a specific QnA pair."""
    key = f"{metadata.get('file', '')}::{metadata.get('lineno', 0)}"
    ctx = contexts.get(key)
    if ctx and ctx.get("extracted_code"):
        return ctx["extracted_code"]
    return ""


def format_oracle_prompt(prefix: str, oracle_code: str, tokenizer=None, max_input_tokens: int = 16384) -> str:
    """
    Prepend extracted source code before the test prefix.
    Budget-based: the prefix is always fully preserved. Oracle context
    fills the remaining token budget so it can never hurt vs pretrained.
    """
    if not oracle_code.strip():
        return prefix

    if tokenizer is None:
        return f"{oracle_code}\n\n\n{prefix}"

    separator = "\n\n\n"
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    separator_tokens = len(tokenizer.encode(separator, add_special_tokens=False))
    budget = max_input_tokens - prefix_tokens - separator_tokens - 2  # 2 for BOS + safety

    if budget <= 50:
        return prefix

    # Tokenize oracle definitions individually (they're separated by \n\n\n)
    definitions = oracle_code.split("\n\n\n")
    kept = []
    used = 0
    for defn in definitions:
        defn_tokens = len(tokenizer.encode(defn, add_special_tokens=False))
        if used + defn_tokens > budget:
            break
        kept.append(defn)
        used += defn_tokens

    if not kept:
        return prefix

    return separator.join(kept) + separator + prefix


def main():
    ap = argparse.ArgumentParser(description="Oracle Context baseline evaluation")
    default_dataset = get_default_splits_dir()
    default_cache = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "ORACLE_CONTEXT_CACHE",
    )

    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--cache-dir", type=str, default=default_cache,
                    help="Dir with pre-built oracle contexts (from build_context.py)")
    ap.add_argument("--split", type=str, default="cr_test_structured")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384,
                    help="Max input tokens (Qwen2.5-Coder supports 32K)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {cache_dir}\n"
            "Run build_context.py first: python baselines/oracle_context/build_context.py"
        )

    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    # Load cached contexts per repo
    repo_names = sorted(set(it["repo"] for it in items))
    print(f"Loading oracle contexts for {len(repo_names)} repos...")
    repo_contexts = {}
    n_with_context = 0
    n_total = 0
    for rn in repo_names:
        repo_contexts[rn] = load_cached_context(cache_dir, rn)
        n_pairs = len(repo_contexts[rn])
        n_with = sum(1 for v in repo_contexts[rn].values() if v.get("n_chars_extracted", 0) > 0)
        print(f"  {rn}: {n_with}/{n_pairs} pairs with context")
        n_with_context += n_with
        n_total += n_pairs

    if n_total > 0:
        print(f"  Overall: {n_with_context}/{n_total} pairs ({100 * n_with_context / n_total:.1f}%) have extracted context")

    print(f"\nLoading model: {args.model_name}")
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

    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    n_had_context = 0
    entries = []
    n = len(items)

    print(f"\nEvaluating {n} examples (max_input={args.max_input_tokens})...")

    for i, it in enumerate(items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{n}...", flush=True)

        prefix = it["prefix"]
        target = it["target"]
        repo = it["repo"]
        metadata = it.get("metadata", {})

        oracle_code = lookup_context(repo_contexts.get(repo, {}), metadata)
        if oracle_code:
            n_had_context += 1

        prompt = format_oracle_prompt(prefix, oracle_code, tokenizer=tok, max_input_tokens=args.max_input_tokens)

        prefix_ids = tok.encode(prompt, add_special_tokens=False)
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
            "had_oracle_context": bool(oracle_code),
        })

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n

    results = {
        "method": "oracle_context",
        "split": args.split,
        "exact_match_pct": exact_match_pct,
        "exact_match_count": em_count,
        "n": n,
        "n_with_context": n_had_context,
        "code_bleu": code_bleu_avg,
        "edit_similarity": edit_sim_avg,
        "config": {
            "max_input_tokens": args.max_input_tokens,
            "model_name": args.model_name,
        },
        "entries": entries,
    }

    if args.output:
        results_path = Path(args.output).expanduser().resolve()
    else:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
        results_path = Path(scratch) / "BASELINES" / f"oracle_context_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Oracle Context Baseline on {args.split}")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print(f"  Pairs w/ context: {n_had_context}/{n} ({100 * n_had_context / n:.1f}%)")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
