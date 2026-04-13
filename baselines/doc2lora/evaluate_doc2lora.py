#!/usr/bin/env python3
"""
Evaluate Doc-to-LoRA (D2L) on RepoPeft benchmark.

D2L uses a context encoder + Perceiver + HyperLoRA to generate adapter weights
from a document. We internalize oracle context (DRC) for each repo, then use
the adapted model for assertion completion.

Supports both the pretrained Gemma-based SakanaAI checkpoint and custom-trained
Qwen-based checkpoints (via --base-model override).

Usage:
    # Pretrained Gemma checkpoint:
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path doc2lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
        --split cr_test \
        --output $SCRATCH/BASELINES/doc2lora_cr_test.json

    # Trained Qwen checkpoint (with oracle context):
    python baselines/doc2lora/evaluate_doc2lora.py \
        --checkpoint-path $SCRATCH/TRAINING_CHECKPOINTS/DOC2LORA_QWEN/checkpoint-XXXX/pytorch_model.bin \
        --base-model Qwen/Qwen2.5-Coder-1.5B \
        --split cr_test \
        --use-oracle \
        --oracle-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V4 \
        --output $SCRATCH/BASELINES/doc2lora_trained_drc_cr_test.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "doc2lora" / "src"))

from evaluation.data_utils import get_default_splits_dir, load_split
from evaluation.metrics import (
    postprocess_prediction, strip_comments,
    exact_match, edit_similarity, code_bleu_score,
)

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


def prepare_input_ids(prefix, tokenizer, max_input_tokens):
    """Tokenize prefix for Gemma model (no BOS prepending — chat model handles it)."""
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) > max_input_tokens:
        ids = ids[-max_input_tokens:]
    return ids


def invoke_batch(model, tokenizer, prefixes, max_input_tokens, max_new_tokens, device):
    """Left-padded batched greedy inference using the base_model inside D2L."""
    if not prefixes:
        return []
    all_input_ids = [prepare_input_ids(p, tokenizer, max_input_tokens) for p in prefixes]
    max_len = max(len(ids) for ids in all_input_ids)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded, attn_masks = [], []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded.append([pad_token_id] * pad_len + ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))

    input_t = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

    # Use base_model.generate (LoRA is already applied via internalize)
    out = model.base_model.generate(
        input_t,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(gen[max_len:].tolist(), skip_special_tokens=True)
            for gen in out]


def main():
    ap = argparse.ArgumentParser(description="Doc-to-LoRA evaluation on RepoPeft")
    ap.add_argument("--checkpoint-path", required=True, type=Path,
                    help="Path to D2L checkpoint (pytorch_model.bin)")
    ap.add_argument("--base-model", type=str, default=None,
                    help="Override base model name (e.g. Qwen/Qwen2.5-Coder-1.5B). "
                         "If not set, uses the model name embedded in the checkpoint.")
    ap.add_argument("--split", default="cr_test")
    ap.add_argument("--splits-dir", type=str, default=get_default_splits_dir())
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--use-oracle", action="store_true",
                    help="Use oracle context as D2L document input")
    ap.add_argument("--oracle-cache-dir", type=str, default=None)
    ap.add_argument("--max-oracle-tokens", type=int, default=4096,
                    help="Max tokens for oracle context fed to D2L encoder")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load D2L model ───────────────────────────────────────────────────────
    print(f"Loading D2L checkpoint: {args.checkpoint_path}")
    state_dict = torch.load(str(args.checkpoint_path), map_location=device, weights_only=False)

    if args.base_model:
        state_dict["base_model_name_or_path"] = args.base_model
        print(f"Using base model override: {args.base_model}")

    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.eval()
    model.to(device)

    tokenizer = get_tokenizer(model.base_model.config.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Context encoder tokenizer (may differ from base model tokenizer)
    ctx_model_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
    if ctx_model_name is None:
        ctx_model_name = model.base_model.config.name_or_path
    ctx_tokenizer = get_tokenizer(ctx_model_name)

    print(f"Base model: {model.base_model.config.name_or_path}")
    print(f"Context encoder: {ctx_model_name}")

    # ── Load test split ──────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if not items:
        print(f"No items found in {args.split}.json")
        return

    # ── Load oracle contexts ─────────────────────────────────────────────────
    oracle_contexts_by_repo = {}
    if args.use_oracle:
        from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context
        if args.oracle_cache_dir is None:
            from evaluation.oracle_utils import get_default_oracle_cache_dir
            args.oracle_cache_dir = get_default_oracle_cache_dir()
        oracle_cache_dir = Path(args.oracle_cache_dir)
        print(f"Loading oracle contexts from {oracle_cache_dir}")

        # Pre-load all oracle caches for repos in this split
        seen_repos = set()
        for item in items:
            rn = item["repo"]
            if rn not in seen_repos:
                seen_repos.add(rn)
                cache = load_oracle_cache(oracle_cache_dir, rn)
                if cache:
                    oracle_contexts_by_repo[rn] = cache
        print(f"  Loaded oracle contexts for {len(oracle_contexts_by_repo)}/{len(seen_repos)} repos")

    # Group by repo
    repo_items = defaultdict(list)
    for item in items:
        repo_items[item["repo"]].append(item)

    repo_names = sorted(repo_items.keys())
    print(f"Split: {args.split}, repos: {len(repo_names)}, pairs: {len(items)}")

    # ── Evaluate per repo ────────────────────────────────────────────────────
    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    results_path = args.output.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for ri, repo_name in enumerate(repo_names):
        print(f"[{ri+1}/{len(repo_names)}] {repo_name}", flush=True)

        # Build context document for D2L from oracle contexts for this repo
        # Collect unique oracle code snippets for all pairs in this repo
        repo_pairs = repo_items[repo_name]

        if args.use_oracle and repo_name in oracle_contexts_by_repo:
            oracle_cache = oracle_contexts_by_repo[repo_name]
            # Gather all unique oracle code for this repo
            oracle_snippets = set()
            for pair in repo_pairs:
                code = lookup_oracle_context(oracle_cache, pair.get("metadata", {}))
                if code:
                    oracle_snippets.add(code)

            if oracle_snippets:
                # Concatenate oracle snippets as the "document" for D2L
                doc_text = "\n\n".join(sorted(oracle_snippets))
                # Truncate to max_oracle_tokens using ctx_tokenizer
                doc_ids = ctx_tokenizer.encode(doc_text, add_special_tokens=False)
                if len(doc_ids) > args.max_oracle_tokens:
                    doc_ids = doc_ids[:args.max_oracle_tokens]
                    doc_text = ctx_tokenizer.decode(doc_ids, skip_special_tokens=True)
            else:
                doc_text = None
        else:
            doc_text = None

        # Internalize the repo context
        model.reset()
        model.patch_lora_forward()
        if doc_text:
            model.internalize(doc_text)

        # Generate predictions
        prefixes = [p["prefix"] for p in repo_pairs]
        targets = [p["target"] for p in repo_pairs]

        for start in range(0, len(prefixes), args.batch_size):
            batch_prefixes = prefixes[start:start + args.batch_size]
            batch_targets = targets[start:start + args.batch_size]

            preds = invoke_batch(
                model, tokenizer, batch_prefixes,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

            for pred_raw, target in zip(preds, batch_targets):
                pred_clean = postprocess_prediction(pred_raw, target)
                target_clean = strip_comments(target)

                em = exact_match(pred_clean, target_clean)
                bleu = code_bleu_score(pred_clean, target_clean)
                edit_sim = edit_similarity(pred_clean, target_clean)
                em_count += 1 if em else 0
                bleu_sum += bleu
                edit_sum += edit_sim

                entries.append({
                    "repo": repo_name,
                    "expected": target_clean,
                    "got": pred_clean,
                    "got_raw": pred_raw,
                    "exact_match": em,
                    "code_bleu": bleu,
                    "edit_similarity": edit_sim,
                })

        # Print per-repo stats
        repo_entries = [e for e in entries if e["repo"] == repo_name]
        if repo_entries:
            repo_em = sum(1 for e in repo_entries if e["exact_match"]) / len(repo_entries) * 100
            repo_es = sum(e["edit_similarity"] for e in repo_entries) / len(repo_entries)
            print(f"  {len(repo_entries)} pairs | EM={repo_em:.1f}% ES={repo_es:.3f}")

        # Save after every repo
        n_eval = len(entries)
        if n_eval > 0:
            method = "doc2lora_drc" if args.use_oracle else "doc2lora"
            results = {
                "method": method,
                "split": args.split,
                "model": model.base_model.config.name_or_path,
                "checkpoint": str(args.checkpoint_path),
                "exact_match_pct": 100.0 * em_count / n_eval,
                "exact_match_count": em_count,
                "n": n_eval,
                "n_total": len(items),
                "code_bleu": bleu_sum / n_eval,
                "edit_similarity": edit_sum / n_eval,
                "use_oracle": args.use_oracle,
                "entries": entries,
            }
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_eval = len(entries)
    method_name = "Doc-to-LoRA" + (" + DRC" if args.use_oracle else "")
    print(f"\n{'='*60}")
    print(f"{method_name} on {args.split}")
    print(f"{'='*60}")
    print(f"  Exact Match:     {100.0 * em_count / n_eval:.2f}% ({em_count}/{n_eval})")
    print(f"  Code BLEU:       {bleu_sum / n_eval:.4f}")
    print(f"  Edit Similarity: {edit_sum / n_eval:.4f}")
    print(f"{'='*60}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
