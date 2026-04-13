#!/usr/bin/env python3
"""
Generate teacher logprobs for Doc2LoRA context distillation training.

Teacher: Qwen2.5-Coder-1.5B with DRC context prepended to the prefix.
For each training example with non-empty DRC, we:
  1. Tokenize DRC text alone -> ctx_ids
  2. Tokenize prefix + target -> input_ids, with response_start_end
  3. Forward pass on [DRC + separator + prefix + target] -> teacher logits
  4. Extract top-16 logprobs at target token positions
  5. Save as D2L-compatible parquet

Output parquet columns (per row = one repo, multiple QAs):
  - ctx_ids:            list[int]           tokenized DRC context
  - input_ids:          list[list[int]]     per-QA tokenized prefix+target
  - response_start_end: list[tuple[int,int]] per-QA (start, end) of target
  - logprobs_vals:      list[np.ndarray]    per-QA (n_target_toks, 16) float16
  - logprobs_indices:   list[np.ndarray]    per-QA (n_target_toks, 16) int32

Usage:
    python baselines/doc2lora/generate_teacher_logprobs.py \
        --splits-dir $SCRATCH/REPO_DATASET \
        --oracle-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V4 \
        --output-dir doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train \
        --model Qwen/Qwen2.5-Coder-1.5B \
        --split train
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.data_utils import get_default_splits_dir, load_split
from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context

TOP_K = 16
DRC_SEPARATOR = "\n\n\n"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--split", default="train")
    ap.add_argument("--splits-dir", default=get_default_splits_dir())
    ap.add_argument("--oracle-cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-ctx-tokens", type=int, default=4096,
                    help="Max tokens for DRC context (ctx_ids)")
    ap.add_argument("--max-teacher-tokens", type=int, default=8192,
                    help="Max total teacher input length (DRC + prefix + target)")
    ap.add_argument("--max-input-tokens", type=int, default=2048,
                    help="Max tokens for prefix + target (input_ids)")
    ap.add_argument("--shard-size", type=int, default=50,
                    help="Number of repos per parquet shard")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


@torch.no_grad()
def extract_teacher_logprobs(
    model,
    teacher_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run teacher forward pass and extract top-k logprobs at target positions.

    teacher_ids: (1, seq_len) on device
    target_start, target_end: positions in teacher_ids where the target tokens are
    Returns: (logprobs_vals, logprobs_indices) each shape (n_target_tokens, top_k)
    """
    outputs = model(teacher_ids)
    logits = outputs.logits[0]  # (seq_len, vocab)

    # Logits at positions [target_start-1 .. target_end-1) predict tokens at
    # positions [target_start .. target_end)
    pred_logits = logits[target_start - 1 : target_end - 1]  # (n_target, vocab)

    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    top_vals, top_idx = torch.topk(log_probs, k=top_k, dim=-1)  # (n_target, k)

    return (
        top_vals.cpu().to(torch.float16).numpy(),
        top_idx.cpu().to(torch.int32).numpy(),
    )


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load split
    splits_dir = Path(args.splits_dir)
    items = load_split(splits_dir, args.split)
    print(f"Loaded {len(items)} items from {args.split}")

    # Load oracle caches
    oracle_cache_dir = Path(args.oracle_cache_dir)
    repo_names = sorted(set(item["repo"] for item in items))
    oracle_caches = {}
    for rn in repo_names:
        cache = load_oracle_cache(oracle_cache_dir, rn)
        if cache:
            oracle_caches[rn] = cache
    print(f"Loaded oracle caches for {len(oracle_caches)}/{len(repo_names)} repos")

    # Group items by repo
    repo_items = defaultdict(list)
    for item in items:
        repo_items[item["repo"]].append(item)

    # Process repos and build parquet rows
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_no_drc = 0
    skipped_empty_target = 0
    shard_idx = 0

    for ri, repo_name in enumerate(tqdm(repo_names, desc="Repos")):
        pairs = repo_items[repo_name]
        cache = oracle_caches.get(repo_name, {})

        # Collect unique DRC snippets for this repo -> ctx_ids
        all_drc_snippets = set()
        pair_drc = []
        for pair in pairs:
            drc = lookup_oracle_context(cache, pair.get("metadata", {}))
            pair_drc.append(drc)
            if drc:
                all_drc_snippets.add(drc)

        if not all_drc_snippets:
            skipped_no_drc += len(pairs)
            continue

        # Build repo-level DRC document (concatenation of unique snippets)
        repo_doc = "\n\n".join(sorted(all_drc_snippets))
        ctx_ids = tokenizer.encode(repo_doc, add_special_tokens=False)
        if len(ctx_ids) > args.max_ctx_tokens:
            ctx_ids = ctx_ids[: args.max_ctx_tokens]

        # Process each QA pair
        qa_input_ids = []
        qa_response_se = []
        qa_logprobs_vals = []
        qa_logprobs_indices = []

        for pair, drc_text in zip(pairs, pair_drc):
            prefix = pair["prefix"]
            target = pair["target"]

            # Tokenize prefix + target for student (input_ids)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                skipped_empty_target += 1
                continue

            student_ids = prefix_ids + target_ids
            if len(student_ids) > args.max_input_tokens:
                # Truncate prefix from the left, keep target
                max_prefix = args.max_input_tokens - len(target_ids)
                if max_prefix <= 0:
                    skipped_empty_target += 1
                    continue
                prefix_ids = prefix_ids[-max_prefix:]
                student_ids = prefix_ids + target_ids

            response_start = len(prefix_ids)
            response_end = len(student_ids)

            # Build teacher input: DRC + separator + prefix + target
            # Use per-example DRC if available, else repo-level
            if drc_text:
                teacher_ctx = drc_text
            else:
                teacher_ctx = repo_doc

            teacher_ctx_ids = tokenizer.encode(teacher_ctx, add_special_tokens=False)
            sep_ids = tokenizer.encode(DRC_SEPARATOR, add_special_tokens=False)

            teacher_full = teacher_ctx_ids + sep_ids + prefix_ids + target_ids
            if len(teacher_full) > args.max_teacher_tokens:
                # Truncate context from the right
                budget = args.max_teacher_tokens - len(sep_ids) - len(prefix_ids) - len(target_ids)
                if budget <= 0:
                    # Can't fit even prefix+target, skip
                    skipped_empty_target += 1
                    continue
                teacher_ctx_ids = teacher_ctx_ids[:budget]
                teacher_full = teacher_ctx_ids + sep_ids + prefix_ids + target_ids

            # Target token positions in teacher_full
            target_start_teacher = len(teacher_full) - len(target_ids)
            target_end_teacher = len(teacher_full)

            # Extract teacher logprobs
            teacher_tensor = torch.tensor([teacher_full], dtype=torch.long, device=device)
            lp_vals, lp_idx = extract_teacher_logprobs(
                model, teacher_tensor,
                target_start_teacher, target_end_teacher,
                top_k=TOP_K,
            )

            qa_input_ids.append(student_ids)
            qa_response_se.append([response_start, response_end])
            qa_logprobs_vals.append(lp_vals)
            qa_logprobs_indices.append(lp_idx)

        if not qa_input_ids:
            continue

        rows.append({
            "ctx_ids": ctx_ids,
            "input_ids": qa_input_ids,
            "response_start_end": qa_response_se,
            "logprobs_vals": qa_logprobs_vals,
            "logprobs_indices": qa_logprobs_indices,
        })

        # Write shard when enough repos accumulated
        if len(rows) >= args.shard_size:
            shard_path = output_dir / f"ds_{shard_idx:04d}.parquet"
            Dataset.from_list(rows).to_parquet(str(shard_path))
            print(f"  Wrote shard {shard_path} ({len(rows)} repos)")
            rows = []
            shard_idx += 1

    # Write remaining rows
    if rows:
        shard_path = output_dir / f"ds_{shard_idx:04d}.parquet"
        Dataset.from_list(rows).to_parquet(str(shard_path))
        print(f"  Wrote shard {shard_path} ({len(rows)} repos)")

    print(f"\nDone. Skipped {skipped_no_drc} pairs (no DRC), "
          f"{skipped_empty_target} pairs (empty target/overflow)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
