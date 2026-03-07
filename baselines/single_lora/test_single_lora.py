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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments
from evaluation.data_utils import get_default_splits_dir, load_split, get_bos_id

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
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    adapter_path = Path(args.adapter).expanduser().resolve()
    print(f"Loading model + single LoRA adapter from {adapter_path}")

    tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

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

        prefixes = [it["prefix"] for it in batch_items]
        preds = invoke_batch(
            model, tok, prefixes, bos_id=bos_id,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=str(device), pad_token_id=tok.pad_token_id,
        )

        for it, pred in zip(batch_items, preds):
            target = it["target"]
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
