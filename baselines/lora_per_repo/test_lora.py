#!/usr/bin/env python3
"""
Evaluate per-repo LoRA adapter on ir_val (validation) or ir_test (final testing).

Uses the same prompt format as train_lora.py: prefix + "\\n### Target:\\n" -> model generates target.
Results go to {adapter_dir}_results/{split}/lora_results.json.

Usage:
  # Final testing (default): ir_test.json
  python baselines/lora_per_repo/test_lora.py --adapter ./lora-adapters/Chen-zexi_vllm-cli/adapter --limit-repos 1
  # Validation during dev: ir_val.json
  python baselines/lora_per_repo/test_lora.py --adapter ./lora-adapters/Chen-zexi_vllm-cli/adapter --split ir_val --limit-repos 1
"""

import argparse
import json
import os
from difflib import SequenceMatcher
from pathlib import Path

import torch

TARGET_MARKER = "### Target:"
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


def strip_fim_tokens(s: str) -> str:
    for tok in FIM_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def strip_comments(s: str) -> str:
    return s.split("#")[0].strip()


def normalize_for_match(s: str) -> str:
    s = s.strip().rstrip(":, \t")
    s = s.replace(", ", ",").replace(" ,", ",")
    return " ".join(s.split())


def _pred_candidates(pred: str, ref: str) -> list[str]:
    candidates = [normalize_for_match(pred)]
    if "," not in ref and "," in pred:
        candidates.append(normalize_for_match(pred.split(",")[0]))
    if len(ref.split()) == 1 and " " in pred:
        candidates.append(normalize_for_match(pred.split()[0]))
    return candidates


def exact_match(pred: str, ref: str) -> bool:
    norm_ref = normalize_for_match(ref)
    return any(c == norm_ref for c in _pred_candidates(pred, ref))


def edit_similarity(pred: str, ref: str) -> float:
    return SequenceMatcher(None, pred, ref).ratio()


def code_bleu_score(pred: str, ref: str, lang: str = "python") -> float:
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([ref], [pred], lang=lang)
        return result["codebleu"]
    except Exception:
        return 0.0


def load_split(splits_dir: Path, split_name: str, limit_repos: int | None = None, repo_filter: str | None = None) -> list:
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if repo_filter is not None:
        repo_names = [r for r in repo_names if r == repo_filter]
    elif limit_repos is not None and limit_repos > 0:
        repo_names = repo_names[:limit_repos]
    items = []
    for repo in repo_names:
        r = repos[repo]
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if prefix and target:
                items.append({"repo": repo, "prefix": prefix, "target": target})
    return items


def _prepare_input(prompt: str, tokenizer, bos_id: int, max_input_tokens: int) -> list:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [bos_id] + prompt_ids
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
    return input_ids


def invoke_batch(model, tokenizer, prompts: list[str], bos_id: int, max_input_tokens: int,
                 max_new_tokens: int, device: str, pad_token_id: int) -> list[str]:
    if not prompts:
        return []
    all_input_ids = [_prepare_input(p, tokenizer, bos_id, max_input_tokens) for p in prompts]
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = pad_token_id

    padded = []
    attn_masks = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded_ids = [pad_id] * pad_len + ids
        padded.append(padded_ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))

    input_t = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_t,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    preds = []
    for gen_full in out:
        gen_ids = gen_full[max_len:].tolist()
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
        preds.append(pred)
    return preds


def get_bos_id(tok):
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return tok.pad_token_id


def main():
    ap = argparse.ArgumentParser(description="Evaluate per-repo LoRA on ir_val/ir_test")
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--adapter", type=str, required=True,
                    help="Path to LoRA adapter (e.g. ./lora-adapters/repo_slug/adapter)")
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Dir with ir_val.json, ir_test.json, etc.")
    ap.add_argument("--split", type=str, default="ir_test",
                    help="Split to evaluate: ir_test for final testing, ir_val for validation (default: ir_test)")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=1,
                    help="Use first N repos when --repo not set (default: 1)")
    ap.add_argument("--repo", type=str, default=None,
                    help="Evaluate only this repo (e.g. 0xricksanchez/like-dbg); auto-derived from adapter path if in PER_REPO_LORA/")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = Path(args.adapter).expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    adapter_dir = adapter_path.parent
    results_root = Path(str(adapter_dir) + "_results")
    output_dir = results_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive repo from adapter path if in PER_REPO_LORA/author/repo_name/adapter
    repo_filter = args.repo
    if repo_filter is None and "PER_REPO_LORA" in str(adapter_path):
        parts = adapter_path.parts
        try:
            idx = parts.index("PER_REPO_LORA")
            if idx + 2 < len(parts):
                repo_filter = f"{parts[idx + 1]}/{parts[idx + 2]}"
        except ValueError:
            pass

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos, repo_filter=repo_filter)
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    print(f"Loading model + LoRA adapter from {adapter_path}")
    tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
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

    # Prompt format: prefix + "\n### Target:\n" (model generates target)
    prompt_suffix = "\n" + TARGET_MARKER + "\n"

    print(f"Evaluating {n} examples on {args.split}.json (batch_size={batch_size})...")
    for start in range(0, n, batch_size):
        batch_items = items[start : start + batch_size]
        if (start // batch_size + 1) % 10 == 0 or start == 0:
            print(f"  {min(start + batch_size, n)}/{n}...", flush=True)

        prompts = [it["prefix"] + prompt_suffix for it in batch_items]
        preds = invoke_batch(
            model, tok, prompts,
            bos_id=bos_id,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=str(device),
            pad_token_id=tok.pad_token_id,
        )

        for it, pred in zip(batch_items, preds):
            target = it["target"]
            pred = strip_fim_tokens(pred)
            if "\n" not in target and "\n" in pred:
                pred = pred.split("\n")[0]

            pred_clean = strip_comments(pred)
            target_clean = strip_comments(target)

            em = exact_match(pred_clean, target_clean)
            bleu = code_bleu_score(pred_clean, target_clean)
            edit_sim = edit_similarity(pred_clean, target_clean)
            em_count += 1 if em else 0
            bleu_sum += bleu
            edit_sum += edit_sim

            entries.append({
                "repo": it["repo"],
                "expected": target_clean,
                "got": pred_clean,
                "exact_match": em,
                "code_bleu": bleu,
                "edit_similarity": edit_sim,
            })

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n

    results = {
        "exact_match_pct": exact_match_pct,
        "exact_match_count": em_count,
        "n": n,
        "code_bleu": code_bleu_avg,
        "edit_similarity": edit_sim_avg,
        "entries": entries,
    }
    results_path = output_dir / "lora_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Results on {args.split}.json (per-repo LoRA)")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
