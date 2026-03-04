#!/usr/bin/env python3
"""
Evaluate pretrained Qwen2.5-Coder-1.5B on split JSON (no hypernetwork/LoRA).

Metrics: exact match, code BLEU, edit similarity.
Results go to {checkpoint_dir}_results/{split}/pretrained_results.json.

Usage:
    python baselines/pretrained/test_qwen_coder.py --checkpoint /path/to/small_test
    python baselines/pretrained/test_qwen_coder.py --checkpoint /path/to/small_test --split cr_test --limit-repos 1
"""

import argparse
import json
import os
from difflib import SequenceMatcher
from pathlib import Path

import torch


FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


def strip_fim_tokens(s: str) -> str:
    """Remove Qwen FIM special tokens from model output."""
    for tok in FIM_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def strip_comments(s: str) -> str:
    """Remove Python comments (everything after #)."""
    return s.split("#")[0].strip()


def normalize_for_match(s: str) -> str:
    """Normalize string for exact match comparison."""
    s = s.strip().rstrip(":, \t")
    s = s.replace(", ", ",").replace(" ,", ",")
    return " ".join(s.split())


def _pred_candidates(pred: str, ref: str) -> list[str]:
    """Return candidate pred strings to try for relaxed match."""
    candidates = [normalize_for_match(pred)]
    # If ref has no comma but pred does, try part before comma (model added extra)
    if "," not in ref and "," in pred:
        candidates.append(normalize_for_match(pred.split(",")[0]))
    # If ref is single token, try first token of pred (model added extra words)
    if len(ref.split()) == 1 and " " in pred:
        candidates.append(normalize_for_match(pred.split()[0]))
    return candidates


def exact_match(pred: str, ref: str) -> bool:
    """Exact match with relaxed postprocessing for common model overgeneration."""
    norm_ref = normalize_for_match(ref)
    return any(c == norm_ref for c in _pred_candidates(pred, ref))


def edit_similarity(pred: str, ref: str) -> float:
    """Edit similarity in [0, 1]. 1 = identical."""
    return SequenceMatcher(None, pred, ref).ratio()


def code_bleu_score(pred: str, ref: str, lang: str = "python") -> float:
    """Code BLEU if codebleu available, else 0."""
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([ref], [pred], lang=lang)
        return result["codebleu"]
    except Exception:
        return 0.0


def get_bos_id(tok):
    """Get BOS token id for generation (fallback: eos, then pad)."""
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return tok.pad_token_id


def load_split(splits_dir: Path, split_name: str, limit_repos: int | None = None) -> list:
    """Load split JSON (e.g. cr_test.json). Returns list of {repo, prefix, target}."""
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if limit_repos is not None and limit_repos > 0:
        repo_names = repo_names[:limit_repos]
    items = []
    for repo in repo_names:
        r = repos[repo]
        pairs = r.get("qna_pairs", [])
        for p in pairs:
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            items.append({"repo": repo, "prefix": prefix, "target": target})
    return items


def _prepare_input(prefix: str, tokenizer, bos_id: int, max_input_tokens: int) -> list:
    """Tokenize prefix, return input_ids list."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = [bos_id] + prefix_ids
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
    return input_ids


def invoke_batch(model, tokenizer, prefixes: list[str], bos_id: int, max_input_tokens: int,
                 max_new_tokens: int, device: str, pad_token_id: int) -> list[str]:
    """
    Run batched inference. Pads on left to max length in batch.
    Returns list of generated strings (one per prefix).
    """
    if not prefixes:
        return []
    all_input_ids = [_prepare_input(p, tokenizer, bos_id, max_input_tokens) for p in prefixes]
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


def main():
    ap = argparse.ArgumentParser(description="Evaluate pretrained Qwen2.5-Coder on split JSON")
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint dir - results go to {dir}_results/{split}/ (ignored if --output is set)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path (e.g. $SCRATCH/BASELINES/qwen_full.json)")
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Dir with cr_test.json, cr_val.json, etc.")
    ap.add_argument("--split", type=str, default="cr_test_structured",
                    help="Split to evaluate (default: cr_test_structured)")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="Max tokens to generate (default 128, use 64 for faster)")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for inference (default 8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of examples to evaluate")
    ap.add_argument("--limit-repos", type=int, default=None,
                    help="Use only first N repos from the split")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output is not None:
        results_path = Path(args.output).expanduser().resolve()
        results_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if args.checkpoint is None:
            raise ValueError("Either --output or --checkpoint must be specified")
        checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
        if checkpoint_dir.is_file():
            checkpoint_dir = checkpoint_dir.parent
        results_root = Path(str(checkpoint_dir) + "_results")
        output_dir = results_root / args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "pretrained_results.json"

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No items in {args.split}.json at {splits_dir}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    bos_id = get_bos_id(tok)
    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    n = len(items)
    batch_size = args.batch_size

    print(f"Evaluating {n} examples (batch_size={batch_size})...")
    for start in range(0, n, batch_size):
        batch_items = items[start : start + batch_size]
        if (start // batch_size + 1) % 10 == 0 or start == 0:
            print(f"  {min(start + batch_size, n)}/{n}...", flush=True)

        prefixes = [it["prefix"] for it in batch_items]
        preds = invoke_batch(
            model, tok, prefixes,
            bos_id=bos_id,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            pad_token_id=tok.pad_token_id,
        )

        for it, pred in zip(batch_items, preds):
            target = it["target"]
            pred = strip_fim_tokens(pred)
            # Stop at newline if target is single-line (common for assertions)
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

        # Write after every batch so partial results survive crashes/timeouts
        n_eval = len(entries)
        results = {
            "exact_match_pct": 100.0 * em_count / n_eval,
            "exact_match_count": em_count,
            "n": n_eval,
            "n_total": n,
            "code_bleu": bleu_sum / n_eval,
            "edit_similarity": edit_sum / n_eval,
            "entries": entries,
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n

    print("\n" + "=" * 60)
    print(f"Results on {args.split}.json (pretrained Qwen2.5-Coder-1.5B)")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
