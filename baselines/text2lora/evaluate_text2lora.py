#!/usr/bin/env python3
"""
Evaluate a trained Text2LoRA hypernetwork on RepoPeft benchmark.

For each test repo:
  1. Load the repo's task description
  2. Embed it with the same embedding model used in training
  3. Generate LoRA weights via the hypernetwork
  4. Set the generated weights on the PeftModel
  5. Run batched inference on the test pairs, compute metrics

Usage:
    python baselines/text2lora/evaluate_text2lora.py \
        --hypermod-dir text2lora/train_outputs/recon/hyper_lora/<run_name> \
        --split cr_test \
        --output $SCRATCH/BASELINES/text2lora_text_cr_test.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "text2lora" / "src"))

from evaluation.data_utils import get_default_splits_dir, load_split, get_bos_id, prepare_input_ids
from evaluation.metrics import (
    postprocess_prediction, strip_comments,
    exact_match, edit_similarity, code_bleu_score,
)

from peft.utils import set_peft_model_state_dict
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils import get_layers, embed_texts


def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def load_description(text2lora_dir: Path, repo_name: str) -> str:
    """Load first description for a repo from metadata.yaml."""
    meta_file = text2lora_dir / "tasks" / slug(repo_name) / "metadata.yaml"
    if not meta_file.exists():
        return f"Python repository {repo_name.split('/')[-1]}."
    meta = yaml.safe_load(meta_file.read_text())
    descs = meta.get("descriptions", [])
    return descs[0] if descs else f"Python repository {repo_name.split('/')[-1]}."


@torch.no_grad()
def generate_lora_for_repo(hypermod, emb_model, emb_tokenizer,
                           task_desc_format_fn, pooling_fn,
                           layer_indices, description, device):
    """Generate LoRA state_dict from a text description."""
    task_emb = embed_texts(
        [description], emb_model, emb_tokenizer,
        task_desc_format_fn, pooling_fn, device
    )
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
    return lora_sd


def invoke_batch(model, tokenizer, prefixes, bos_id, max_input_tokens,
                 max_new_tokens, device, pad_token_id):
    """Left-padded batched greedy inference."""
    if not prefixes:
        return []
    all_input_ids = [prepare_input_ids(p, tokenizer, bos_id, max_input_tokens)
                     for p in prefixes]
    max_len = max(len(ids) for ids in all_input_ids)

    padded, attn_masks = [], []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded.append([pad_token_id] * pad_len + ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))

    input_t = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

    out = model.generate(
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypermod-dir", required=True, type=Path,
                    help="Path to trained Text2LoRA checkpoint dir (contains hypermod.pt)")
    ap.add_argument("--split", default="cr_test")
    ap.add_argument("--splits-dir", type=str, default=get_default_splits_dir())
    ap.add_argument("--text2lora-dir", default=Path("text2lora"), type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--use-oracle", action="store_true",
                    help="Prepend oracle context to prefixes")
    ap.add_argument("--oracle-cache-dir", type=str, default=None,
                    help="Dir with pre-built oracle contexts")
    ap.add_argument("--max-oracle-tokens", type=int, default=None,
                    help="Compress oracle context to fit within this token budget")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load hypernetwork checkpoint ──────────────────────────────────────────
    # CWD must be text2lora/ for get_model_and_tokenizer (chat template) and
    # create_hypermod (oracle LoRA lookup) to find their relative paths.
    orig_cwd = os.getcwd()
    text2lora_abs = Path(orig_cwd) / args.text2lora_dir
    os.chdir(text2lora_abs)

    ckpt_path = Path(orig_cwd) / args.hypermod_dir / "hypermod.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    (
        hargs, hypermod, model, tokenizer,
        emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
    ) = load_hypermod_checkpoint(str(ckpt_path), device)

    os.chdir(orig_cwd)

    hypermod.eval()
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    layer_indices = torch.tensor(
        list(range(len(get_layers(model)))), dtype=torch.long, device=device
    )

    # ── Load test split ───────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if not items:
        print(f"No items found in {args.split}.json at {splits_dir}")
        return

    # ── Oracle context augmentation ────────────────────────────────────────
    if args.use_oracle:
        from evaluation.oracle_utils import (
            load_oracle_cache, lookup_oracle_context,
            augment_prefix_with_oracle, augment_prefix_with_compressed_oracle,
            get_default_oracle_cache_dir,
        )
        oracle_cache_dir = Path(args.oracle_cache_dir or get_default_oracle_cache_dir())
        use_compression = args.max_oracle_tokens is not None
        print(f"Augmenting prefixes with oracle context from {oracle_cache_dir}")
        if use_compression:
            print(f"  Compressing to {args.max_oracle_tokens} tokens")
        n_aug = 0
        for item in items:
            oracle_contexts = load_oracle_cache(oracle_cache_dir, item["repo"])
            if oracle_contexts:
                oracle_code = lookup_oracle_context(oracle_contexts, item.get("metadata", {}))
                if oracle_code:
                    if use_compression:
                        item["prefix"] = augment_prefix_with_compressed_oracle(
                            item["prefix"], oracle_code, tokenizer, args.max_oracle_tokens,
                        )
                    else:
                        item["prefix"] = augment_prefix_with_oracle(item["prefix"], oracle_code)
                    n_aug += 1
        print(f"  Augmented {n_aug}/{len(items)} pairs")

    # Group items by repo for per-repo LoRA generation
    repo_items = defaultdict(list)
    for item in items:
        repo_items[item["repo"]].append(item)

    repo_names = sorted(repo_items.keys())
    print(f"Split: {args.split}, repos: {len(repo_names)}, pairs: {len(items)}")

    # ── Evaluate per repo ─────────────────────────────────────────────────────
    bos_id = get_bos_id(tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    results_path = args.output.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for ri, repo_name in enumerate(repo_names):
        print(f"\n[{ri+1}/{len(repo_names)}] {repo_name}", flush=True)

        # Generate LoRA for this repo
        desc = load_description(text2lora_abs, repo_name)
        print(f"  desc: {desc[:100]}...")

        lora_sd = generate_lora_for_repo(
            hypermod, emb_model, emb_tokenizer,
            task_desc_format_fn, pooling_fn,
            layer_indices, desc, device
        )

        # Set generated LoRA weights directly on the PeftModel
        set_peft_model_state_dict(model, lora_sd)

        # Batched inference for all pairs in this repo
        repo_pairs = repo_items[repo_name]
        prefixes = [p["prefix"] for p in repo_pairs]
        targets = [p["target"] for p in repo_pairs]

        for start in range(0, len(prefixes), args.batch_size):
            batch_prefixes = prefixes[start:start + args.batch_size]
            batch_targets = targets[start:start + args.batch_size]

            preds = invoke_batch(
                model, tokenizer, batch_prefixes,
                bos_id=bos_id,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                device=str(device),
                pad_token_id=pad_token_id,
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

        # Per-repo progress
        repo_entries = [e for e in entries if e["repo"] == repo_name]
        if repo_entries:
            repo_em = sum(1 for e in repo_entries if e["exact_match"]) / len(repo_entries) * 100
            repo_es = sum(e["edit_similarity"] for e in repo_entries) / len(repo_entries)
            print(f"  {len(repo_entries)} pairs | EM={repo_em:.1f}% ES={repo_es:.3f}")

        # Save after every repo for crash recovery
        n_eval = len(entries)
        results = {
            "method": "text2lora_text",
            "split": args.split,
            "model": "Qwen/Qwen2.5-Coder-1.5B",
            "hypermod_dir": str(args.hypermod_dir),
            "exact_match_pct": 100.0 * em_count / n_eval,
            "exact_match_count": em_count,
            "n": n_eval,
            "n_total": len(items),
            "code_bleu": bleu_sum / n_eval,
            "edit_similarity": edit_sum / n_eval,
            "entries": entries,
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Final summary ─────────────────────────────────────────────────────────
    n_eval = len(entries)
    print(f"\n{'='*60}")
    print(f"Text2LoRA (text-conditioned) on {args.split}")
    print(f"{'='*60}")
    print(f"  Exact Match:     {100.0 * em_count / n_eval:.2f}% ({em_count}/{n_eval})")
    print(f"  Code BLEU:       {bleu_sum / n_eval:.4f}")
    print(f"  Edit Similarity: {edit_sum / n_eval:.4f}")
    print(f"{'='*60}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
