#!/usr/bin/env python3
"""
Evaluate a code-conditioned Text2LoRA hypernetwork on RepoPeft benchmark.

Same as evaluate_text2lora.py but uses pre-computed code embeddings
instead of embedding text descriptions.

Usage:
    python baselines/text2lora/evaluate_text2lora_code.py \
        --hypermod-dir text2lora/train_outputs/recon/hyper_lora/<code_run> \
        --code-emb-path $SCRATCH/TEXT2LORA_DATA/code_embeddings.pt \
        --split cr_test \
        --output $SCRATCH/BASELINES/text2lora_code_cr_test.json
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

from peft import PeftConfig, get_peft_config
from peft.utils import set_peft_model_state_dict
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.utils import get_layers
from hyper_llm_modulator.utils.model_loading import get_model_and_tokenizer


def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


@torch.no_grad()
def generate_lora_for_repo(hypermod, code_emb, layer_indices, device):
    """Generate LoRA state_dict from a pre-computed code embedding."""
    task_emb = code_emb.unsqueeze(0).to(device)
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
    ap.add_argument("--hypermod-dir", required=True, type=Path)
    ap.add_argument("--code-emb-path", required=True, type=Path,
                    help="Path to code_embeddings.pt")
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

    # ── Load pre-computed code embeddings ─────────────────────────────────────
    print(f"Loading code embeddings: {args.code_emb_path}")
    code_embs = torch.load(args.code_emb_path, map_location="cpu", weights_only=True)
    task_emb_size = next(iter(code_embs.values())).shape[-1]
    print(f"  {len(code_embs)} repo embeddings, dim={task_emb_size}")

    # ── Load hypernetwork checkpoint manually ─────────────────────────────────
    # Can't use load_hypermod_checkpoint because it infers task_emb_size from
    # the embedding model (1536 for base LLM), but we need 2048 from code embs.
    orig_cwd = os.getcwd()
    text2lora_abs = Path(orig_cwd) / args.text2lora_dir
    os.chdir(text2lora_abs)

    hypermod_dir = Path(orig_cwd) / args.hypermod_dir
    ckpt_path = hypermod_dir / "hypermod.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    hargs = argparse.Namespace(
        **yaml.safe_load(open(hypermod_dir / "args.yaml"))
    )
    peft_config = get_peft_config(
        PeftConfig.from_json_file(str(hypermod_dir / "adapter_config.json"))
    )
    state_dict = torch.load(str(ckpt_path), map_location=device)

    model, tokenizer = get_model_and_tokenizer(
        hargs.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )

    os.chdir(orig_cwd)

    layer_indices = torch.tensor(
        list(range(len(get_layers(model)))), dtype=torch.long, device=device
    )

    hypermod = create_hypermod(
        hargs, peft_config.peft_type.lower(), device, model,
        layer_indices, task_emb_size, from_scratch=False
    )
    info = hypermod.load_state_dict(state_dict, strict=False)
    print(f"Loaded hypermod state dict: {info}")

    hypermod.eval()
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # ── Load test split ───────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    items = load_split(splits_dir, args.split, limit_repos=args.limit_repos)
    if not items:
        print(f"No items found in {args.split}.json")
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
    n_skipped_repos = 0
    results_path = args.output.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for ri, repo_name in enumerate(repo_names):
        repo_slug = slug(repo_name)
        if repo_slug not in code_embs:
            print(f"[{ri+1}/{len(repo_names)}] {repo_name} — no code embedding, skipping")
            n_skipped_repos += 1
            continue

        print(f"[{ri+1}/{len(repo_names)}] {repo_name}", flush=True)

        code_emb = code_embs[repo_slug].squeeze(0)
        lora_sd = generate_lora_for_repo(hypermod, code_emb, layer_indices, device)
        set_peft_model_state_dict(model, lora_sd)

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

        repo_entries = [e for e in entries if e["repo"] == repo_name]
        if repo_entries:
            repo_em = sum(1 for e in repo_entries if e["exact_match"]) / len(repo_entries) * 100
            repo_es = sum(e["edit_similarity"] for e in repo_entries) / len(repo_entries)
            print(f"  {len(repo_entries)} pairs | EM={repo_em:.1f}% ES={repo_es:.3f}")

        # Save after every repo
        n_eval = len(entries)
        if n_eval > 0:
            results = {
                "method": "text2lora_code",
                "split": args.split,
                "model": "Qwen/Qwen2.5-Coder-1.5B",
                "hypermod_dir": str(args.hypermod_dir),
                "exact_match_pct": 100.0 * em_count / n_eval,
                "exact_match_count": em_count,
                "n": n_eval,
                "n_total": len(items),
                "n_skipped_repos": n_skipped_repos,
                "code_bleu": bleu_sum / n_eval,
                "edit_similarity": edit_sum / n_eval,
                "entries": entries,
            }
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_eval = len(entries)
    print(f"\n{'='*60}")
    print(f"Text2LoRA (code-conditioned) on {args.split}")
    print(f"{'='*60}")
    print(f"  Exact Match:     {100.0 * em_count / n_eval:.2f}% ({em_count}/{n_eval})")
    print(f"  Code BLEU:       {bleu_sum / n_eval:.4f}")
    print(f"  Edit Similarity: {edit_sum / n_eval:.4f}")
    print(f"  Skipped repos:   {n_skipped_repos}")
    print(f"{'='*60}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
