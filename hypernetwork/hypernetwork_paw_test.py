#!/usr/bin/env python3
"""
Evaluate PAW-style hypernetwork (LoraMapper) checkpoint on split JSON files.

Metrics: exact match, code BLEU, edit similarity.
Results go to {checkpoint_dir}_results/{split}/results.json.

Usage:
    python hypernetwork/hypernetwork_paw_test.py --checkpoint /path/to/lora_mapper_best.pt
    python hypernetwork/hypernetwork_paw_test.py --checkpoint /path/to/output_dir
    python hypernetwork/hypernetwork_paw_test.py --checkpoint ./output --split cr_test --limit 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data_utils import get_default_splits_dir, load_split_with_embeddings
from evaluation.metrics import (
    postprocess_prediction, strip_comments,
    exact_match, edit_similarity, code_bleu_score,
)


def resolve_checkpoint(base_path: Path) -> Path:
    """Find the best lora_mapper checkpoint in a directory."""
    if base_path.is_file():
        return base_path

    candidates = [
        base_path / "lora_mapper_best.pt",
        base_path / "lora_mapper_state.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"Using checkpoint: {c}")
            return c

    ckpt_dirs = sorted(
        (p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else -1,
    )
    if ckpt_dirs:
        last = ckpt_dirs[-1] / "lora_mapper_state.pt"
        if last.exists():
            print(f"Using last checkpoint: {last}")
            return last

    raise FileNotFoundError(
        f"No lora_mapper checkpoint in {base_path}. "
        "Expected lora_mapper_best.pt, lora_mapper_state.pt, or checkpoint-N/lora_mapper_state.pt"
    )


def main():
    ap = argparse.ArgumentParser(description="Evaluate PAW-style hypernetwork on split JSON")
    default_dataset = get_default_splits_dir()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to lora_mapper_best.pt or output dir")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--repo", type=str, default=None)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--splits", type=str, nargs="+", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Prepend oracle context to prefixes at eval time")
    ap.add_argument("--oracle-cache-dir", type=str, default=None)
    args = ap.parse_args()

    splits_to_run = args.splits if args.splits else [args.split]

    import hypernetwork_paw as _paw
    LoraMapper = _paw.LoraMapper
    discover_target_modules = _paw.discover_target_modules
    apply_lora_hooks = _paw.apply_lora_hooks
    remove_lora_hooks = _paw.remove_lora_hooks
    get_bos_id = _paw.get_bos_id
    from transformers import AutoModelForCausalLM, AutoTokenizer

    oracle_cache_dir = None
    if args.use_oracle:
        from evaluation.oracle_utils import (
            get_default_oracle_cache_dir, load_oracle_cache,
            lookup_oracle_context, augment_prefix_with_oracle,
        )
        oracle_cache_dir = Path(args.oracle_cache_dir or get_default_oracle_cache_dir()).expanduser().resolve()
        if not oracle_cache_dir.exists():
            raise FileNotFoundError(f"Oracle cache not found: {oracle_cache_dir}")
        print(f"Using oracle context from {oracle_cache_dir}")

    checkpoint_path = resolve_checkpoint(Path(args.checkpoint).expanduser().resolve())

    checkpoint_dir = checkpoint_path.parent
    results_root = Path(str(checkpoint_dir) + "_results")
    splits_dir = Path(args.splits_dir).expanduser().resolve()

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

    target_module_names = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    target_modules_dict, module_dims, num_layers = discover_target_modules(model, target_module_names)
    print(f"Discovered {len(target_modules_dict)} target modules across {num_layers} layers")

    print(f"Loading LoraMapper from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["lora_mapper_config"]
    lora_mapper = LoraMapper(
        input_dim=cfg["input_dim"],
        num_layers=cfg["num_layers"],
        module_dims=cfg["module_dims"],
        hidden_dim=cfg["hidden_dim"],
        rank=cfg["rank"],
        alpha=cfg["rank"] * cfg["lora_scaling"],
        num_bases=cfg["num_bases"],
    )
    lora_mapper.load_state_dict(ckpt["lora_mapper_state_dict"])
    lora_mapper.to(args.device)
    lora_mapper.eval()
    n_params = sum(p.numel() for p in lora_mapper.parameters())
    print(f"LoraMapper: {n_params:,} params, scaling={lora_mapper.lora_scaling}")

    bos_id = get_bos_id(tok)

    for split in splits_to_run:
        output_dir = results_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        items = load_split_with_embeddings(splits_dir, split, limit_repos=args.limit_repos, repo_filter=args.repo)
        if args.limit is not None and args.limit > 0:
            items = items[:args.limit]
        if not items:
            print(f"[split={split}] No items, skipping")
            continue

        if oracle_cache_dir:
            from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context, augment_prefix_with_oracle
            n_aug = 0
            for it in items:
                oc = load_oracle_cache(oracle_cache_dir, it["repo"])
                if oc:
                    code = lookup_oracle_context(oc, it.get("metadata", {}))
                    if code:
                        it["prefix"] = augment_prefix_with_oracle(it["prefix"], code)
                        n_aug += 1
            print(f"  Oracle: augmented {n_aug}/{len(items)} ({100*n_aug/len(items):.1f}%)")

        em_count = 0
        bleu_sum = 0.0
        edit_sum = 0.0
        entries = []
        n = len(items)

        print(f"\n[split={split}] Evaluating {n} examples...")
        for i, it in enumerate(items):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  {i + 1}/{n}...", flush=True)

            prefix = it["prefix"]
            target = it["target"]
            emb = it["embedding"]

            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            input_ids = [bos_id] + prefix_ids
            if len(input_ids) > args.max_input_tokens:
                input_ids = input_ids[-args.max_input_tokens:]

            ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(args.device)
            ctx = F.normalize(ctx, p=2, dim=-1)

            with torch.no_grad():
                lora_params = lora_mapper(ctx)

            hooks = apply_lora_hooks(target_modules_dict, lora_params, lora_mapper.lora_scaling)
            try:
                input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
                with torch.no_grad():
                    out = model.generate(
                        input_t,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
            finally:
                remove_lora_hooks(hooks)

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
                "repo": it["repo"],
                "expected": target_clean,
                "got": pred_clean,
                "got_raw": pred_raw,
                "exact_match": em,
                "code_bleu": bleu,
                "edit_similarity": edit_sim,
            })

            if (i + 1) % 50 == 0 or (i + 1) == n:
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
                results_path = output_dir / "results.json"
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        exact_match_pct = 100.0 * em_count / n
        code_bleu_avg = bleu_sum / n
        edit_sim_avg = edit_sum / n
        print("\n" + "=" * 60)
        print(f"Results on {split}.json (PAW-style)")
        print("=" * 60)
        print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
        print(f"  Code BLEU:       {code_bleu_avg:.4f}")
        print(f"  Edit Similarity: {edit_sim_avg:.4f}")
        print("=" * 60)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
