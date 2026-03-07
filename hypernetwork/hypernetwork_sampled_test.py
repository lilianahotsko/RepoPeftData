#!/usr/bin/env python3
"""
Evaluate hypernetwork checkpoint on split JSON files.

Metrics: exact match, code BLEU, edit similarity.
Results go to {checkpoint_dir}_results/{split}/results.json (e.g. small_test_results/cr_test/).

Usage:
    python hypernetwork/hypernetwork_sampled_test.py --checkpoint /path/to/hypernet_state.pt
    python hypernetwork/hypernetwork_sampled_test.py --checkpoint /path/to/output_dir  # uses last checkpoint
    python hypernetwork/hypernetwork_sampled_test.py --checkpoint ./output --split cr_val
    python hypernetwork/hypernetwork_sampled_test.py --checkpoint ./output --split cr_test --limit 100 --limit-repos 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Import from hypernetwork_sampled (after torch to avoid trl deps for --help)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.data_utils import get_default_splits_dir, load_split_with_embeddings
from evaluation.metrics import (
    strip_fim_tokens, strip_comments, postprocess_prediction,
    exact_match, edit_similarity, code_bleu_score,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate hypernetwork on split JSON (cr_test, cr_val, etc.)")
    default_dataset = get_default_splits_dir()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to hypernet_state.pt or output dir (uses last checkpoint)")
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Dir with cr_test.json, cr_val.json, etc.")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=16384,
                    help="Max input tokens (default 16384 for parity with baselines)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of examples to evaluate")
    ap.add_argument("--limit-repos", type=int, default=None,
                    help="Use only first N repos from the split")
    ap.add_argument("--repo", type=str, default=None,
                    help="Evaluate only this repo (e.g. 0xricksanchez/like-dbg)")
    ap.add_argument("--split", type=str, default="cr_test",
                    help="Split to evaluate (default: cr_test). Ignored if --splits is set.")
    ap.add_argument("--splits", type=str, nargs="+", default=None,
                    help="Multiple splits to evaluate (e.g. cr_test ir_test)")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    splits_to_run = args.splits if args.splits else [args.split]

    # Lazy imports (hypernetwork_sampled is in same dir when run as python hypernetwork/hypernetwork_sampled_test.py)
    import hypernetwork_sampled as _hn
    Hypernetwork = _hn.Hypernetwork
    LoRA = _hn.LoRA
    get_module_specs = _hn.get_module_specs
    replace_with_lora = _hn.replace_with_lora
    inject_lora_weights = _hn.inject_lora_weights
    get_bos_id = _hn.get_bos_id
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # If directory: prefer hypernet_best.pt, else hypernet_state.pt, else checkpoint-N/hypernet_state.pt
    if checkpoint_path.is_dir():
        best_path = checkpoint_path / "hypernet_best.pt"
        final_path = checkpoint_path / "hypernet_state.pt"
        if best_path.exists():
            checkpoint_path = best_path
            print(f"Using best hypernet: {checkpoint_path}")
        elif final_path.exists():
            checkpoint_path = final_path
            print(f"Using final hypernet: {checkpoint_path}")
        else:
            ckpt_dirs = sorted(
                (p for p in checkpoint_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
                key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else -1,
            )
            if ckpt_dirs:
                last_ckpt = ckpt_dirs[-1] / "hypernet_state.pt"
                if last_ckpt.exists():
                    checkpoint_path = last_ckpt
                    print(f"Using last checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No hypernet_state.pt in {ckpt_dirs[-1]}")
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_path}. "
                    "Expected hypernet_best.pt, hypernet_state.pt, or checkpoint-N/hypernet_state.pt"
                )

    # Output dir: checkpoint parent dir + "_results", with split subdir
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

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    rank = 16
    alpha = 32
    replace_with_lora(model, specs, r=rank, alpha=alpha)

    print(f"Loading hypernetwork from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hypernet_config = ckpt["hypernet_config"]
    hidden_dim = hypernet_config.get("hidden_dim", 512)
    hypernet = Hypernetwork(
        input_dim=hypernet_config["input_dim"],
        module_specs=ckpt["module_specs"],
        hidden_dim=hidden_dim,
        rank=hypernet_config["rank"],
    )
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.to(args.device)
    hypernet.eval()

    bos_id = get_bos_id(tok)

    for split in splits_to_run:
        output_dir = results_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        items = load_split_with_embeddings(splits_dir, split, limit_repos=args.limit_repos, repo_filter=args.repo)
        if args.limit is not None and args.limit > 0:
            items = items[: args.limit]
        if not items:
            print(f"[split={split}] No items in {split}.json at {splits_dir}, skipping")
            continue

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

            # Encode prefix, left truncate if needed
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            input_ids = [bos_id] + prefix_ids
            if len(input_ids) > args.max_input_tokens:
                input_ids = input_ids[-args.max_input_tokens :]

            ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(args.device)
            ctx = F.normalize(ctx, p=2, dim=-1)
            with torch.no_grad():
                h_out = hypernet(ctx)
            inject_lora_weights(model, specs, h_out, batch_index=0)

            input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                out = model.generate(
                    input_t,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][len(input_ids) :].tolist()
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
                "repo": it["repo"],
                "expected": target_clean,
                "got": pred_clean,
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
        print(f"Results on {split}.json")
        print("=" * 60)
        print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
        print(f"  Code BLEU:       {code_bleu_avg:.4f}")
        print(f"  Edit Similarity: {edit_sim_avg:.4f}")
        print("=" * 60)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
