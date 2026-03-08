#!/usr/bin/env python3
"""
Incremental adaptation experiment: evaluate how performance changes as we
add more file-level LoRAs incrementally.

For each repo in the test set:
1. Order files by relevance to the test prefix (cosine similarity)
2. Start with 0 files, add one at a time
3. Measure performance at each step

This demonstrates the "continuous adaptation" story.

Usage:
    python hypernetwork/eval_incremental.py --checkpoint $SCRATCH/.../composable --split cr_test --limit-repos 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments
from evaluation.data_utils import get_default_splits_dir, get_bos_id


def main():
    ap = argparse.ArgumentParser(description="Incremental adaptation evaluation")
    default_dataset = get_default_splits_dir()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--max-files-to-test", type=int, default=20,
                    help="Max number of files to incrementally add")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    # Load split with file embeddings
    split_path = splits_dir / f"{args.split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})

    # Filter repos with file embeddings
    repo_names = sorted(repos.keys())
    if args.limit_repos:
        repo_names = repo_names[:args.limit_repos]

    repos_with_files = {}
    for rn in repo_names:
        r = repos[rn]
        fe = r.get("file_embeddings")
        emb = r.get("embedding")
        pairs = r.get("qna_pairs", [])
        if fe and emb and pairs:
            valid_pairs = [p for p in pairs if p.get("prefix") and p.get("target")
                          and not p["target"].lstrip().startswith(",")]
            if valid_pairs:
                repos_with_files[rn] = {
                    "pairs": valid_pairs,
                    "embedding": emb,
                    "file_embeddings": fe,
                }

    if not repos_with_files:
        print("No repos with file embeddings found. Run embed_repos.py with --overwrite first.")
        return

    print(f"Found {len(repos_with_files)} repos with file embeddings")

    # Load model and hypernetwork
    from hypernetwork.hypernetwork_composable import ComposableHypernetwork
    from hypernetwork.hypernetwork_sampled import get_module_specs, replace_with_lora, LoRA

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

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    rank = 16
    alpha = 32
    replace_with_lora(model, specs, r=rank, alpha=alpha)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.is_dir():
        for name in ["hypernet_best.pt", "hypernet_state.pt"]:
            if (checkpoint_path / name).exists():
                checkpoint_path = checkpoint_path / name
                break

    print(f"Loading composable hypernetwork from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hconfig = ckpt["hypernet_config"]

    hypernet = ComposableHypernetwork(
        input_dim=hconfig["input_dim"],
        module_specs=ckpt["module_specs"],
        hidden_dim=hconfig.get("hidden_dim", 512),
        rank=hconfig["rank"],
        composition=hconfig.get("composition", "weighted"),
    )
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.to(args.device).eval()

    bos_id = get_bos_id(tok)

    # Incremental evaluation
    results_by_n_files = {}  # n_files -> {em_count, total, edit_sum}
    max_k = args.max_files_to_test

    for repo_name, rdata in repos_with_files.items():
        file_embs = rdata["file_embeddings"]
        pairs = rdata["pairs"]
        if args.limit:
            pairs = pairs[:args.limit]

        print(f"\n{repo_name}: {len(file_embs)} files, {len(pairs)} pairs")

        for pi, p in enumerate(pairs):
            prefix = p["prefix"]
            target = p["target"]

            # Encode prefix
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            input_ids = [bos_id] + prefix_ids
            if len(input_ids) > args.max_input_tokens:
                input_ids = input_ids[-args.max_input_tokens:]

            input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)

            # Test with 0, 1, 2, ... K files
            k_range = range(0, min(len(file_embs) + 1, max_k + 1))
            for k in k_range:
                if k == 0:
                    # Use repo-level embedding (no file composition)
                    ctx = torch.tensor(rdata["embedding"], dtype=torch.float32).unsqueeze(0).to(args.device)
                    ctx = F.normalize(ctx, p=2, dim=-1)
                    with torch.no_grad():
                        h_out = hypernet.forward_single(ctx)
                else:
                    # Use first k file embeddings
                    selected_embs = [fe["embedding"] for fe in file_embs[:k]]
                    file_ctx = torch.tensor(selected_embs, dtype=torch.float32).unsqueeze(0).to(args.device)
                    ctx = torch.tensor(rdata["embedding"], dtype=torch.float32).unsqueeze(0).to(args.device)
                    ctx = F.normalize(ctx, p=2, dim=-1)
                    with torch.no_grad():
                        h_out = hypernet(ctx, file_embeddings=file_ctx)

                # Inject and generate
                named = dict(model.named_modules())
                for full_name, _, t, _, _ in specs:
                    A0 = h_out["A"][t][0].to(device=args.device)
                    B0 = h_out["B"][t][0].to(device=args.device)
                    named[full_name].set_lora_weights(A0, B0)

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
                edit_sim = edit_similarity(pred_clean, target_clean)

                if k not in results_by_n_files:
                    results_by_n_files[k] = {"em_count": 0, "total": 0, "edit_sum": 0.0}
                results_by_n_files[k]["em_count"] += int(em)
                results_by_n_files[k]["total"] += 1
                results_by_n_files[k]["edit_sum"] += edit_sim

    # Print results
    print("\n" + "=" * 60)
    print("Incremental Adaptation Results")
    print("=" * 60)
    print(f"{'N Files':>8} | {'EM%':>8} | {'EditSim':>8} | {'N':>6}")
    print("-" * 40)

    curve_data = []
    for k in sorted(results_by_n_files.keys()):
        r = results_by_n_files[k]
        em_pct = 100.0 * r["em_count"] / r["total"] if r["total"] > 0 else 0
        edit_avg = r["edit_sum"] / r["total"] if r["total"] > 0 else 0
        print(f"{k:>8} | {em_pct:>7.2f}% | {edit_avg:>8.4f} | {r['total']:>6}")
        curve_data.append({
            "n_files": k,
            "exact_match_pct": em_pct,
            "edit_similarity": edit_avg,
            "n_examples": r["total"],
        })

    # Save results
    output_results = {
        "method": "incremental_adaptation",
        "split": args.split,
        "curve": curve_data,
    }

    if args.output:
        results_path = Path(args.output).expanduser().resolve()
    else:
        results_path = checkpoint_path.parent / f"incremental_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
