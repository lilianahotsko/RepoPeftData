#!/usr/bin/env python3
"""
Progressive Internalization: evaluate how performance improves as we
progressively "internalize" repository files by generating and accumulating
per-file LoRA adapters.

Two modes:
  A) re-embed: recompute weighted-mean+max repo embedding from files 1..k
               and generate a single LoRA from the updated embedding.
  B) accumulate: generate a separate LoRA for each file and average them.
               Adding a new file is O(1) — no recomputation of previous LoRAs.

For each test repo we:
  1. Compute per-file embeddings on-the-fly (Qwen3-Embedding-0.6B).
  2. Order files by token count (largest first = most informative).
  3. For k = 0, 1, 2, …, K files, measure EM and EditSim.
  4. Report the performance curve.

Usage:
    python hypernetwork/eval_progressive.py \
        --checkpoint $SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/no_oracle \
        --split cr_test --limit-repos 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    postprocess_prediction, exact_match, edit_similarity,
    code_bleu_score, strip_comments,
)
from evaluation.data_utils import get_default_splits_dir, get_bos_id
from create_dataset.embed_repos import (
    iter_source_files, read_text_file, embed_texts,
    pool_file_embeddings, pool_repo_embeddings_weighted,
    DEFAULT_EXTS, chunk_token_ids,
)


def compute_file_embeddings(
    repo_dir: Path,
    emb_model,
    emb_tokenizer,
    device: str,
    chunk_tokens: int = 4096,
    chunk_overlap: int = 512,
    max_files: int = 200,
    max_file_bytes: int = 2_000_000,
):
    """Compute per-file embeddings for a repository, returning a list of
    (relative_path, embedding_1024d, token_count, line_count) tuples.
    """
    files = iter_source_files(repo_dir, DEFAULT_EXTS, max_files, max_file_bytes)
    results = []
    for fp in files:
        text = read_text_file(fp)
        if not text:
            continue
        rel = str(fp.relative_to(repo_dir)).replace("\\", "/")
        lines = text.splitlines()
        ids = emb_tokenizer.encode(text, add_special_tokens=False)
        tok_count = len(ids)
        windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=chunk_overlap)
        if not windows:
            continue
        chunks = [emb_tokenizer.decode(w, skip_special_tokens=True) for w in windows]
        chunk_embs = embed_texts(emb_model, emb_tokenizer, chunks, device, batch_size=32, max_length=chunk_tokens)
        fvec = pool_file_embeddings(chunk_embs)
        if fvec is None:
            continue
        results.append((rel, fvec, tok_count, len(lines)))
    results.sort(key=lambda x: -x[2])
    return results


def make_repo_embedding(file_embs, tok_counts, line_counts, paths,
                        alpha_mean=0.6, beta_max=1.4):
    """Build a 2048-dim repo embedding from a set of file embeddings using
    the same weighted mean+max procedure as embed_repos.py."""
    stacked = torch.stack(file_embs, dim=0)
    tok_t = torch.tensor(tok_counts, dtype=torch.int64)
    line_t = torch.tensor(line_counts, dtype=torch.int64)
    return pool_repo_embeddings_weighted(
        file_embs=stacked,
        file_token_counts=tok_t,
        file_line_counts=line_t,
        file_paths=paths,
        a_distinct=1.0, b_size=0.3, tau=0.15,
        alpha_mean=alpha_mean, beta_max=beta_max,
        aggregation="concat",
    )


def file_to_repo_embedding(file_emb, alpha_mean=0.6, beta_max=1.4):
    """Construct a 2048-dim 'repo embedding' from a single 1024-dim file
    embedding, matching the concat(mean, max) format."""
    return torch.cat([alpha_mean * file_emb, beta_max * file_emb], dim=0)


def generate_lora(hypernet, repo_emb_2048, device):
    """Run the hypernetwork on a single repo embedding and return the output."""
    ctx = repo_emb_2048.unsqueeze(0).to(device=device, dtype=torch.float32)
    ctx = F.normalize(ctx, p=2, dim=-1)
    with torch.no_grad():
        return hypernet(ctx)


def accumulate_loras(lora_list, types):
    """Average a list of hypernetwork outputs into a single output."""
    n = len(lora_list)
    if n == 0:
        return None
    if n == 1:
        return lora_list[0]
    A_acc = {}
    B_acc = {}
    for t in types:
        A_acc[t] = sum(h["A"][t] for h in lora_list) / n
        B_acc[t] = sum(h["B"][t] for h in lora_list) / n
    return {"A": A_acc, "B": B_acc}


def inject_and_generate(model, specs, h_out, input_ids, tok, device, max_new_tokens):
    """Inject LoRA weights and generate a prediction."""
    named = dict(model.named_modules())
    for full_name, _, t, _, _ in specs:
        A0 = h_out["A"][t][0].to(device=device)
        B0 = h_out["B"][t][0].to(device=device)
        named[full_name].set_lora_weights(A0, B0)

    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_t, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][len(input_ids):].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True)


def zero_lora(model, specs):
    """Set all LoRA weights to zero (pretrained baseline)."""
    named = dict(model.named_modules())
    for full_name, _, t, in_f, out_f in specs:
        mod = named[full_name]
        mod.set_lora_weights(
            torch.zeros(mod.r, in_f, device=next(model.parameters()).device),
            torch.zeros(out_f, mod.r, device=next(model.parameters()).device),
        )


def main():
    ap = argparse.ArgumentParser(description="Progressive internalization evaluation")
    default_dataset = get_default_splits_dir()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--max-files", type=int, default=30,
                    help="Max number of files to progressively add")
    ap.add_argument("--file-steps", type=str, default="0,1,2,3,5,10,15,20,30",
                    help="Comma-separated list of k values (number of files) to evaluate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--limit-pairs-per-repo", type=int, default=None,
                    help="Limit test pairs per repo (for speed)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--mode", type=str, default="both",
                    choices=["re-embed", "accumulate", "both"],
                    help="Which progressive strategy to evaluate")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    import hypernetwork_sampled as _hn

    Hypernetwork = _hn.Hypernetwork
    get_module_specs = _hn.get_module_specs
    replace_with_lora = _hn.replace_with_lora
    inject_lora_weights = _hn.inject_lora_weights

    file_steps = sorted(set(int(x) for x in args.file_steps.split(",")))

    # ── Load split data ──────────────────────────────────────────
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = splits_dir / "repositories"
    split_path = splits_dir / f"{args.split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    data = json.loads(split_path.read_text())
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if args.limit_repos:
        repo_names = repo_names[:args.limit_repos]

    # ── Load embedding model (small, 0.6B) ──────────────────────
    print(f"Loading embedding model: {args.embed_model}")
    emb_tokenizer = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)
    emb_model = AutoModel.from_pretrained(args.embed_model, dtype=torch.bfloat16)
    emb_model.to(args.device).eval()

    # ── Load code LLM + hypernetwork ─────────────────────────────
    print(f"Loading code LLM: {args.model_name}")
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
    replace_with_lora(model, specs, r=16, alpha=32)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.is_dir():
        for name in ["hypernet_best.pt", "hypernet_state.pt"]:
            if (checkpoint_path / name).exists():
                checkpoint_path = checkpoint_path / name
                break

    print(f"Loading hypernetwork from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hconfig = ckpt["hypernet_config"]
    hypernet = Hypernetwork(
        input_dim=hconfig["input_dim"],
        module_specs=ckpt["module_specs"],
        hidden_dim=hconfig.get("hidden_dim", 512),
        rank=hconfig["rank"],
    )
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.to(args.device).eval()

    bos_id = get_bos_id(tok)
    types = hconfig["types"]

    # Unload embedding model from GPU to free memory
    emb_device_for_later = args.device

    # ── Main evaluation loop ─────────────────────────────────────
    # results[mode][k] = {em_count, total, edit_sum, bleu_sum}
    modes_to_run = []
    if args.mode in ("re-embed", "both"):
        modes_to_run.append("re-embed")
    if args.mode in ("accumulate", "both"):
        modes_to_run.append("accumulate")

    results = {m: defaultdict(lambda: {"em_count": 0, "total": 0, "edit_sum": 0.0, "bleu_sum": 0.0})
               for m in modes_to_run}

    total_repos = len(repo_names)
    for ri, repo_name in enumerate(repo_names):
        rdata = repos[repo_name]
        pairs = rdata.get("qna_pairs", [])
        pairs = [p for p in pairs if p.get("prefix") and p.get("target")
                 and not p["target"].lstrip().startswith(",")]
        if not pairs:
            continue
        if args.limit_pairs_per_repo:
            pairs = pairs[:args.limit_pairs_per_repo]

        # Resolve repo directory (author/repo structure)
        parts = repo_name.split("/")
        if len(parts) == 2:
            repo_dir = repos_root / parts[0] / parts[1]
        else:
            repo_dir = repos_root / repo_name
        if not repo_dir.exists():
            print(f"  [{ri+1}/{total_repos}] {repo_name}: repo dir not found, skipping")
            continue

        # Compute file embeddings on the fly
        print(f"  [{ri+1}/{total_repos}] {repo_name}: computing file embeddings...", flush=True)
        file_data = compute_file_embeddings(
            repo_dir, emb_model, emb_tokenizer, emb_device_for_later,
            max_files=args.max_files,
        )
        if not file_data:
            print(f"    No files embedded, skipping")
            continue

        n_files = len(file_data)
        file_embs = [fd[1] for fd in file_data]
        file_toks = [fd[2] for fd in file_data]
        file_lines = [fd[3] for fd in file_data]
        file_paths = [fd[0] for fd in file_data]

        print(f"    {n_files} files, {len(pairs)} test pairs")

        # Pre-generate per-file LoRAs for accumulate mode
        per_file_loras = []
        if "accumulate" in modes_to_run:
            for fi, femb in enumerate(file_embs):
                repo_emb_from_file = file_to_repo_embedding(femb)
                lora = generate_lora(hypernet, repo_emb_from_file, args.device)
                per_file_loras.append(lora)

        # Pre-compute re-embed repo embeddings for each k
        re_embed_loras = {}
        if "re-embed" in modes_to_run:
            for k in file_steps:
                if k == 0 or k > n_files:
                    continue
                repo_emb = make_repo_embedding(
                    file_embs[:k], file_toks[:k], file_lines[:k], file_paths[:k],
                )
                if repo_emb is not None:
                    re_embed_loras[k] = generate_lora(hypernet, repo_emb, args.device)

        # Evaluate each pair at each k
        for pi, p in enumerate(pairs):
            prefix = p["prefix"]
            target = p["target"]
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            input_ids = [bos_id] + prefix_ids
            if len(input_ids) > args.max_input_tokens:
                input_ids = input_ids[-args.max_input_tokens:]

            target_clean = strip_comments(target)

            # k=0 (pretrained baseline) — shared across modes
            zero_pred = None

            for k in file_steps:
                if k > n_files:
                    continue

                for mode in modes_to_run:
                    if k == 0:
                        if zero_pred is None:
                            zero_lora(model, specs)
                            input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
                            with torch.no_grad():
                                out = model.generate(
                                    input_t, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id,
                                )
                            gen_ids = out[0][len(input_ids):].tolist()
                            zero_pred = tok.decode(gen_ids, skip_special_tokens=True)
                        pred = zero_pred
                    elif mode == "re-embed":
                        h_out = re_embed_loras.get(k)
                        if h_out is None:
                            continue
                        pred = inject_and_generate(
                            model, specs, h_out,
                            input_ids, tok, args.device, args.max_new_tokens,
                        )
                    elif mode == "accumulate":
                        h_out = accumulate_loras(per_file_loras[:k], types)
                        if h_out is None:
                            continue
                        pred = inject_and_generate(
                            model, specs, h_out,
                            input_ids, tok, args.device, args.max_new_tokens,
                        )
                    else:
                        continue

                    pred_clean = postprocess_prediction(pred, target)
                    em = exact_match(pred_clean, target_clean)
                    edit_sim = edit_similarity(pred_clean, target_clean)

                    r = results[mode][k]
                    r["em_count"] += int(em)
                    r["total"] += 1
                    r["edit_sum"] += edit_sim

    # ── Print results ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Progressive Internalization Results")
    print("=" * 70)

    curve_data = {}
    for mode in modes_to_run:
        print(f"\n  Mode: {mode}")
        print(f"  {'k':>5} | {'EM%':>8} | {'EditSim':>8} | {'N':>6}")
        print("  " + "-" * 40)
        curve_data[mode] = []
        for k in sorted(results[mode].keys()):
            r = results[mode][k]
            if r["total"] == 0:
                continue
            em_pct = 100.0 * r["em_count"] / r["total"]
            edit_avg = r["edit_sum"] / r["total"]
            print(f"  {k:>5} | {em_pct:>7.2f}% | {edit_avg:>8.4f} | {r['total']:>6}")
            curve_data[mode].append({
                "n_files": k,
                "exact_match_pct": em_pct,
                "edit_similarity": edit_avg,
                "n_examples": r["total"],
            })

    # ── Save results ─────────────────────────────────────────────
    output_results = {
        "method": "progressive_internalization",
        "split": args.split,
        "modes": {m: curve_data.get(m, []) for m in modes_to_run},
        "file_steps": file_steps,
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = checkpoint_path.parent / f"progressive_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
