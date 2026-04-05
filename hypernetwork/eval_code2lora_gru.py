#!/usr/bin/env python3
"""
Evaluation script for Code2LoRA-GRU.

Two evaluation modes:
  A) Standard: Evaluate on CR/IR test splits using the full file sequence
     per repository. Reports EM, EditSim, CodeBLEU.
  B) Progressive: For each repo, evaluate at commit percentiles (25%, 50%,
     75%, 100%) to show how LoRA quality improves as more commit history
     is processed. Supports streaming GRU updates.

Usage:
    # Standard evaluation
    python hypernetwork/eval_code2lora_gru.py \
        --checkpoint $SCRATCH/TRAINING_CHECKPOINTS/CODE2LORA_GRU/full_repos \
        --split cr_test

    # Progressive evaluation
    python hypernetwork/eval_code2lora_gru.py \
        --checkpoint $SCRATCH/TRAINING_CHECKPOINTS/CODE2LORA_GRU/full_repos \
        --split cr_test --mode progressive --commit-steps 0,25,50,75,100
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.metrics import (
    code_bleu_score,
    compute_metrics,
    edit_similarity,
    exact_match,
    postprocess_prediction,
    strip_comments,
)
from evaluation.baseline_config import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
)

from code2lora_gru import Code2LoRAGRU
from train_code2lora_gru import (
    apply_lora_hooks,
    discover_target_modules,
    get_bos_id,
    remove_lora_hooks,
)


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict, Dict]:
    """Load a Code2LoRA-GRU checkpoint. Returns (state_dict, config)."""
    if checkpoint_path.is_dir():
        for name in ["code2lora_gru_best.pt", "code2lora_gru_state.pt"]:
            candidate = checkpoint_path / name
            if candidate.exists():
                checkpoint_path = candidate
                break

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return ckpt["model_state_dict"], ckpt["model_config"]


def build_model_from_config(config: Dict, module_dims: Dict, num_layers: int) -> Code2LoRAGRU:
    """Reconstruct a Code2LoRA-GRU model from a saved config."""
    lora_cfg = config.get("lora_generator", {})
    return Code2LoRAGRU(
        file_embed_dim=config["file_embed_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        gru_num_layers=config["gru_num_layers"],
        num_target_layers=num_layers,
        module_dims=module_dims,
        lora_hidden_dim=lora_cfg.get("hidden_dim", 512),
        lora_rank=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("lora_scaling", 2.0) * lora_cfg.get("rank", 16),
        lora_num_bases=lora_cfg.get("num_bases", 16),
        lora_trunk_depth=2,
        init_type=config.get("init_type", "mamba2"),
        bptt_window=None,
    )


def load_split_data(splits_dir: Path, split_name: str, limit_repos: Optional[int] = None):
    """Load a split JSON and return repo data."""
    gru_path = splits_dir / f"gru_{split_name}.json"
    standard_path = splits_dir / f"{split_name}.json"
    path = gru_path if gru_path.exists() else standard_path

    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if limit_repos:
        repo_names = repo_names[:limit_repos]
    return {r: repos[r] for r in repo_names}, path


def get_file_sequence(rdata: Dict, file_order: str = "chronological"):
    """Extract ordered file embeddings from repo data."""
    file_embs_raw = rdata.get("file_embeddings", [])
    if not file_embs_raw:
        return [], []

    emb_by_path = {fe["path"]: fe["embedding"] for fe in file_embs_raw}
    history = rdata.get("commit_history")

    if history and history.get("file_order"):
        ordered_paths = [fo["path"] for fo in history["file_order"]]
    else:
        ordered_paths = [fe["path"] for fe in file_embs_raw]

    if file_order == "alphabetical":
        ordered_paths = sorted(ordered_paths)
    elif file_order == "reverse":
        ordered_paths = list(reversed(ordered_paths))

    file_embs = []
    paths_used = []
    for p in ordered_paths:
        if p in emb_by_path:
            file_embs.append(emb_by_path[p])
            paths_used.append(p)

    preamble_embs = []
    if history:
        preamble_files = set(history.get("preamble_files", []))
        for p in ordered_paths:
            if p in preamble_files and p in emb_by_path:
                preamble_embs.append(emb_by_path[p])

    return file_embs, preamble_embs


def run_generation(
    model,
    code2lora_gru,
    target_modules_dict,
    file_embs_tensor,
    file_length,
    preamble_tensor,
    preamble_length,
    input_ids,
    tok,
    device,
    max_new_tokens,
):
    """Run full forward: GRU -> LoRA generator -> inject -> generate."""
    with torch.no_grad():
        h_final, lora_params = code2lora_gru(
            file_embeddings=file_embs_tensor,
            file_lengths=file_length,
            preamble_embeddings=preamble_tensor,
            preamble_lengths=preamble_length,
        )

    handles = apply_lora_hooks(
        target_modules_dict,
        lora_params,
        code2lora_gru.lora_generator.lora_scaling,
    )

    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_t,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][len(input_ids) :].tolist()
    pred = tok.decode(gen_ids, skip_special_tokens=True)

    remove_lora_hooks(handles)
    return pred


def evaluate_standard(
    model,
    code2lora_gru,
    target_modules_dict,
    repos_data,
    tok,
    device,
    max_new_tokens,
    max_input_tokens,
    limit_pairs_per_repo=None,
    file_order="chronological",
):
    """Standard evaluation: full file sequence per repo."""
    bos_id = get_bos_id(tok)
    results = {"em_count": 0, "total": 0, "edit_sum": 0.0, "bleu_sum": 0.0}
    per_repo_results = {}

    total_repos = len(repos_data)
    for ri, (repo_name, rdata) in enumerate(repos_data.items()):
        pairs = rdata.get("qna_pairs", [])
        pairs = [
            p for p in pairs
            if p.get("prefix") and p.get("target")
            and not p["target"].lstrip().startswith(",")
        ]
        if not pairs:
            continue
        if limit_pairs_per_repo:
            pairs = pairs[:limit_pairs_per_repo]

        file_embs, preamble_embs = get_file_sequence(rdata, file_order)

        if file_embs:
            file_tensor = torch.tensor([file_embs], dtype=torch.float32, device=device)
            file_len = torch.tensor([len(file_embs)], dtype=torch.long, device=device)
        else:
            repo_emb = rdata.get("embedding", [0.0])
            file_tensor = torch.tensor([[repo_emb]], dtype=torch.float32, device=device)
            file_len = torch.tensor([1], dtype=torch.long, device=device)

        preamble_tensor = None
        preamble_len = None
        if preamble_embs:
            preamble_tensor = torch.tensor(
                [preamble_embs], dtype=torch.float32, device=device
            )
            preamble_len = torch.tensor(
                [len(preamble_embs)], dtype=torch.long, device=device
            )

        repo_em = 0
        repo_total = 0
        repo_edit = 0.0

        for pi, p in enumerate(pairs):
            prefix = p["prefix"]
            target = p["target"]
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            input_ids = [bos_id] + prefix_ids
            if len(input_ids) > max_input_tokens:
                input_ids = input_ids[-max_input_tokens:]

            pred = run_generation(
                model, code2lora_gru, target_modules_dict,
                file_tensor, file_len, preamble_tensor, preamble_len,
                input_ids, tok, device, max_new_tokens,
            )

            target_clean = strip_comments(target)
            pred_clean = postprocess_prediction(pred, target)
            em = exact_match(pred_clean, target_clean)
            edit_sim = edit_similarity(pred_clean, target_clean)
            bleu = code_bleu_score(pred_clean, target_clean)

            results["em_count"] += int(em)
            results["total"] += 1
            results["edit_sum"] += edit_sim
            results["bleu_sum"] += bleu

            repo_em += int(em)
            repo_total += 1
            repo_edit += edit_sim

        if repo_total > 0:
            per_repo_results[repo_name] = {
                "em_pct": 100.0 * repo_em / repo_total,
                "edit_sim": repo_edit / repo_total,
                "n": repo_total,
                "n_files": len(file_embs),
            }

        if (ri + 1) % 10 == 0 or ri == total_repos - 1:
            running_em = (
                100.0 * results["em_count"] / results["total"]
                if results["total"] > 0
                else 0
            )
            print(
                f"  [{ri+1}/{total_repos}] Running EM={running_em:.2f}% "
                f"({results['em_count']}/{results['total']})",
                flush=True,
            )

    return results, per_repo_results


def evaluate_progressive(
    model,
    code2lora_gru,
    target_modules_dict,
    repos_data,
    tok,
    device,
    max_new_tokens,
    max_input_tokens,
    commit_steps,
    limit_pairs_per_repo=None,
    file_order="chronological",
):
    """Progressive evaluation: performance at different fractions of the file sequence."""
    bos_id = get_bos_id(tok)
    results_by_pct = defaultdict(
        lambda: {"em_count": 0, "total": 0, "edit_sum": 0.0, "bleu_sum": 0.0}
    )

    total_repos = len(repos_data)
    for ri, (repo_name, rdata) in enumerate(repos_data.items()):
        pairs = rdata.get("qna_pairs", [])
        pairs = [
            p for p in pairs
            if p.get("prefix") and p.get("target")
            and not p["target"].lstrip().startswith(",")
        ]
        if not pairs:
            continue
        if limit_pairs_per_repo:
            pairs = pairs[:limit_pairs_per_repo]

        file_embs, preamble_embs = get_file_sequence(rdata, file_order)
        n_files = len(file_embs)
        if n_files == 0:
            continue

        preamble_tensor = None
        preamble_len = None
        if preamble_embs:
            preamble_tensor = torch.tensor(
                [preamble_embs], dtype=torch.float32, device=device
            )
            preamble_len = torch.tensor(
                [len(preamble_embs)], dtype=torch.long, device=device
            )

        for pct in commit_steps:
            if pct == 0:
                k = 0
            else:
                k = max(1, int(n_files * pct / 100.0))
                k = min(k, n_files)

            if k == 0:
                file_tensor = torch.zeros(
                    1, 1, len(file_embs[0]),
                    dtype=torch.float32, device=device,
                )
                file_len = torch.tensor([1], dtype=torch.long, device=device)
            else:
                subset = file_embs[:k]
                file_tensor = torch.tensor(
                    [subset], dtype=torch.float32, device=device
                )
                file_len = torch.tensor([k], dtype=torch.long, device=device)

            for p in pairs:
                prefix = p["prefix"]
                target = p["target"]
                prefix_ids = tok.encode(prefix, add_special_tokens=False)
                input_ids = [bos_id] + prefix_ids
                if len(input_ids) > max_input_tokens:
                    input_ids = input_ids[-max_input_tokens:]

                pred = run_generation(
                    model, code2lora_gru, target_modules_dict,
                    file_tensor, file_len, preamble_tensor, preamble_len,
                    input_ids, tok, device, max_new_tokens,
                )

                target_clean = strip_comments(target)
                pred_clean = postprocess_prediction(pred, target)
                em = exact_match(pred_clean, target_clean)
                edit_sim = edit_similarity(pred_clean, target_clean)
                bleu = code_bleu_score(pred_clean, target_clean)

                r = results_by_pct[pct]
                r["em_count"] += int(em)
                r["total"] += 1
                r["edit_sum"] += edit_sim
                r["bleu_sum"] += bleu

        if (ri + 1) % 10 == 0 or ri == total_repos - 1:
            latest_pct = max(commit_steps)
            r = results_by_pct[latest_pct]
            running_em = (
                100.0 * r["em_count"] / r["total"]
                if r["total"] > 0
                else 0
            )
            print(
                f"  [{ri+1}/{total_repos}] @{latest_pct}%: "
                f"EM={running_em:.2f}% ({r['em_count']}/{r['total']})",
                flush=True,
            )

    return dict(results_by_pct)


def main():
    ap = argparse.ArgumentParser(description="Evaluate Code2LoRA-GRU")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--splits-dir",
        type=str,
        default=os.path.join(
            os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
            "REPO_DATASET",
        ),
    )
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--limit-pairs-per-repo", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "progressive"],
    )
    ap.add_argument(
        "--commit-steps",
        type=str,
        default="0,25,50,75,100",
        help="Comma-separated percentages for progressive eval",
    )
    ap.add_argument(
        "--file-order",
        type=str,
        default="chronological",
        choices=["chronological", "reverse", "alphabetical"],
    )
    target_modules_default = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ]
    ap.add_argument("--target-modules", nargs="+", default=target_modules_default)
    args = ap.parse_args()

    # ── Load code LLM ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading code LLM: {args.model_name}")
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

    target_modules_dict, module_dims, num_layers = discover_target_modules(
        model, args.target_modules
    )
    print(f"Target modules: {len(target_modules_dict)} across {num_layers} layers")

    # ── Load Code2LoRA-GRU ──
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    state_dict, config = load_checkpoint(ckpt_path)

    code2lora_gru = build_model_from_config(config, module_dims, num_layers)
    code2lora_gru.load_state_dict(state_dict)
    code2lora_gru.to(args.device).eval()
    print(
        f"Code2LoRA-GRU loaded: "
        f"{sum(p.numel() for p in code2lora_gru.parameters()):,} params"
    )

    # ── Load split data ──
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_data, split_path = load_split_data(
        splits_dir, args.split, args.limit_repos
    )
    total_pairs = sum(
        len(r.get("qna_pairs", [])) for r in repos_data.values()
    )
    print(f"Split: {split_path.name} ({len(repos_data)} repos, {total_pairs} pairs)")

    # ── Run evaluation ──
    if args.mode == "standard":
        print("\nRunning standard evaluation...")
        results, per_repo = evaluate_standard(
            model, code2lora_gru, target_modules_dict,
            repos_data, tok, args.device,
            args.max_new_tokens, args.max_input_tokens,
            args.limit_pairs_per_repo, args.file_order,
        )

        n = results["total"]
        em_pct = 100.0 * results["em_count"] / n if n > 0 else 0
        edit_avg = results["edit_sum"] / n if n > 0 else 0
        bleu_avg = results["bleu_sum"] / n if n > 0 else 0

        print("\n" + "=" * 60)
        print(f"Code2LoRA-GRU — {args.split}")
        print("=" * 60)
        print(f"  Exact Match:     {em_pct:.2f}% ({results['em_count']}/{n})")
        print(f"  Edit Similarity: {edit_avg:.4f}")
        print(f"  CodeBLEU:        {bleu_avg:.4f}")
        print(f"  N:               {n}")

        output_data = {
            "method": "code2lora_gru",
            "split": args.split,
            "exact_match_pct": em_pct,
            "edit_similarity": edit_avg,
            "code_bleu": bleu_avg,
            "n": n,
            "per_repo": per_repo,
            "config": config,
        }

    else:
        commit_steps = sorted(set(int(x) for x in args.commit_steps.split(",")))
        print(f"\nRunning progressive evaluation at {commit_steps}%...")

        results_by_pct = evaluate_progressive(
            model, code2lora_gru, target_modules_dict,
            repos_data, tok, args.device,
            args.max_new_tokens, args.max_input_tokens,
            commit_steps, args.limit_pairs_per_repo, args.file_order,
        )

        print("\n" + "=" * 60)
        print(f"Code2LoRA-GRU Progressive — {args.split}")
        print("=" * 60)
        print(f"  {'%':>6} | {'EM%':>8} | {'EditSim':>8} | {'CB':>8} | {'N':>6}")
        print("  " + "-" * 50)

        curve_data = []
        for pct in sorted(results_by_pct.keys()):
            r = results_by_pct[pct]
            n = r["total"]
            if n == 0:
                continue
            em_pct = 100.0 * r["em_count"] / n
            edit_avg = r["edit_sum"] / n
            bleu_avg = r["bleu_sum"] / n
            print(
                f"  {pct:>5}% | {em_pct:>7.2f}% | {edit_avg:>8.4f} | "
                f"{bleu_avg:>8.4f} | {n:>6}"
            )
            curve_data.append({
                "pct": pct,
                "exact_match_pct": em_pct,
                "edit_similarity": edit_avg,
                "code_bleu": bleu_avg,
                "n": n,
            })

        output_data = {
            "method": "code2lora_gru_progressive",
            "split": args.split,
            "commit_steps": commit_steps,
            "curve": curve_data,
            "config": config,
        }

    # ── Save results ──
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        suffix = f"_{args.mode}" if args.mode != "standard" else ""
        out_path = ckpt_dir / f"eval_{args.split}{suffix}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
