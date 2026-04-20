#!/usr/bin/env python3
"""
Stand-alone commit-sequential evaluation for Code2LoRA-GRU checkpoints
trained by ``train_code2lora_gru_commits.py``.

Given a checkpoint and a Parquet dataset, walks every kept commit of each
held-out repo, generates a LoRA at each step, and reports per-suite loss:

* ``in_repo_val``        -- training repos, ``in_repo_split == 'val'``
* ``in_repo_test``       -- training repos, ``in_repo_split == 'test'``
* ``cross_repo_cr_val``  -- ``cross_repo_split == 'cr_val'`` repos
* ``cross_repo_cr_test`` -- ``cross_repo_split == 'cr_test'`` repos

Each metric is logged to stdout and optionally to W&B.

Example:
    python hypernetwork/eval_code2lora_gru_commits.py \\
        --checkpoint $CKPT_DIR/CODE2LORA_GRU/commit_level_parquet/code2lora_gru_best.pt \\
        --parquet-dir $SCRATCH/REPO_DATASET/commit_parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code2lora_gru import Code2LoRAGRU
from parquet_commit_dataset import (
    load_commit_sequences_from_parquet,
    resolve_parquet_sources,
)
from train_code2lora_gru import discover_target_modules, set_seed
from train_code2lora_gru_commits import (
    DiffEmbedder,
    DEFAULT_EMBED_MODEL,
    evaluate_commit_sequential,
)


def _build_model_from_ckpt(
    ckpt_path: Path,
    base_model_name: str,
    file_embed_dim: int,
    target_modules,
    device: torch.device,
):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("model_config", {})
    lcfg = cfg.get("lora_generator", {}) or {}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    for p in base_model.parameters():
        p.requires_grad = False
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, target_modules,
    )

    # Prefer module_dims from the checkpoint if present to avoid shape mismatch.
    ckpt_module_dims = lcfg.get("module_dims")
    if ckpt_module_dims:
        module_dims = {k: tuple(v) for k, v in ckpt_module_dims.items()}
    ckpt_num_layers = lcfg.get("num_layers", num_layers)

    # LoRA alpha = rank * scaling
    rank = lcfg.get("rank", 16)
    scaling = lcfg.get("lora_scaling", 2.0)
    alpha = float(rank) * float(scaling)

    model = Code2LoRAGRU(
        file_embed_dim=cfg.get("file_embed_dim", file_embed_dim),
        gru_hidden_dim=cfg.get("gru_hidden_dim", 1024),
        gru_num_layers=cfg.get("gru_num_layers", 1),
        num_target_layers=ckpt_num_layers,
        module_dims=module_dims,
        lora_hidden_dim=lcfg.get("hidden_dim", 512),
        lora_rank=rank,
        lora_alpha=alpha,
        lora_num_bases=lcfg.get("num_bases", 16),
        lora_trunk_depth=2,
        init_type=cfg.get("init_type", "zeros"),
        gru_dropout=0.0,
        bptt_window=cfg.get("bptt_window", 32),
    ).to(device=device, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.eval()
    return model, base_model, target_modules_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to code2lora_gru_{best,final}.pt file.")
    ap.add_argument("--parquet-dir", type=str,
                    default=os.path.join(
                        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
                        "REPO_DATASET", "commit_parquet",
                    ))
    ap.add_argument("--parquet-prefer", type=str, default="auto",
                    choices=["auto", "concat", "shards"])
    ap.add_argument("--commits-parquet", type=str, default=None)
    ap.add_argument("--qna-parquet", type=str, default=None)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-assertions-per-commit", type=int, default=32)
    ap.add_argument("--assertion-mode", type=str, default="cumulative",
                    choices=["cumulative", "new"])
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument(
        "--suites", nargs="+",
        default=["in_repo_val", "in_repo_test", "cross_repo_cr_val",
                 "cross_repo_cr_test"],
        help="Which eval suites to run.",
    )
    ap.add_argument("--output-json", type=str, default=None,
                    help="Optional path to dump the results JSON.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--target-modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ])
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0")

    # ── Parquet sources ──
    sources = resolve_parquet_sources(
        parquet_dir=args.parquet_dir,
        commits_path=args.commits_parquet,
        qna_path=args.qna_parquet,
        prefer=args.parquet_prefer,
    )

    # ── Diff embedder ──
    etok = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
    emod = AutoModel.from_pretrained(
        args.embed_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    for p in emod.parameters():
        p.requires_grad = False
    diff_embedder = DiffEmbedder(model=emod, tokenizer=etok, device=str(device))
    file_embed_dim = diff_embedder.embed_dim

    # ── Tokenizer + base LLM + target modules + Code2LoRA-GRU ──
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, base_model, target_modules_dict = _build_model_from_ckpt(
        Path(args.checkpoint), args.model_name,
        file_embed_dim, args.target_modules, device,
    )

    # ── Eval suites ──
    results: Dict[str, Any] = {"checkpoint": str(args.checkpoint)}
    for suite in args.suites:
        if suite == "in_repo_val":
            data = load_commit_sequences_from_parquet(
                sources, cross_repo_splits=["train"], in_repo_splits=None,
                limit_repos=args.limit_repos,
            )
            keep = ["val"]
        elif suite == "in_repo_test":
            data = load_commit_sequences_from_parquet(
                sources, cross_repo_splits=["train"], in_repo_splits=None,
                limit_repos=args.limit_repos,
            )
            keep = ["test"]
        elif suite == "cross_repo_cr_val":
            data = load_commit_sequences_from_parquet(
                sources, cross_repo_splits=["cr_val"], in_repo_splits=None,
                limit_repos=args.limit_repos,
            )
            keep = None
        elif suite == "cross_repo_cr_test":
            data = load_commit_sequences_from_parquet(
                sources, cross_repo_splits=["cr_test"], in_repo_splits=None,
                limit_repos=args.limit_repos,
            )
            keep = None
        else:
            print(f"  [warn] unknown suite {suite}, skipping")
            continue

        if not data:
            print(f"  [{suite}] no data; skipping.")
            results[suite] = None
            continue

        print(f"\n=== Suite: {suite} (repos={len(data)}, keep={keep}) ===")
        m = evaluate_commit_sequential(
            model, base_model, diff_embedder, data,
            target_modules_dict, tok, args.max_seq_len, device,
            max_assertions_per_commit=args.max_assertions_per_commit,
            keep_splits=keep,
            assertion_mode=args.assertion_mode,
        )
        print(f"  loss={m['eval_loss']:.4f} "
              f"repos={m['eval_repos']} "
              f"commits_w_loss={m['eval_commits_with_loss']} "
              f"assertions={m['eval_assertions']}")
        results[suite] = m

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults JSON -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
