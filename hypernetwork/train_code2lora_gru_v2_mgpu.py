#!/usr/bin/env python3
"""Multi-GPU (model-sharded) variant of ``train_code2lora_gru_v2``.

Motivation
----------
``Code2LoRA-GRU`` with a 3B+ frozen base OOMs on a single 80 GB GPU: the GRU
loop keeps several LM forward graphs alive at once (``retain_graph=True`` over
up to ``max_qna_per_commit`` micro-batches) and does *not* use gradient
checkpointing, so the LM activation + logits memory dominates.

This trainer keeps **every hyperparameter identical** to
``train_code2lora_gru_v2`` and instead spreads the cost across GPUs:

  * The frozen base LLM is loaded with ``device_map`` (default ``"auto"``), so
    HuggingFace/accelerate shards its transformer layers across **all visible
    GPUs**. This halves (or better) the per-GPU LM activation + logits memory
    that causes the OOM.
  * The trainable ``Code2LoRAHead`` + ``CommitGRU`` + optimizer live on the
    base model's *input-embedding* device (typically ``cuda:0``).
  * The ONLY behavioural change vs the single-GPU trainer: generated LoRA
    ``(A, B)`` tensors are cast onto each target module's own device at
    injection time. ``Tensor.to(device)`` is differentiable, so the LM loss's
    backward graph still flows across devices straight into the head -- exactly
    the same autograd contract as the single-GPU path. The cast is a no-op when
    head and module already share a device.

Everything else -- data loading, the per-repo BPTT rollout (``train_one_repo``),
eval (``evaluate_suite``), checkpointing -- is reused unchanged from
``train_code2lora_gru_v2``.

Usage::

    sbatch scripts/slurm/train_code2lora_gru_v2_mgpu.sh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import code2lora_core
from code2lora_core import (
    Code2LoRAHead,
    CommitGRU,
    LoRA,
    discover_module_types_and_dims,
    get_module_specs,
    load_commit_rows_for_gru,
    load_qna_rows,
)

# Reuse the single-GPU trainer's per-repo rollout, eval, and checkpoint helpers
# verbatim -- they already accept an explicit ``device`` and do not assume the
# base model is single-device.
import train_code2lora_gru_v2 as gru_v2
from train_code2lora_gru_v2 import (
    train_one_repo,
    _do_eval,
    _save_ckpt,
    _group_qnas_by_key,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Device-aware LoRA injection
# ---------------------------------------------------------------------------
# When the base model is sharded across GPUs, a given LoRA-wrapped Linear may
# live on cuda:1 while the head (which produced A, B) lives on cuda:0. The core
# ``LoRA.set_lora_weights`` deliberately does NO device cast (it assumes head
# and base share a device). We patch it here so generated weights are moved onto
# each module's own device. ``.to(dev)`` is differentiable -> gradients flow
# back to the head across the device boundary; it is a no-op when devices match,
# so single-GPU behaviour is unchanged.
def _set_lora_weights_device_aware(self: "code2lora_core.LoRA",
                                   A: torch.Tensor, B: torch.Tensor) -> None:
    dev = self.base.weight.device
    self.A = A.to(dev)
    self.B = B.to(dev)


code2lora_core.LoRA.set_lora_weights = _set_lora_weights_device_aware


# ---------------------------------------------------------------------------
# Device-preserving LoRA wrapping (sharding-safe)
# ---------------------------------------------------------------------------
# Core's ``replace_with_lora`` relocates every wrapped module to
# ``next(model.parameters()).device`` (i.e. cuda:0). With a device_map-sharded
# base that pulls the q/k/v/... weights off their assigned shard while
# accelerate's hooks keep feeding those layers hidden states on the original
# shard -> "tensors on different devices". This variant wraps each target
# Linear **in place**, leaving its (frozen) base weights exactly where the
# device_map put them. The LoRA wrapper owns no parameters of its own (A/B are
# plain attributes set later), so no relocation is needed.
def replace_with_lora_sharded(model: nn.Module, specs, rank: int,
                              alpha: float) -> None:
    named = dict(model.named_modules())
    for sp in specs:
        parent_name, attr = sp.full_name.rsplit(".", 1)
        orig = getattr(named[parent_name], attr)
        if isinstance(orig, LoRA):
            continue
        assert isinstance(orig, nn.Linear), \
            f"{sp.full_name} is not nn.Linear (got {type(orig)})"
        # Keep base on its current shard/dtype; do NOT move across devices.
        wrapped = LoRA(orig, sp.in_features, sp.out_features, rank, alpha)
        setattr(named[parent_name], attr, wrapped)


# ---------------------------------------------------------------------------
# Args (mirror train_code2lora_gru_v2 + multi-GPU knobs)
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--commits-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2")
    ap.add_argument("--qnas-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap",
                    help="Where qna/train.parquet (smart-cap) lives for training.")
    ap.add_argument("--eval-qnas-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf",
                    help="Where qna/{ir_val,ir_test,cr_val,cr_test}.parquet live "
                         "for the eval suites.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    # head + gru
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--head-hidden-dim", type=int, default=1024)
    ap.add_argument("--gru-hidden-dim", type=int, default=2048)

    # optim
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--bptt-window", type=int, default=16)

    # batching
    ap.add_argument("--max-qna-per-commit", type=int, default=8)
    ap.add_argument("--lm-micro-batch", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=4096)

    # eval / ckpt
    ap.add_argument("--eval-every-repos", type=int, default=80)
    ap.add_argument("--eval-suites", nargs="+", default=["cr_val"],
                    choices=["cr_val", "cr_test", "ir_val", "ir_test"])
    ap.add_argument("--primary-eval-suite", default="cr_val")
    ap.add_argument("--limit-eval-repos", type=int, default=10)
    ap.add_argument("--save-every-eval", action="store_true")

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit-train-repos", type=int, default=0)

    # ---- multi-GPU specific ----
    ap.add_argument("--device-map", default="auto",
                    help="HuggingFace device_map for sharding the frozen base "
                         "across visible GPUs. 'auto'/'balanced' split evenly; "
                         "'balanced_low_0' keeps GPU0 lighter to leave room for "
                         "the head + optimizer.")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Optionally enable gradient checkpointing on the base "
                         "(orthogonal to sharding; cuts activation memory "
                         "further at the cost of a recompute in backward).")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commits_dir = Path(args.commits_dir)
    qnas_dir = Path(args.qnas_dir)
    eval_qnas_dir = Path(args.eval_qnas_dir)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("No CUDA device visible; this trainer requires GPUs.")
    print(f"Visible GPUs: {n_gpus}", flush=True)
    for i in range(n_gpus):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # ---- Load training commits + QnAs ----
    print("Loading train commits parquet ...", flush=True)
    train_by_repo = load_commit_rows_for_gru(commits_dir / "commits" / "train.parquet")
    print(f"  train: {len(train_by_repo)} repos, "
          f"{sum(len(v) for v in train_by_repo.values())} commits", flush=True)
    if args.limit_train_repos:
        keep = sorted(train_by_repo.keys())[: args.limit_train_repos]
        train_by_repo = {k: train_by_repo[k] for k in keep}
        print(f"  limited to {len(train_by_repo)} repos", flush=True)
    print("Loading train QnAs ...", flush=True)
    train_qna_rows = load_qna_rows(qnas_dir / "qna" / "train.parquet")
    train_qnas = _group_qnas_by_key(train_qna_rows)
    print(f"  {len(train_qna_rows)} train QnA pairs across "
          f"{len(train_qnas)} (repo, commit) keys", flush=True)

    # ---- Build LLM (sharded across GPUs) + wrap with LoRA ----
    print(f"Loading {args.model_name} with device_map={args.device_map!r} ...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device_map,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    if args.gradient_checkpointing:
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        print("  gradient checkpointing: ON", flush=True)

    # Report the resulting layer->device sharding so OOMs are easy to diagnose.
    devmap = getattr(base_model, "hf_device_map", None)
    if devmap:
        dev_counts: Dict[Any, int] = {}
        for d in devmap.values():
            dev_counts[d] = dev_counts.get(d, 0) + 1
        print(f"  hf_device_map module->device counts: {dev_counts}", flush=True)

    specs = get_module_specs(base_model, args.target_modules)
    type_dims = discover_module_types_and_dims(specs)
    print(f"  {len(specs)} target modules, {len(type_dims)} types", flush=True)
    replace_with_lora_sharded(base_model, specs, rank=args.rank, alpha=args.alpha)

    # ---- Place the trainable hypernet on the base's input-embedding device ----
    # Inputs and the head/GRU live where the embeddings are (the first shard).
    # accelerate's dispatch hooks move hidden states across shard boundaries; the
    # final loss lands on the last shard and backward flows back across devices.
    head_device = base_model.get_input_embeddings().weight.device
    print(f"  head/GRU + input device: {head_device}", flush=True)

    diff_dim = next(iter(train_by_repo.values()))[0].diff_embedding.shape[0]
    repo_dim = next(iter(train_by_repo.values()))[0].repo_state_embedding.shape[0]
    gru = CommitGRU(
        diff_input_dim=diff_dim,
        repo_state_dim=repo_dim,
        hidden_dim=args.gru_hidden_dim,
    ).to(head_device)
    head = Code2LoRAHead(
        input_dim=args.gru_hidden_dim,
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
    ).to(head_device)

    # ---- Optim ----
    optim = torch.optim.AdamW(
        list(gru.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    n_repo_steps = sum(
        sum(1 for r in rows if r.in_repo_split == "train" and
            (r.repo_id, r.commit_sha) in train_qnas)
        for rows in train_by_repo.values()
    )
    total_steps = max(1, n_repo_steps * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # ---- Eval suites ----
    print("Loading eval suites ...", flush=True)
    eval_suites: Dict[str, Dict[str, Any]] = {}
    for suite in args.eval_suites:
        rows_by_repo = load_commit_rows_for_gru(
            commits_dir / "commits" / f"{suite}.parquet",
        )
        if args.limit_eval_repos and len(rows_by_repo) > args.limit_eval_repos:
            keep = sorted(rows_by_repo.keys())[: args.limit_eval_repos]
            rows_by_repo = {k: rows_by_repo[k] for k in keep}
        in_repo_splits_for_suite = (
            None if suite.startswith("cr_")
            else (["val"] if suite == "ir_val" else ["test"])
        )
        qna_rows = load_qna_rows(eval_qnas_dir / "qna" / f"{suite}.parquet")
        qnas_by_key = _group_qnas_by_key(qna_rows)
        eval_suites[suite] = {
            "rows_by_repo": rows_by_repo,
            "qnas_by_key": qnas_by_key,
            "in_repo_splits_to_score": in_repo_splits_for_suite,
        }
        print(f"  {suite}: {len(rows_by_repo)} repos, "
              f"{sum(len(v) for v in rows_by_repo.values())} commits, "
              f"{len(qna_rows)} qnas", flush=True)

    # ---- Train (identical loop to single-GPU trainer) ----
    metrics_log: List[Dict[str, Any]] = []
    best_eval = float("inf")
    repo_ids = sorted(train_by_repo.keys())
    t0 = time.time()
    repos_done = 0
    for epoch in range(args.epochs):
        rng.shuffle(repo_ids)
        for ri, r in enumerate(repo_ids):
            sum_loss, ntok, nex = train_one_repo(
                rows=train_by_repo[r],
                qnas_by_key=train_qnas,
                gru=gru, head=head, base_model=base_model,
                specs=specs, tokenizer=tokenizer,
                optim=optim, sched=sched, device=head_device,
                bptt_window=args.bptt_window,
                max_qna_per_commit=args.max_qna_per_commit,
                lm_micro_batch=args.lm_micro_batch,
                max_seq_len=args.max_seq_len,
                max_grad_norm=args.max_grad_norm,
                rng=rng,
            )
            repos_done += 1
            avg = sum_loss / max(ntok, 1) if ntok else 0.0
            elapsed = (time.time() - t0) / 60
            print(f"[ep{epoch} repo{ri+1}/{len(repo_ids)}={r}] "
                  f"loss={avg:.4f} tok={ntok} ex={nex} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}m",
                  flush=True)

            if args.eval_every_repos and repos_done % args.eval_every_repos == 0:
                _do_eval(
                    args, base_model, gru, head, specs, tokenizer,
                    eval_suites, head_device, out_dir, metrics_log,
                    best_eval_ref=[best_eval], epoch=epoch, repos_done=repos_done,
                )
                best_eval = min(best_eval, metrics_log[-1]["eval_loss"])

        # End of epoch: save FIRST, then validate.
        type_dims = head.type_dims
        ep_path = _save_ckpt(out_dir, gru, head, type_dims, args, name=f"ep{epoch}")
        latest_path = _save_ckpt(out_dir, gru, head, type_dims, args, name="latest")
        print(f"  [ckpt] end-of-epoch ep{epoch} -> {ep_path} "
              f"(also updated {latest_path})", flush=True)
        _do_eval(
            args, base_model, gru, head, specs, tokenizer, eval_suites,
            head_device, out_dir, metrics_log, best_eval_ref=[best_eval],
            epoch=epoch, repos_done=repos_done, end_of_epoch=True,
        )
        best_eval = min(best_eval, metrics_log[-1]["eval_loss"])

    print(f"\nDone. Best primary eval = {best_eval:.4f}", flush=True)


if __name__ == "__main__":
    main()
