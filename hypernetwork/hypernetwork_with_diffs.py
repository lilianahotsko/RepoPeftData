#!/usr/bin/env python3
"""Train the V1 ``Hypernetwork`` on the v2 *snapshots* dataset.

This is the "regular hypernetwork" model from ``hypernetwork_sampled.py``
re-trained on the v2 per-commit *cutoff* dataset that
``train_code2lora_static_v2.py`` consumes, so the only thing that differs
between this trainer and the V2 static trainer is the **model class** -
the LoRA-generation function itself:

  * V2 static  : ``code2lora_core.Code2LoRAHead``       (this file: NOT used)
  * V1 hypernet: ``hypernetwork_sampled.Hypernetwork``  (this file: USED)

Everything else - the parquet readers, the per-snapshot iteration order,
the QnA sub-sampling, the LM tokenization, the micro-batch backward
pattern, the eval suites, the optimizer / scheduler - is reused verbatim
from ``train_code2lora_static_v2.py``.

To stay faithful to how the V1 hypernet was originally trained, ``ctx``
is L2-normalized **before** being passed into the model (this matches
``HypernetTrainer.compute_loss`` in ``hypernetwork_sampled.py``). Use
``--no-ctx-l2norm`` to disable that step.

Inputs (parquets produced by ``create_dataset/build_code2lora_snapshots_parquet.py``):

    <snapshots-dir>/
        commits/
            train.parquet      anchor rows (one per train repo)
            ir_val.parquet     train repos x val   commits
            ir_test.parquet    train repos x test  commits
            cr_val.parquet     cr_val   repos, all kept commits
            cr_test.parquet    cr_test  repos, all kept commits
        qna/
            train.parquet      v2-extracted QnAs at the anchor commit
            ir_val.parquet     canonical QnAs at val  commits
            ir_test.parquet    canonical QnAs at test commits
            cr_val.parquet     canonical QnAs across cr_val   commits
            cr_test.parquet    canonical QnAs across cr_test  commits

Usage::

    sbatch scripts/slurm/train_hypernet_with_diffs.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# v2 data pipeline + LoRA wrapping (the same building blocks that
# ``train_code2lora_static_v2.py`` uses).
from code2lora_core import (
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_qna_rows,
    load_snapshot_rows,
    replace_with_lora,
)

# Re-use the static-v2 trainer's helpers verbatim so "all same as the
# static trainer" is literal -- swapping these out would mean a different
# data path / different micro-batch policy.
from train_code2lora_static_v2 import (
    StaticSnapshotDataset,
    _anchor_map_from_train,
    _group_qnas_by_key,
    _tokenize_lm_batch,
    evaluate_suite,
)

# The V1 hypernet model (THIS is the only architectural change vs static V2).
from hypernetwork_sampled import Hypernetwork


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Thin wrapper: V1 Hypernetwork + the L2-norm of ctx that the V1 trainer
# applied in ``HypernetTrainer.compute_loss``. Bundling this lets us pass
# ``hyper`` straight to ``evaluate_suite`` (which only calls ``hyper(ctx)``
# and ``hyper.eval()``).
# ---------------------------------------------------------------------------

class HypernetWithCtxNorm(nn.Module):
    """Wraps :class:`Hypernetwork` and L2-normalizes ``ctx`` along the last
    dim before the forward pass (matches ``HypernetTrainer.compute_loss``).

    Args:
        hypernet: a constructed :class:`hypernetwork_sampled.Hypernetwork`.
        normalize_ctx: when False, ctx is passed through untouched (useful
            for ablations against the static V2 head that doesn't normalize
            ctx externally).
    """

    def __init__(self, hypernet: Hypernetwork, normalize_ctx: bool = True):
        super().__init__()
        self.hypernet = hypernet
        self.normalize_ctx = normalize_ctx

    def forward(self, ctx: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.normalize_ctx:
            ctx = F.normalize(ctx.float(), p=2, dim=-1)
        return self.hypernet(ctx)

    @property
    def hidden_dim(self) -> int:
        return self.hypernet.hidden_dim

    @property
    def rank(self) -> int:
        return self.hypernet.rank

    @property
    def input_dim(self) -> int:
        return self.hypernet.input_dim

    @property
    def types(self) -> List[str]:
        return self.hypernet.types

    @property
    def type_shapes(self) -> Dict[str, Tuple[int, int]]:
        return self.hypernet.type_shapes


# ---------------------------------------------------------------------------
# Helpers: specs adapter + checkpoint I/O matching the V1 hypernet schema
# ---------------------------------------------------------------------------

def _v2_specs_to_v1_tuples(specs) -> List[Tuple[str, int, str, int, int]]:
    """Convert the V2 :class:`code2lora_core.ModuleSpec` list to the
    tuple layout the V1 :class:`Hypernetwork.__init__` expects."""
    return [
        (s.full_name, s.layer_idx, s.type, s.in_features, s.out_features)
        for s in specs
    ]


def _save_hypernet_ckpt(
    out_dir: Path,
    hyper: HypernetWithCtxNorm,
    specs_v1: List[Tuple[str, int, str, int, int]],
    args: argparse.Namespace,
    name: str = "latest",
) -> Path:
    """V1-style payload: ``hypernet_state_dict`` + ``module_specs`` +
    ``hypernet_config``. Same shape that ``SaveHypernetCallback`` writes,
    so existing evaluators that load V1 hypernets keep working."""
    out = out_dir / f"hypernet.{name}.pt"
    hn = hyper.hypernet
    torch.save({
        "hypernet_state_dict": hn.state_dict(),
        "module_specs": specs_v1,
        "hypernet_config": {
            "input_dim": hn.input_dim,
            "hidden_dim": hn.hidden_dim,
            "rank": hn.rank,
            "types": hn.types,
            "type_shapes": hn.type_shapes,
            "normalize_ctx": hyper.normalize_ctx,
        },
        "args": vars(args),
    }, out)
    return out


# ---------------------------------------------------------------------------
# Eval / checkpoint dispatcher (mirror of static_v2._do_eval, V1 save schema)
# ---------------------------------------------------------------------------

def _do_eval(
    args,
    base_model,
    hyper: HypernetWithCtxNorm,
    specs,
    specs_v1,
    tokenizer,
    eval_suites,
    device,
    out_dir,
    metrics_log,
    *,
    best_eval_ref,
    global_step,
    epoch,
    end_of_epoch: bool = False,
) -> None:
    suite_metrics: Dict[str, Dict[str, float]] = {}
    for name, suite in eval_suites.items():
        m = evaluate_suite(
            base_model, hyper, specs, tokenizer,
            suite["snap_rows"], suite["qnas_by_key"],
            device=device,
            max_seq_len=args.max_seq_len,
            lm_micro_batch=args.lm_micro_batch,
            max_qna_per_snapshot=args.max_qna_per_snapshot,
        )
        suite_metrics[name] = m
        print(f"  [eval {name}] step={global_step} loss={m['eval_loss']:.4f} "
              f"snap={m['n_snapshots']} tok={m['n_tokens']}", flush=True)
    primary = suite_metrics.get(args.primary_eval_suite)
    primary_loss = primary["eval_loss"] if primary else float("inf")
    row = {"step": global_step, "epoch": epoch, "end_of_epoch": end_of_epoch,
           "eval_loss": primary_loss, "suites": suite_metrics}
    metrics_log.append(row)
    (out_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")
    if primary_loss < best_eval_ref[0]:
        best_eval_ref[0] = primary_loss
        p = _save_hypernet_ckpt(out_dir, hyper, specs_v1, args, name="best")
        print(f"  [ckpt] best updated -> {p}  (loss={primary_loss:.4f})",
              flush=True)
    if args.save_every_steps and global_step % args.save_every_steps == 0:
        _save_hypernet_ckpt(out_dir, hyper, specs_v1, args,
                            name=f"step{global_step}")
    _save_hypernet_ckpt(out_dir, hyper, specs_v1, args, name="latest")
    hyper.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf")
    ap.add_argument("--output-dir", required=True,
                    help="Where to dump checkpoints + metrics.")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    # Hypernetwork hyperparameters (V1 architecture). The V1 default of
    # ``hidden_dim=512`` is preserved here; bump to 1024 with
    # --hidden-dim 1024 to match the static-v2 ``head-hidden-dim`` default
    # for a strict capacity-matched comparison.
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--no-ctx-l2norm", dest="ctx_l2norm",
                    action="store_false",
                    help="Disable the L2-normalization of ctx that the V1 "
                         "HypernetTrainer applied. Default: ON (match V1).")
    ap.set_defaults(ctx_l2norm=True)

    # Optimizer (mirrors static_v2 defaults).
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)

    # Batching (mirrors static_v2).
    ap.add_argument("--max-qna-per-snapshot", type=int, default=32)
    ap.add_argument("--lm-micro-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="Matches the canonical Code2LoRA / DRC-baseline setting.")

    # Eval / checkpointing (mirrors static_v2).
    ap.add_argument("--eval-every-steps", type=int, default=500)
    ap.add_argument("--eval-suites", nargs="+",
                    default=["cr_val", "ir_val"],
                    choices=["cr_val", "cr_test", "ir_val", "ir_test"])
    ap.add_argument("--primary-eval-suite", default="cr_val")
    ap.add_argument("--limit-eval-snapshots", type=int, default=200,
                    help="Cap snapshots per eval suite (for speed; 0 = no cap).")
    ap.add_argument("--save-every-steps", type=int, default=0,
                    help="0 = save only when primary eval improves.")
    ap.add_argument("--log-every-iters", type=int, default=50,
                    help="Print running training loss every N training iters.")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip ALL validation (mid-step + end-of-epoch). "
                         "Only train and save per-epoch checkpoints.")

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-train-repos", type=int, default=0)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="Recompute base-model activations during backward "
                         "(memory-friendly). Pass --no-gradient-checkpointing "
                         "to disable.")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(args.snapshots_dir)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load training set ----
    print("Loading training snapshots ...", flush=True)
    train_snap = load_snapshot_rows(snap_dir / "commits" / "train.parquet")
    if args.limit_train_repos:
        train_snap = train_snap[: args.limit_train_repos]
    anchor_map = _anchor_map_from_train(train_snap)
    print(f"  {len(anchor_map)} train repos", flush=True)
    print("Loading training QnAs ...", flush=True)
    train_qnas = load_qna_rows(snap_dir / "qna" / "train.parquet")
    qnas_train = _group_qnas_by_key(train_qnas)
    print(f"  {sum(len(v) for v in qnas_train.values())} QnA pairs across "
          f"{len(qnas_train)} (repo, commit) keys", flush=True)

    repo_ids = sorted(anchor_map.keys())
    ds = StaticSnapshotDataset(
        snapshots_by_repo=anchor_map,
        qnas_by_key=qnas_train,
        repo_ids=repo_ids,
        max_qna_per_snapshot=args.max_qna_per_snapshot,
        seed=args.seed,
    )

    # ---- Build LLM, discover modules, wrap them ----
    print(f"Loading {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
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
    specs = get_module_specs(base_model, args.target_modules)
    type_dims = discover_module_types_and_dims(specs)
    print(f"  discovered {len(specs)} target modules, "
          f"{len(type_dims)} types: {sorted(type_dims)}", flush=True)
    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)

    # ---- Build the V1 Hypernetwork on the v2 ctx ----
    # ``Hypernetwork.__init__`` parses module_specs as 5-tuples, so adapt
    # the v2 ``ModuleSpec`` dataclass list.
    specs_v1 = _v2_specs_to_v1_tuples(specs)
    input_dim = train_snap[0].repo_state_embedding.shape[0]
    hypernet = Hypernetwork(
        input_dim=input_dim,
        module_specs=specs_v1,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).to(device)
    hyper = HypernetWithCtxNorm(hypernet, normalize_ctx=args.ctx_l2norm).to(device)
    n_params = sum(p.numel() for p in hyper.parameters())
    print(f"  Hypernetwork: {n_params:,} params  "
          f"(input_dim={input_dim} hidden_dim={args.hidden_dim} "
          f"rank={args.rank}, ctx_l2norm={args.ctx_l2norm})", flush=True)

    # ---- Optim ----
    optim = torch.optim.AdamW(hyper.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(ds))
    total_steps = steps_per_epoch * args.epochs // max(1, args.grad_accum)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # ---- Eval suites ----
    eval_suites: Dict[str, Dict[str, Any]] = {}
    if args.skip_eval:
        print("Skipping eval suite loading (--skip-eval)", flush=True)
    else:
        print("Loading eval suites ...", flush=True)
        for suite in args.eval_suites:
            c = load_snapshot_rows(snap_dir / "commits" / f"{suite}.parquet")
            if args.limit_eval_snapshots and len(c) > args.limit_eval_snapshots:
                c = c[: args.limit_eval_snapshots]
            keys = [(r.repo_id, r.commit_sha) for r in c]
            q = load_qna_rows(snap_dir / "qna" / f"{suite}.parquet",
                              commit_keys=keys)
            eval_suites[suite] = {
                "snap_rows": c,
                "qnas_by_key": _group_qnas_by_key(q),
            }
            print(f"  {suite}: {len(c)} snapshots, {len(q)} qnas", flush=True)

    # ---- Train ----
    metrics_log: List[Dict[str, Any]] = []
    best_eval = float("inf")
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(ds)))
        random.shuffle(order)
        hyper.train()
        running_loss = 0.0
        running_n = 0
        for it, di in enumerate(order):
            sample = ds[di]
            if sample is None:
                continue
            ctx = torch.from_numpy(sample["embedding"]).to(device).unsqueeze(0)

            qnas = sample["qnas"]
            prefixes = [q["prefix"] for q in qnas]
            targets = [q["target"] for q in qnas]
            micro_batches: List[Dict[str, torch.Tensor]] = []
            for i in range(0, len(prefixes), args.lm_micro_batch):
                j = min(i + args.lm_micro_batch, len(prefixes))
                b = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                       max_seq_len=args.max_seq_len)
                if b:
                    micro_batches.append({k: v.to(device) for k, v in b.items()})
            if not micro_batches:
                continue
            n_tok_seen = 0
            loss_acc = 0.0
            # Same memory pattern as static_v2: recompute hyper(ctx) once
            # per micro-batch so each ``loss.backward()`` frees its graph
            # (no retain_graph). Grads accumulate in hyper.parameters()
            # across micro-batches.
            for batch in micro_batches:
                head_out = hyper(ctx)
                inject_lora_weights(base_model, specs, head_out, batch_index=0)
                out = base_model(**batch)
                ntok = (batch["labels"] != -100).sum().item()
                loss = out.loss * ntok
                loss.backward()
                loss_acc += loss.detach().item()
                n_tok_seen += ntok
            if n_tok_seen == 0:
                continue

            if (it + 1) % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(hyper.parameters(),
                                               args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss_acc
            running_n += n_tok_seen
            if it % max(1, args.log_every_iters) == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = (time.time() - t0) / 60
                print(f"[ep{epoch} it{it}/{len(order)} step{global_step}] "
                      f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed:.1f}m", flush=True)
                running_loss = 0.0
                running_n = 0

            if not args.skip_eval and args.eval_every_steps > 0 \
                    and global_step > 0 \
                    and global_step % args.eval_every_steps == 0 \
                    and (it + 1) % args.grad_accum == 0:
                _do_eval(args, base_model, hyper, specs, specs_v1, tokenizer,
                         eval_suites, device, out_dir, metrics_log,
                         best_eval_ref=[best_eval],
                         global_step=global_step, epoch=epoch)
                best_eval = min(best_eval,
                                metrics_log[-1].get("eval_loss", float("inf")))

        # End of epoch: ALWAYS save first, then (optionally) validate.
        ep_path = _save_hypernet_ckpt(out_dir, hyper, specs_v1, args,
                                      name=f"ep{epoch}")
        latest_path = _save_hypernet_ckpt(out_dir, hyper, specs_v1, args,
                                          name="latest")
        print(f"  [ckpt] end-of-epoch ep{epoch} -> {ep_path} "
              f"(also updated {latest_path})", flush=True)
        if args.skip_eval:
            print(f"  [eval] skipped (--skip-eval) after ep{epoch}", flush=True)
        else:
            _do_eval(args, base_model, hyper, specs, specs_v1, tokenizer,
                     eval_suites, device, out_dir, metrics_log,
                     best_eval_ref=[best_eval],
                     global_step=global_step, epoch=epoch, end_of_epoch=True)
            best_eval = min(best_eval,
                            metrics_log[-1].get("eval_loss", float("inf")))

    if args.skip_eval:
        print(f"\nTraining done. (eval skipped)", flush=True)
    else:
        print(f"\nTraining done. Best primary eval = {best_eval:.4f}", flush=True)


if __name__ == "__main__":
    main()
