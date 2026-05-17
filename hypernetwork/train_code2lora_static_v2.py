#!/usr/bin/env python3
"""Train ``Code2LoRA-direct`` (static hypernetwork) on the v2 snapshots dataset.

This is the "regular Code2LoRA" baseline: a feed-forward hypernet that maps a
**precomputed** 2048-d ``repo_state_embedding`` -> per-(module-type) LoRA
weights via :class:`hypernetwork.code2lora_core.Code2LoRAHead`. There is no
encoder forward pass during training -- every (repo, commit) snapshot already
carries its embedding in the parquet, courtesy of the v2 dataset builder.

Inputs (parquets produced by ``create_dataset/build_code2lora_snapshots_parquet.py``):

    <snapshots-dir>/
        commits/
            train.parquet      400 anchor rows
            ir_val.parquet     train repos x val commits
            ir_test.parquet    train repos x test commits
            cr_val.parquet     cr_val repos, all kept commits
            cr_test.parquet    cr_test repos, all kept commits
        qna/
            train.parquet      v2-extracted QnAs at the anchor
            ir_val.parquet     canonical QnAs at val commits
            ir_test.parquet    canonical QnAs at test commits
            cr_val.parquet     canonical QnAs across cr_val commits
            cr_test.parquet    canonical QnAs across cr_test commits

The trainer:

  1. Loads ``commits/train.parquet`` -> {repo_id: (sha, idx, repo_state_emb)}.
  2. Loads ``qna/train.parquet``     -> [(repo_id, sha, prefix, target), ...].
  3. Builds the base LLM, discovers target modules, wraps them with :class:`LoRA`.
  4. For each batched snapshot: forward the head to get (A, B), inject into the
     wrappers, compute LM cross-entropy on the K QnAs at that snapshot.
  5. Periodically evaluates against ``cr_val`` (primary checkpoint signal) and
     dumps a checkpoint when the eval loss improves.

Eval suites (chosen on the command line via --eval-suites): any of
``ir_val ir_test cr_val cr_test`` -- joined per-commit via ``(repo_id, commit_sha)``.

Usage::

    sbatch scripts/slurm/train_code2lora_static_v2.sh

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from code2lora_core import (
    Code2LoRAHead,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_qna_rows,
    load_snapshot_rows,
    replace_with_lora,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset & batching
# ---------------------------------------------------------------------------

class StaticSnapshotDataset(Dataset):
    """One example = one (repo, snapshot) with its associated QnAs.

    Iteration order is shuffled per epoch; within an iteration the LM loss is
    computed on up to ``max_qna_per_snapshot`` QnAs (randomly sub-sampled when
    the repo has more).
    """

    def __init__(
        self,
        snapshots_by_repo: Dict[str, Dict[str, Any]],
        qnas_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
        repo_ids: List[str],
        max_qna_per_snapshot: int = 32,
        seed: int = 3407,
    ):
        # snapshots_by_repo[r] = {"sha": str, "commit_index": int, "emb": np.ndarray[2048]}
        # qnas_by_key[(r, sha)] = [{"prefix": str, "target": str}, ...]
        self.repo_ids = list(repo_ids)
        self.snapshots = snapshots_by_repo
        self.qnas = qnas_by_key
        self.max_qna = max_qna_per_snapshot
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.repo_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.repo_ids[idx]
        snap = self.snapshots[r]
        sha = snap["sha"]
        pairs = list(self.qnas.get((r, sha), []))
        if not pairs:
            return None
        if len(pairs) > self.max_qna:
            pairs = self.rng.sample(pairs, self.max_qna)
        return {
            "repo_id": r,
            "commit_sha": sha,
            "embedding": snap["emb"],
            "qnas": pairs,
        }


def _tokenize_lm_batch(tokenizer, prefixes: List[str], targets: List[str],
                       max_seq_len: int = 8192) -> Dict[str, torch.Tensor]:
    """Build a causal-LM batch where the loss is masked on the prefix tokens.

    Truncation policy (matches ``hypernetwork_sampled.py:left_truncate_left_pad``):
    keep the *rightmost* ``max_seq_len - len(target_ids)`` prefix tokens so the
    code immediately preceding the assertion is always retained. Targets are
    never truncated; prefixes shed from the left when the combined length
    overflows the cap.
    """
    eos = tokenizer.eos_token or ""
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    for p, t in zip(prefixes, targets):
        t_ids = tokenizer(t + eos, add_special_tokens=False)["input_ids"]
        if not t_ids:
            continue
        # cap is whatever's left after the (untruncated) target.
        prefix_budget = max(8, max_seq_len - len(t_ids))
        p_ids_full = tokenizer(p, add_special_tokens=False)["input_ids"]
        if len(p_ids_full) > prefix_budget:
            p_ids = p_ids_full[-prefix_budget:]
        else:
            p_ids = p_ids_full
        ids = p_ids + t_ids
        labels = ([-100] * len(p_ids)) + list(t_ids)
        attn = [1] * len(ids)
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.tensor(attn, dtype=torch.long))
    if not input_ids_list:
        return {}
    L = max(t.size(0) for t in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    # Left-pad so the assertion sits at the rightmost positions (matches the
    # original trainers' attention layout and is friendlier to KV-cached eval).
    def _lpad(x, val):
        return F.pad(x, (L - x.size(0), 0), value=val)
    input_ids = torch.stack([_lpad(t, pad_id) for t in input_ids_list], 0)
    labels = torch.stack([_lpad(t, -100) for t in labels_list], 0)
    attn = torch.stack([_lpad(t, 0) for t in attn_list], 0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite(
    base_model: nn.Module,
    head: Code2LoRAHead,
    specs,
    tokenizer,
    snap_rows: List[Any],   # SnapshotRow
    qnas_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
    *,
    device: torch.device,
    max_seq_len: int = 1024,
    lm_micro_batch: int = 4,
    max_qna_per_snapshot: int = 32,
) -> Dict[str, float]:
    """Token-level avg cross-entropy across every (snapshot, qna) we have."""
    base_model.eval()
    head.eval()
    total_loss = 0.0
    total_tokens = 0
    n_snap = 0
    for sr in snap_rows:
        pairs = qnas_by_key.get((sr.repo_id, sr.commit_sha))
        if not pairs:
            continue
        if len(pairs) > max_qna_per_snapshot:
            pairs = pairs[:max_qna_per_snapshot]
        ctx = torch.from_numpy(sr.repo_state_embedding).to(device).unsqueeze(0)
        head_out = head(ctx)
        inject_lora_weights(base_model, specs, head_out, batch_index=0)
        prefixes = [p["prefix"] for p in pairs]
        targets = [p["target"] for p in pairs]
        for i in range(0, len(prefixes), lm_micro_batch):
            j = min(i + lm_micro_batch, len(prefixes))
            batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                       max_seq_len=max_seq_len)
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model(**batch)
            loss = out.loss
            ntok = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * ntok
            total_tokens += ntok
        n_snap += 1
    avg = total_loss / max(total_tokens, 1)
    return {"eval_loss": avg, "n_snapshots": n_snap, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _anchor_map_from_train(snap_rows) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sr in snap_rows:
        out[sr.repo_id] = {
            "sha": sr.commit_sha,
            "commit_index": sr.commit_index,
            "emb": sr.repo_state_embedding,
        }
    return out


def _group_qnas_by_key(rows) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for qr in rows:
        out.setdefault((qr.repo_id, qr.commit_sha), []).append({
            "prefix": qr.prefix, "target": qr.target,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf")
    ap.add_argument("--output-dir", required=True,
                    help="Where to dump checkpoints + metrics.")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    # head hyperparameters
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--head-hidden-dim", type=int, default=1024)

    # optim
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)

    # batching
    ap.add_argument("--max-qna-per-snapshot", type=int, default=32)
    ap.add_argument("--lm-micro-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="Matches the canonical Code2LoRA / DRC-baseline setting.")

    # eval / checkpointing
    ap.add_argument("--eval-every-steps", type=int, default=500)
    ap.add_argument("--eval-suites", nargs="+",
                    default=["cr_val", "ir_val"],
                    choices=["cr_val", "cr_test", "ir_val", "ir_test"])
    ap.add_argument("--primary-eval-suite", default="cr_val")
    ap.add_argument("--limit-eval-snapshots", type=int, default=200,
                    help="Cap snapshots per eval suite (for speed; 0 = no cap).")
    ap.add_argument("--save-every-steps", type=int, default=0,
                    help="0 = save only when primary eval improves.")

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-train-repos", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(args.snapshots_dir)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
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
    specs = get_module_specs(base_model, args.target_modules)
    type_dims = discover_module_types_and_dims(specs)
    print(f"  discovered {len(specs)} target modules, "
          f"{len(type_dims)} types: {sorted(type_dims)}", flush=True)
    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)

    head = Code2LoRAHead(
        input_dim=train_snap[0].repo_state_embedding.shape[0],
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
    ).to(device)

    # ---- Optim ----
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(ds))
    total_steps = steps_per_epoch * args.epochs // max(1, args.grad_accum)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # ---- Eval suites ----
    print("Loading eval suites ...", flush=True)
    eval_suites: Dict[str, Dict[str, Any]] = {}
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
        head.train()
        running_loss = 0.0
        running_n = 0
        for it, di in enumerate(order):
            sample = ds[di]
            if sample is None:
                continue
            ctx = torch.from_numpy(sample["embedding"]).to(device).unsqueeze(0)
            head_out = head(ctx)
            inject_lora_weights(base_model, specs, head_out, batch_index=0)

            qnas = sample["qnas"]
            prefixes = [q["prefix"] for q in qnas]
            targets = [q["target"] for q in qnas]
            n_tok_seen = 0
            loss_acc = 0.0
            for i in range(0, len(prefixes), args.lm_micro_batch):
                j = min(i + args.lm_micro_batch, len(prefixes))
                batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                           max_seq_len=args.max_seq_len)
                if not batch:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                out = base_model(**batch)
                ntok = (batch["labels"] != -100).sum().item()
                loss = out.loss * ntok
                loss.backward()
                loss_acc += loss.detach().item()
                n_tok_seen += ntok
            if n_tok_seen == 0:
                continue

            if (it + 1) % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(),
                                               args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss_acc
            running_n += n_tok_seen
            if it % 50 == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = (time.time() - t0) / 60
                print(f"[ep{epoch} it{it}/{len(order)} step{global_step}] "
                      f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed:.1f}m", flush=True)
                running_loss = 0.0
                running_n = 0

            if args.eval_every_steps > 0 and global_step > 0 and \
                    global_step % args.eval_every_steps == 0 and \
                    (it + 1) % args.grad_accum == 0:
                _do_eval(args, base_model, head, specs, tokenizer,
                         eval_suites, device, out_dir, metrics_log,
                         best_eval_ref=[best_eval],
                         global_step=global_step, epoch=epoch)
                best_eval = min(best_eval, metrics_log[-1].get("eval_loss", float("inf")))

        # End of epoch: ALWAYS save the checkpoint FIRST, then validate. The
        # epoch's weights are persisted even if eval later runs out of time
        # or the SLURM job is killed.
        type_dims = head.type_dims
        ep_path = _save_ckpt(out_dir, head, type_dims, args, name=f"ep{epoch}")
        latest_path = _save_ckpt(out_dir, head, type_dims, args, name="latest")
        print(f"  [ckpt] end-of-epoch ep{epoch} -> {ep_path} "
              f"(also updated {latest_path})", flush=True)
        _do_eval(args, base_model, head, specs, tokenizer, eval_suites,
                 device, out_dir, metrics_log, best_eval_ref=[best_eval],
                 global_step=global_step, epoch=epoch, end_of_epoch=True)
        best_eval = min(best_eval, metrics_log[-1].get("eval_loss", float("inf")))

    print(f"\nTraining done. Best primary eval = {best_eval:.4f}", flush=True)


def _save_ckpt(out_dir: Path, head: Code2LoRAHead, type_dims, args,
               name: str = "latest") -> Path:
    out = out_dir / f"head.{name}.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "config": head.config_dict(),
        "type_dims": type_dims,
        "args": vars(args),
    }, out)
    return out


def _do_eval(args, base_model, head, specs, tokenizer, eval_suites, device,
             out_dir, metrics_log, *, best_eval_ref, global_step, epoch,
             end_of_epoch: bool = False) -> None:
    suite_metrics: Dict[str, Dict[str, float]] = {}
    for name, suite in eval_suites.items():
        m = evaluate_suite(
            base_model, head, specs, tokenizer,
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
    type_dims = head.type_dims
    if primary_loss < best_eval_ref[0]:
        best_eval_ref[0] = primary_loss
        p = _save_ckpt(out_dir, head, type_dims, args, name="best")
        print(f"  [ckpt] best updated -> {p}  (loss={primary_loss:.4f})", flush=True)
    if args.save_every_steps and global_step % args.save_every_steps == 0:
        _save_ckpt(out_dir, head, type_dims, args, name=f"step{global_step}")
    _save_ckpt(out_dir, head, type_dims, args, name="latest")
    head.train()


if __name__ == "__main__":
    main()
