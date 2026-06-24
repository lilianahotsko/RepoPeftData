#!/usr/bin/env python3
"""Train ``Code2LoRA-GRU<sub>commit</sub>`` on the v2 commits dataset.

This trainer is a **drop-in upgrade** of ``train_code2lora_gru_commits.py``
with three substantive changes:

  1. The LoRA generator is the **same** ``Code2LoRAHead`` used by the static
     trainer (``train_code2lora_static_v2.py``) -- not the PAW-style
     ``LoraGenerator`` from ``code2lora_gru.py``. The GRU sits *on top of* the
     same head, so "Code2LoRA-GRU = Code2LoRA + GRU" architecturally.

  2. Diff embeddings and the initial repo-state embedding are **read from
     parquet** (``repopeft-gru-commits-v2``) instead of being computed on the
     fly with a Qwen3 encoder. No encoder is loaded.

  3. Loss is computed by walking the chronological commit list per repo,
     stepping the GRU one commit at a time, and -- whenever a commit carries
     in-repo-train QnAs -- injecting LoRA(A, B) = head(h_t) into the wrapped
     base LLM and computing cross-entropy on those QnAs. Truncated BPTT keeps
     gradients bounded.

The base LLM is wrapped with ``replace_with_lora`` (identical to the static
trainer) -- so the only difference between the two trainers, at the model
level, is the GRU recurrence stage that produces ctx.

Inputs (parquets produced by ``create_dataset/merge_gru_v2_embeddings.py``):

    <commits-dir>/commits/{train,ir_val,ir_test,cr_val,cr_test}.parquet
        repo_id, commit_sha, commit_index, in_repo_split, cross_repo_split,
        production_code_diff, ..., diff_embedding (2048-d fp16),
        repo_state_embedding (2048-d fp16)
    <qnas-dir>/qna/{train,ir_val,ir_test,cr_val,cr_test}.parquet
        repo_id, commit_sha, commit_index, in_repo_split, prefix, target, ...

(The commits parquets are sharded by ``cross_repo_split`` -- ``train.parquet``
is the union of train repos' commits. The QnA parquets are the *smart-capped*
versions for ``train`` and the *canonical* versions for the eval suites.)

Usage::

    sbatch scripts/slurm/train_code2lora_gru_v2.sh
"""

from __future__ import annotations

import argparse
import json
import math
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
from code2lora_core import (
    Code2LoRAHead,
    CommitGRU,
    CommitRow,
    PER_STEP_INPUT_MODES,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_commit_rows_for_gru,
    load_qna_rows,
    make_per_step_input,
    per_step_input_dim,
    replace_with_lora,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Tokenization (same helper as static trainer; kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _tokenize_lm_batch(tokenizer, prefixes: List[str], targets: List[str],
                       max_seq_len: int = 4096) -> Dict[str, torch.Tensor]:
    """Causal-LM batch with **left-truncate / left-pad** -- mirrors
    ``hypernetwork_sampled.py:left_truncate_left_pad``. Targets are always
    kept; prefixes shed tokens from the left when total length > max_seq_len."""
    eos = tokenizer.eos_token or ""
    ids_list, lbl_list, am_list = [], [], []
    for p, t in zip(prefixes, targets):
        t_ids = tokenizer(t + eos, add_special_tokens=False)["input_ids"]
        if not t_ids:
            continue
        # Skip pathological QnAs whose target alone leaves no room for any
        # prefix. Targets are never truncated (truncating would teach a broken
        # answer), so an oversized target would otherwise blow the sequence
        # length far past max_seq_len and OOM the fp32 logits cast.
        if len(t_ids) > max_seq_len - 8:
            continue
        prefix_budget = max(8, max_seq_len - len(t_ids))
        p_full = tokenizer(p, add_special_tokens=False)["input_ids"]
        p_ids = p_full[-prefix_budget:] if len(p_full) > prefix_budget else p_full
        ids = p_ids + t_ids
        labels = ([-100] * len(p_ids)) + list(t_ids)
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        lbl_list.append(torch.tensor(labels, dtype=torch.long))
        am_list.append(torch.ones(len(ids), dtype=torch.long))
    if not ids_list:
        return {}
    L = max(t.size(0) for t in ids_list)
    pad_id = tokenizer.pad_token_id or 0
    def _lpad(x, val): return F.pad(x, (L - x.size(0), 0), value=val)
    return {
        "input_ids": torch.stack([_lpad(t, pad_id) for t in ids_list], 0),
        "labels":    torch.stack([_lpad(t, -100) for t in lbl_list], 0),
        "attention_mask": torch.stack([_lpad(t, 0) for t in am_list], 0),
    }


# ---------------------------------------------------------------------------
# Per-repo rollout (training step)
# ---------------------------------------------------------------------------

def _bptt_detach(h: torch.Tensor) -> torch.Tensor:
    return h.detach().requires_grad_(False)


def train_one_repo(
    *,
    rows: List[CommitRow],
    qnas_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
    gru: CommitGRU,
    head: Code2LoRAHead,
    base_model: nn.Module,
    specs,
    tokenizer,
    optim: torch.optim.Optimizer,
    sched,
    device: torch.device,
    bptt_window: int = 16,
    max_qna_per_commit: int = 8,
    lm_micro_batch: int = 4,
    max_seq_len: int = 1024,
    max_grad_norm: float = 1.0,
    per_step_input: str = "diff",
    rng: random.Random,
    task_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[float, int, int]:
    """One pass over the chronological commit list of a single repo.

    Returns (sum_loss, n_tokens, n_qna_examples).
    """
    if not rows:
        return 0.0, 0, 0

    base_model.eval()
    head.train(); gru.train()

    # Initial repo-state embedding @ commit 0 -> h_0 (always repo_state).
    repo_emb_0 = torch.from_numpy(rows[0].repo_state_embedding).to(device).unsqueeze(0)
    h = gru.init_hidden(repo_emb_0)

    sum_loss = 0.0
    n_tokens = 0
    n_examples = 0
    steps_since_detach = 0

    for t, row in enumerate(rows):
        step_emb = torch.from_numpy(
            make_per_step_input(row, per_step_input)
        ).to(device).unsqueeze(0)
        # Step the GRU forward by one commit.
        h = gru.step(step_emb, h)

        if row.in_repo_split != "train":
            steps_since_detach += 1
            if steps_since_detach >= bptt_window:
                h = _bptt_detach(h)
                steps_since_detach = 0
            continue

        pairs = qnas_by_key.get((row.repo_id, row.commit_sha), [])
        if not pairs:
            steps_since_detach += 1
            if steps_since_detach >= bptt_window:
                h = _bptt_detach(h)
                steps_since_detach = 0
            continue
        if len(pairs) > max_qna_per_commit:
            pairs = rng.sample(pairs, max_qna_per_commit)

        # ctx = output_norm(h_T)  -- gru.output_norm matches eval path.
        ctx = gru.output_norm(h[-1])
        repo_loss = 0.0
        repo_tokens = 0
        # Multi-task: inject a task-conditioned adapter per task group so a
        # single commit can train several tasks. Single-task runs collapse to
        # one group with task_id=None (identical to the original behaviour).
        for task_name, task_pairs in _by_task(pairs).items():
            tid = task_to_idx.get(task_name) if task_to_idx else None
            head_out = head(ctx, task_id=tid)
            inject_lora_weights(base_model, specs, head_out, batch_index=0)
            prefixes = [p["prefix"] for p in task_pairs]
            targets = [p["target"] for p in task_pairs]
            for i in range(0, len(prefixes), lm_micro_batch):
                j = min(i + lm_micro_batch, len(prefixes))
                batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                           max_seq_len=max_seq_len)
                if not batch:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                out = base_model(**batch)
                ntok = (batch["labels"] != -100).sum().item()
                loss = out.loss * ntok
                loss.backward(retain_graph=True)
                repo_loss += loss.detach().item()
                repo_tokens += ntok
        if repo_tokens > 0:
            torch.nn.utils.clip_grad_norm_(
                list(gru.parameters()) + list(head.parameters()),
                max_grad_norm,
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            # After backward, detach h so the next commit's BPTT starts fresh.
            h = _bptt_detach(h)
            steps_since_detach = 0
            sum_loss += repo_loss
            n_tokens += repo_tokens
            n_examples += len(pairs)
        else:
            steps_since_detach += 1
            if steps_since_detach >= bptt_window:
                h = _bptt_detach(h)
                steps_since_detach = 0

    return sum_loss, n_tokens, n_examples


# ---------------------------------------------------------------------------
# Eval: per-commit loss curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite(
    *,
    suite_name: str,
    rows_by_repo: Dict[str, List[CommitRow]],
    qnas_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
    gru: CommitGRU,
    head: Code2LoRAHead,
    base_model: nn.Module,
    specs,
    tokenizer,
    device: torch.device,
    max_qna_per_commit: int = 32,
    lm_micro_batch: int = 4,
    max_seq_len: int = 1024,
    in_repo_splits_to_score: Optional[List[str]] = None,
    max_repos: int = 0,
    per_step_input: str = "diff",
    task_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Per-commit eval loss across one suite. For 'cr_*' suites we score
    every commit; for 'ir_*' we score commits whose in_repo_split matches
    ``in_repo_splits_to_score`` (typically {val} or {test})."""
    base_model.eval(); head.eval(); gru.eval()
    per_commit: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_tokens = 0
    total_qna = 0
    repos = sorted(rows_by_repo.keys())
    if max_repos:
        repos = repos[:max_repos]
    for r in repos:
        rows = rows_by_repo[r]
        if not rows:
            continue
        repo_emb_0 = torch.from_numpy(rows[0].repo_state_embedding).to(device).unsqueeze(0)
        h = gru.init_hidden(repo_emb_0)
        for row in rows:
            step_emb = torch.from_numpy(
                make_per_step_input(row, per_step_input)
            ).to(device).unsqueeze(0)
            h = gru.step(step_emb, h)
            score_this = True
            if in_repo_splits_to_score is not None and \
                    row.in_repo_split not in in_repo_splits_to_score:
                score_this = False
            pairs = qnas_by_key.get((row.repo_id, row.commit_sha), [])
            if not pairs or not score_this:
                continue
            if len(pairs) > max_qna_per_commit:
                pairs = pairs[:max_qna_per_commit]
            ctx = gru.output_norm(h[-1])
            commit_loss = 0.0
            commit_tokens = 0
            for task_name, task_pairs in _by_task(pairs).items():
                tid = task_to_idx.get(task_name) if task_to_idx else None
                head_out = head(ctx, task_id=tid)
                inject_lora_weights(base_model, specs, head_out, batch_index=0)
                prefixes = [p["prefix"] for p in task_pairs]
                targets = [p["target"] for p in task_pairs]
                for i in range(0, len(prefixes), lm_micro_batch):
                    j = min(i + lm_micro_batch, len(prefixes))
                    batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                               max_seq_len=max_seq_len)
                    if not batch:
                        continue
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = base_model(**batch)
                    ntok = (batch["labels"] != -100).sum().item()
                    commit_loss += out.loss.item() * ntok
                    commit_tokens += ntok
            if commit_tokens > 0:
                per_commit.append({
                    "repo_id": row.repo_id,
                    "commit_sha": row.commit_sha,
                    "commit_index": row.commit_index,
                    "in_repo_split": row.in_repo_split,
                    "loss": commit_loss / commit_tokens,
                    "n_tokens": commit_tokens,
                    "n_qnas": len(pairs),
                })
                total_loss += commit_loss
                total_tokens += commit_tokens
                total_qna += len(pairs)
    return {
        "suite": suite_name,
        "n_repos": len(repos),
        "n_scored_commits": len(per_commit),
        "n_tokens": total_tokens,
        "n_qnas": total_qna,
        "eval_loss": total_loss / max(total_tokens, 1),
        "per_commit": per_commit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _by_task(pairs: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group a commit's QnA pairs by their task id (preserving order)."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for p in pairs:
        out.setdefault(p.get("task", "assert_rhs"), []).append(p)
    return out


def _group_qnas_by_key(rows) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for qr in rows:
        out.setdefault((qr.repo_id, qr.commit_sha), []).append({
            "prefix": qr.prefix, "target": qr.target,
            "task": getattr(qr, "task", "assert_rhs"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2")
    ap.add_argument("--qnas-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap",
                    help="Where qna/train.parquet (smart-cap) lives for training.")
    ap.add_argument("--eval-qnas-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf",
                    help="Where qna/{ir_val,ir_test,cr_val,cr_test}.parquet "
                         "live for the eval suites. Defaults to the v2 "
                         "snapshots dataset which has the same canonical "
                         "QnAs as the static Code2LoRA / baseline pipelines.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    # head + gru
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--tasks", nargs="+", default=["assert_rhs"],
                    help="Task ids in stable index order. >1 enables the "
                         "task-conditioned (multi-task) head; the QnA parquet's "
                         "'task' column selects each example's task.")
    ap.add_argument("--task-dim", type=int, default=64,
                    help="Task-embedding width when multi-task is enabled.")
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
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Matches the canonical Code2LoRA-GRU 'paper-grade 5ep_full' setting.")
    ap.add_argument("--per-step-input", default="diff",
                    choices=list(PER_STEP_INPUT_MODES),
                    help="What the GRU ingests at each commit step: "
                         "'diff' (default, original v2 GRU), 'repo_state' "
                         "(whole-repo embedding per commit), or 'concat' "
                         "([diff;repo_state]). h_0 is always initialized from "
                         "the first commit's repo_state_embedding.")

    # eval / ckpt
    ap.add_argument("--eval-every-repos", type=int, default=80,
                    help="Run eval after this many trained repos (set 0 = end of epoch only).")
    ap.add_argument("--eval-suites", nargs="+",
                    default=["cr_val"],
                    choices=["cr_val", "cr_test", "ir_val", "ir_test"])
    ap.add_argument("--primary-eval-suite", default="cr_val")
    ap.add_argument("--limit-eval-repos", type=int, default=10)
    ap.add_argument("--save-every-eval", action="store_true")

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-train-repos", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commits_dir = Path(args.commits_dir)
    qnas_dir = Path(args.qnas_dir)
    eval_qnas_dir = Path(args.eval_qnas_dir)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
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

    # ---- Build LLM + wrap with LoRA ----
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
    print(f"  {len(specs)} target modules, {len(type_dims)} types", flush=True)
    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)

    # ---- Hypernet pieces ----
    diff_dim = next(iter(train_by_repo.values()))[0].diff_embedding.shape[0]
    repo_dim = next(iter(train_by_repo.values()))[0].repo_state_embedding.shape[0]
    step_dim = per_step_input_dim(diff_dim, repo_dim, args.per_step_input)
    print(f"  per-step input mode: {args.per_step_input} (dim={step_dim})",
          flush=True)
    gru = CommitGRU(
        diff_input_dim=step_dim,
        repo_state_dim=repo_dim,
        hidden_dim=args.gru_hidden_dim,
    ).to(device)
    # Multi-task: more than one task id enables the head's task embedding.
    # A single task keeps num_tasks=0 -> byte-identical to the original head.
    task_to_idx = {t: i for i, t in enumerate(args.tasks)}
    num_tasks = len(args.tasks) if len(args.tasks) > 1 else 0
    if num_tasks:
        print(f"  multi-task head: {args.tasks} -> {task_to_idx}", flush=True)
    head = Code2LoRAHead(
        input_dim=args.gru_hidden_dim,
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
        num_tasks=num_tasks,
        task_dim=args.task_dim,
    ).to(device)

    # ---- Optim ----
    optim = torch.optim.AdamW(
        list(gru.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    # Rough total-step estimate: epochs * avg commits per repo with QnAs.
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
        qna_rows = load_qna_rows(
            eval_qnas_dir / "qna" / f"{suite}.parquet",
        )
        qnas_by_key = _group_qnas_by_key(qna_rows)
        eval_suites[suite] = {
            "rows_by_repo": rows_by_repo,
            "qnas_by_key": qnas_by_key,
            "in_repo_splits_to_score": in_repo_splits_for_suite,
        }
        print(f"  {suite}: {len(rows_by_repo)} repos, "
              f"{sum(len(v) for v in rows_by_repo.values())} commits, "
              f"{len(qna_rows)} qnas", flush=True)

    # ---- Train ----
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
                optim=optim, sched=sched, device=device,
                bptt_window=args.bptt_window,
                max_qna_per_commit=args.max_qna_per_commit,
                lm_micro_batch=args.lm_micro_batch,
                max_seq_len=args.max_seq_len,
                max_grad_norm=args.max_grad_norm,
                per_step_input=args.per_step_input,
                rng=rng,
                task_to_idx=task_to_idx,
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
                    eval_suites, device, out_dir, metrics_log,
                    best_eval_ref=[best_eval], epoch=epoch, repos_done=repos_done,
                )
                best_eval = min(best_eval, metrics_log[-1]["eval_loss"])

        # End of epoch: ALWAYS save the checkpoint FIRST, then validate.
        # This guarantees the epoch's weights are persisted even if eval
        # later runs out of time, fails, or the job is killed.
        type_dims = head.type_dims
        ep_path = _save_ckpt(out_dir, gru, head, type_dims, args,
                             name=f"ep{epoch}")
        latest_path = _save_ckpt(out_dir, gru, head, type_dims, args,
                                 name="latest")
        print(f"  [ckpt] end-of-epoch ep{epoch} -> {ep_path} "
              f"(also updated {latest_path})", flush=True)
        _do_eval(
            args, base_model, gru, head, specs, tokenizer, eval_suites, device,
            out_dir, metrics_log, best_eval_ref=[best_eval],
            epoch=epoch, repos_done=repos_done, end_of_epoch=True,
        )
        best_eval = min(best_eval, metrics_log[-1]["eval_loss"])

    print(f"\nDone. Best primary eval = {best_eval:.4f}", flush=True)


def _save_ckpt(out_dir: Path, gru: CommitGRU, head: Code2LoRAHead,
               type_dims, args, name: str = "latest") -> Path:
    out = out_dir / f"gru_head.{name}.pt"
    torch.save({
        "gru_state": gru.state_dict(),
        "head_state": head.state_dict(),
        "head_config": head.config_dict(),
        "gru_config": {
            "diff_input_dim": gru.diff_input_dim,
            "repo_state_dim": gru.repo_state_dim,
            "hidden_dim": gru.hidden_dim,
        },
        "type_dims": type_dims,
        "args": vars(args),
        "per_step_input": getattr(args, "per_step_input", "diff"),
    }, out)
    return out


def _do_eval(args, base_model, gru, head, specs, tokenizer, eval_suites,
             device, out_dir, metrics_log, *, best_eval_ref,
             epoch, repos_done, end_of_epoch: bool = False) -> None:
    task_to_idx = {t: i for i, t in enumerate(getattr(args, "tasks", ["assert_rhs"]))}
    suite_metrics: Dict[str, Any] = {}
    for name, s in eval_suites.items():
        m = evaluate_suite(
            suite_name=name,
            rows_by_repo=s["rows_by_repo"],
            qnas_by_key=s["qnas_by_key"],
            gru=gru, head=head, base_model=base_model,
            specs=specs, tokenizer=tokenizer, device=device,
            max_qna_per_commit=args.max_qna_per_commit,
            lm_micro_batch=args.lm_micro_batch,
            max_seq_len=args.max_seq_len,
            in_repo_splits_to_score=s["in_repo_splits_to_score"],
            max_repos=args.limit_eval_repos,
            per_step_input=args.per_step_input,
            task_to_idx=task_to_idx,
        )
        # Write per-commit JSON for decay-curve plotting.
        out = out_dir / f"eval_per_commit_{name}_repos{repos_done}.json"
        out.write_text(json.dumps({
            "summary": {k: v for k, v in m.items() if k != "per_commit"},
            "per_commit": m["per_commit"],
        }))
        suite_metrics[name] = {k: v for k, v in m.items() if k != "per_commit"}
        print(f"  [eval {name}] repos={repos_done} loss={m['eval_loss']:.4f} "
              f"scored_commits={m['n_scored_commits']} tok={m['n_tokens']}",
              flush=True)
    primary = suite_metrics.get(args.primary_eval_suite, {})
    primary_loss = primary.get("eval_loss", float("inf"))
    row = {"epoch": epoch, "repos_done": repos_done, "end_of_epoch": end_of_epoch,
           "eval_loss": primary_loss, "suites": suite_metrics}
    metrics_log.append(row)
    (out_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")
    type_dims = head.type_dims
    if primary_loss < best_eval_ref[0]:
        best_eval_ref[0] = primary_loss
        p = _save_ckpt(out_dir, gru, head, type_dims, args, name="best")
        print(f"  [ckpt] best -> {p} (loss={primary_loss:.4f})", flush=True)
    _save_ckpt(out_dir, gru, head, type_dims, args, name="latest")
    if args.save_every_eval:
        _save_ckpt(out_dir, gru, head, type_dims, args, name=f"repos{repos_done}")
    gru.train(); head.train()


if __name__ == "__main__":
    main()
