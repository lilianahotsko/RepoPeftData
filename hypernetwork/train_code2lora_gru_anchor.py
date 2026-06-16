#!/usr/bin/env python3
"""Train ``Code2LoRA-GRU`` with **anchor-only** repo embeddings.

This is an ablation of :mod:`train_code2lora_gru_v2`: keep the GRU architecture
and per-repo step count, but feed the **same** anchor-commit
``repo_state_embedding`` at every GRU step (no per-commit ``diff_embedding``
signal) and supervise with the **same anchor-commit QnAs used by the static
Code2LoRA trainer** (``train_code2lora_static_v2.py``).

Concretely, for each repo we:

  1. Resolve the anchor ``(commit_sha, repo_state_embedding)`` from the
     Code2LoRA-snapshots dataset (``code2lora_snapshots_hf/commits/*.parquet``).
     For train + ir_* suites the anchor lives in ``commits/train.parquet``
     (one row per train repo). For cr_val / cr_test we use the most recent
     kept commit's row from ``commits/{cr_val,cr_test}.parquet`` as the
     per-repo anchor.
  2. Walk the chronological kept-commit list from the v2 commits dataset
     (``commit_parquet_hf_v2/commits/*.parquet``) just to know how many GRU
     steps to unroll per repo (= number of kept commits).
  3. At every step, call ``gru.step(anchor_emb, h)``. Diff embeddings from
     the v2 parquet are **never** read.
  4. Fire LM loss only at the GRU step whose commit_sha matches the anchor,
     using the anchor QnAs from ``code2lora_snapshots_hf/qna/train.parquet``.

The base LLM wrap and the ``Code2LoRAHead`` are identical to
``train_code2lora_gru_v2``; so is the optimizer / scheduler / checkpoint
layout. Eval mirrors training: same anchor embedding, scored per-commit
against the snapshots-dataset eval QnAs.

Usage::

    sbatch scripts/slurm/train_code2lora_gru_anchor.sh
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
    SnapshotRow,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_commit_rows_for_gru,
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
# Tokenization (verbatim from train_code2lora_gru_v2)
# ---------------------------------------------------------------------------

def _tokenize_lm_batch(tokenizer, prefixes: List[str], targets: List[str],
                       max_seq_len: int = 4096) -> Dict[str, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    ids_list, lbl_list, am_list = [], [], []
    for p, t in zip(prefixes, targets):
        t_ids = tokenizer(t + eos, add_special_tokens=False)["input_ids"]
        if not t_ids:
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
# Anchor table: per-repo (commit_sha, repo_state_embedding[2048])
# ---------------------------------------------------------------------------

def _select_anchor_per_repo(snap_rows: List[SnapshotRow],
                            *, latest: bool) -> Dict[str, Tuple[str, np.ndarray]]:
    """Reduce a list of SnapshotRows to one per repo.

    If ``latest`` is True, take the row with the maximum commit_index for
    each repo (used for cr_val / cr_test where the snapshot parquet has
    multiple rows per repo). Otherwise take the first row encountered
    (snapshots/commits/train.parquet has a single anchor row per repo, so
    this is unambiguous; for ir_* it picks an arbitrary kept commit -- the
    caller normally passes the train anchor table instead).
    """
    out: Dict[str, Tuple[str, np.ndarray, int]] = {}
    for r in snap_rows:
        prev = out.get(r.repo_id)
        if prev is None:
            out[r.repo_id] = (r.commit_sha, r.repo_state_embedding, r.commit_index)
            continue
        if latest and r.commit_index > prev[2]:
            out[r.repo_id] = (r.commit_sha, r.repo_state_embedding, r.commit_index)
    return {k: (v[0], v[1]) for k, v in out.items()}


def build_anchor_table(snapshots_dir: Path, split: str,
                       train_anchors: Optional[Dict[str, Tuple[str, np.ndarray]]] = None,
                       ) -> Dict[str, Tuple[str, np.ndarray]]:
    """Return ``{repo_id: (anchor_sha, repo_state_embedding[2048])}`` for ``split``.

    * ``split == 'train'`` -> read snapshots/commits/train.parquet (one row
      per train repo, already the anchor).
    * ``split in ('ir_val', 'ir_test')`` -> reuse the train anchor table
      (ir_* uses train repos, and we want the same anchor embedding the
      model was trained on).
    * ``split in ('cr_val', 'cr_test')`` -> read snapshots/commits/{split}.parquet
      and pick the latest kept commit per repo as that repo's anchor.
    """
    parquet = snapshots_dir / "commits" / f"{split}.parquet"
    if split in ("ir_val", "ir_test"):
        if train_anchors is None:
            train_rows = load_snapshot_rows(snapshots_dir / "commits" / "train.parquet")
            train_anchors = _select_anchor_per_repo(train_rows, latest=False)
        return train_anchors
    rows = load_snapshot_rows(parquet)
    latest = split.startswith("cr_")
    return _select_anchor_per_repo(rows, latest=latest)


# ---------------------------------------------------------------------------
# Per-repo rollout (training step) -- anchor-only variant
# ---------------------------------------------------------------------------

def _bptt_detach(h: torch.Tensor) -> torch.Tensor:
    return h.detach().requires_grad_(False)


def train_one_repo(
    *,
    rows: List[CommitRow],
    anchor_sha: str,
    anchor_emb: torch.Tensor,           # [1, repo_state_dim], device, fp32
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
    rng: random.Random,
) -> Tuple[float, int, int]:
    """One pass over the chronological commit list of a single repo, feeding
    ``anchor_emb`` at every GRU step. Loss is fired only at the commit whose
    sha matches ``anchor_sha`` (which holds the static-trainer QnAs).

    Returns (sum_loss, n_tokens, n_qna_examples).
    """
    if not rows:
        return 0.0, 0, 0

    base_model.eval()
    head.train(); gru.train()

    # h_0 from the anchor repo-state embedding (same source as every step).
    h = gru.init_hidden(anchor_emb)

    sum_loss = 0.0
    n_tokens = 0
    n_examples = 0
    steps_since_detach = 0

    for t, row in enumerate(rows):
        # The GRU only ever sees the anchor embedding.
        h = gru.step(anchor_emb, h)

        # Loss fires only at the anchor commit.
        if row.commit_sha != anchor_sha:
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

        ctx = gru.output_norm(h[-1])
        head_out = head(ctx)
        inject_lora_weights(base_model, specs, head_out, batch_index=0)

        prefixes = [p["prefix"] for p in pairs]
        targets = [p["target"] for p in pairs]
        repo_loss = 0.0
        repo_tokens = 0
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
# Eval: per-commit loss curve, anchor-only variant
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite(
    *,
    suite_name: str,
    rows_by_repo: Dict[str, List[CommitRow]],
    anchor_by_repo: Dict[str, Tuple[str, np.ndarray]],
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
) -> Dict[str, Any]:
    """Per-commit eval loss across one suite, feeding the per-repo anchor
    embedding at every GRU step (mirrors training)."""
    base_model.eval(); head.eval(); gru.eval()
    per_commit: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_tokens = 0
    total_qna = 0
    skipped_no_anchor = 0
    repos = sorted(rows_by_repo.keys())
    if max_repos:
        repos = repos[:max_repos]
    for r in repos:
        rows = rows_by_repo[r]
        if not rows:
            continue
        anchor = anchor_by_repo.get(r)
        if anchor is None:
            skipped_no_anchor += 1
            continue
        _anchor_sha, anchor_emb_np = anchor
        anchor_emb = torch.from_numpy(anchor_emb_np).to(device).unsqueeze(0)
        h = gru.init_hidden(anchor_emb)
        for row in rows:
            h = gru.step(anchor_emb, h)
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
            head_out = head(ctx)
            inject_lora_weights(base_model, specs, head_out, batch_index=0)
            prefixes = [p["prefix"] for p in pairs]
            targets = [p["target"] for p in pairs]
            commit_loss = 0.0
            commit_tokens = 0
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
        "n_repos_missing_anchor": skipped_no_anchor,
        "n_scored_commits": len(per_commit),
        "n_tokens": total_tokens,
        "n_qnas": total_qna,
        "eval_loss": total_loss / max(total_tokens, 1),
        "per_commit": per_commit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--commits-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2",
                    help="Source of chronological per-commit lists (we only read "
                         "commit_sha/commit_index/in_repo_split from here; "
                         "diff_embedding is ignored).")
    ap.add_argument("--snapshots-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf",
                    help="Code2LoRA-snapshots dataset: provides the anchor "
                         "repo_state_embedding per train repo "
                         "(commits/train.parquet), per-suite anchors "
                         "(commits/{ir_val,ir_test,cr_val,cr_test}.parquet), "
                         "and all QnAs (qna/*.parquet).")
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
    snapshots_dir = Path(args.snapshots_dir)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # ---- Load training chronology + anchor table + train QnAs ----
    print("Loading train commits chronology (v2 parquet) ...", flush=True)
    train_by_repo = load_commit_rows_for_gru(commits_dir / "commits" / "train.parquet")
    print(f"  train chronology: {len(train_by_repo)} repos, "
          f"{sum(len(v) for v in train_by_repo.values())} commits", flush=True)
    if args.limit_train_repos:
        keep = sorted(train_by_repo.keys())[: args.limit_train_repos]
        train_by_repo = {k: train_by_repo[k] for k in keep}
        print(f"  limited to {len(train_by_repo)} repos", flush=True)

    print("Loading train anchor embeddings (snapshots/commits/train.parquet) ...",
          flush=True)
    train_anchors = build_anchor_table(snapshots_dir, "train")
    print(f"  {len(train_anchors)} train repos have an anchor", flush=True)

    # Drop train repos that lack an anchor in the snapshots dataset.
    missing = [r for r in train_by_repo if r not in train_anchors]
    if missing:
        print(f"  WARN: {len(missing)} train repos have no anchor in "
              f"snapshots; dropping them.", flush=True)
        for r in missing:
            train_by_repo.pop(r)

    # Also drop train repos whose anchor_sha is not present in the v2
    # chronology (would mean zero supervision steps).
    pruned = 0
    anchor_idx_stats: List[int] = []
    for r in list(train_by_repo.keys()):
        anchor_sha, _ = train_anchors[r]
        shas = [row.commit_sha for row in train_by_repo[r]]
        if anchor_sha not in shas:
            train_by_repo.pop(r)
            pruned += 1
        else:
            anchor_idx_stats.append(shas.index(anchor_sha))
    if pruned:
        print(f"  WARN: {pruned} train repos had an anchor_sha not in the "
              f"v2 chronology; dropped.", flush=True)
    if anchor_idx_stats:
        arr = np.array(anchor_idx_stats)
        print(f"  anchor position within chronology: "
              f"min={arr.min()} median={int(np.median(arr))} "
              f"max={arr.max()} (avg N={float(np.mean([len(v) for v in train_by_repo.values()])):.1f})",
              flush=True)

    print("Loading train QnAs (snapshots/qna/train.parquet) ...", flush=True)
    train_qna_rows = load_qna_rows(snapshots_dir / "qna" / "train.parquet")
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
    # The GRU is fed only the anchor repo_state_embedding, so the per-step
    # input dim == repo_state_embedding dim. We still construct CommitGRU
    # with matching diff_input_dim so that gru.step accepts the anchor
    # tensor without shape errors.
    repo_dim = next(iter(train_anchors.values()))[1].shape[0]
    diff_dim = repo_dim
    gru = CommitGRU(
        diff_input_dim=diff_dim,
        repo_state_dim=repo_dim,
        hidden_dim=args.gru_hidden_dim,
    ).to(device)
    head = Code2LoRAHead(
        input_dim=args.gru_hidden_dim,
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
    ).to(device)

    # ---- Optim ----
    optim = torch.optim.AdamW(
        list(gru.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    # Total-step estimate: each repo contributes exactly one supervised step
    # (its anchor) -> total_steps ≈ epochs * n_train_repos_with_qnas.
    n_repo_steps = sum(
        1 for r in train_by_repo
        if (r, train_anchors[r][0]) in train_qnas
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
            snapshots_dir / "qna" / f"{suite}.parquet",
        )
        qnas_by_key = _group_qnas_by_key(qna_rows)
        anchor_table = build_anchor_table(
            snapshots_dir, suite, train_anchors=train_anchors,
        )
        eval_suites[suite] = {
            "rows_by_repo": rows_by_repo,
            "qnas_by_key": qnas_by_key,
            "anchor_by_repo": anchor_table,
            "in_repo_splits_to_score": in_repo_splits_for_suite,
        }
        print(f"  {suite}: {len(rows_by_repo)} repos, "
              f"{sum(len(v) for v in rows_by_repo.values())} commits, "
              f"{len(qna_rows)} qnas, "
              f"{len(anchor_table)} anchors", flush=True)

    # ---- Train ----
    metrics_log: List[Dict[str, Any]] = []
    best_eval = float("inf")
    repo_ids = sorted(train_by_repo.keys())
    t0 = time.time()
    repos_done = 0
    for epoch in range(args.epochs):
        rng.shuffle(repo_ids)
        for ri, r in enumerate(repo_ids):
            anchor_sha, anchor_emb_np = train_anchors[r]
            anchor_emb = torch.from_numpy(anchor_emb_np).to(device).unsqueeze(0)
            sum_loss, ntok, nex = train_one_repo(
                rows=train_by_repo[r],
                anchor_sha=anchor_sha,
                anchor_emb=anchor_emb,
                qnas_by_key=train_qnas,
                gru=gru, head=head, base_model=base_model,
                specs=specs, tokenizer=tokenizer,
                optim=optim, sched=sched, device=device,
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
                  f"N={len(train_by_repo[r])} loss={avg:.4f} tok={ntok} ex={nex} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}m",
                  flush=True)

            if args.eval_every_repos and repos_done % args.eval_every_repos == 0:
                _do_eval(
                    args, base_model, gru, head, specs, tokenizer,
                    eval_suites, device, out_dir, metrics_log,
                    best_eval_ref=[best_eval], epoch=epoch, repos_done=repos_done,
                )
                best_eval = min(best_eval, metrics_log[-1]["eval_loss"])

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
        "anchor_only": True,
    }, out)
    return out


def _do_eval(args, base_model, gru, head, specs, tokenizer, eval_suites,
             device, out_dir, metrics_log, *, best_eval_ref,
             epoch, repos_done, end_of_epoch: bool = False) -> None:
    suite_metrics: Dict[str, Any] = {}
    for name, s in eval_suites.items():
        m = evaluate_suite(
            suite_name=name,
            rows_by_repo=s["rows_by_repo"],
            anchor_by_repo=s["anchor_by_repo"],
            qnas_by_key=s["qnas_by_key"],
            gru=gru, head=head, base_model=base_model,
            specs=specs, tokenizer=tokenizer, device=device,
            max_qna_per_commit=args.max_qna_per_commit,
            lm_micro_batch=args.lm_micro_batch,
            max_seq_len=args.max_seq_len,
            in_repo_splits_to_score=s["in_repo_splits_to_score"],
            max_repos=args.limit_eval_repos,
        )
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
