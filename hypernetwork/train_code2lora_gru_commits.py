#!/usr/bin/env python3
"""
Commit-level training for Code2LoRA-GRU: iterate over each kept commit in a
repository's history, embed the diff on-the-fly, update the GRU hidden state,
generate a LoRA, and compute loss on assertions available up to that commit.

Supported data sources (mutually exclusive at the CLI level):

* ``--parquet-dir DIR`` (preferred) reads the Parquet dataset built by
  ``create_dataset/build_commit_parquet_db.py`` (``commits.parquet`` +
  ``qna_pairs.parquet`` or per-repo ``shards/``). Each row carries
  ``cross_repo_split`` (train / cr_val / cr_test) and ``in_repo_split``
  (train / val / test, chronological 80/10/10 over kept commits).
* ``--db-path FILE`` reads the legacy SQLite database built by
  ``create_dataset/build_commit_assertion_db.py`` (all commits, one diff each).

Training loop per repository:
  1. Initialize h_0 (zeros or from initializer).
  2. For each commit k in chronological order:
     a. Embed production_code_diff via frozen Qwen3-Embedding (chunk + MaxPool||MeanPool).
     b. GRU step: h_k = encode_repository_commit(diff_emb_k, h_{k-1}).
     c. Collect assertions where commit_index <= k (cumulative) filtered by
        in-repo split (train for training, val/test for evaluation).
     d. If assertions exist: generate LoRA from h_k, apply hooks, compute LM loss.
     e. Detach h_k for truncated BPTT (TBPTT window = 1 across commits).
  3. Backprop accumulated loss, step optimizer.

Usage:
    python hypernetwork/train_code2lora_gru_commits.py \\
        --parquet-dir $SCRATCH/REPO_DATASET/commit_parquet
    python hypernetwork/train_code2lora_gru_commits.py \\
        --db-path    $SCRATCH/REPO_DATASET/commits_assertions.db
"""

import argparse
import math
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code2lora_gru import Code2LoRAGRU, pool_file_chunks_maxmean
from parquet_commit_dataset import (
    load_commit_sequences_from_parquet,
    resolve_parquet_sources,
    summarize_items,
)
from train_code2lora_gru import (
    apply_lora_hooks,
    discover_target_modules,
    get_bos_id,
    left_truncate_left_pad,
    prepare_tokens_and_labels,
    remove_lora_hooks,
    set_seed,
)

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


# ---------------------------------------------------------------------------
# DiffEmbedder: on-the-fly diff text -> dense embedding
# ---------------------------------------------------------------------------

def _chunk_token_ids(
    token_ids: List[int], chunk_tokens: int, overlap: int,
) -> List[List[int]]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be < chunk_tokens")
    chunks = []
    step = chunk_tokens - overlap
    n = len(token_ids)
    for start in range(0, n, step):
        end = min(start + chunk_tokens, n)
        window = token_ids[start:end]
        if len(window) < 16:
            continue
        chunks.append(window)
        if end >= n:
            break
    return chunks


def _make_text_chunks(
    tokenizer, text: str, chunk_tokens: int, overlap: int,
) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    windows = _chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=overlap)
    return [tokenizer.decode(w, skip_special_tokens=True) for w in windows]


class DiffEmbedder:
    """Wraps a frozen embedding model to produce [2*D] vectors from raw diff text."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0",
        chunk_tokens: int = 512,
        overlap: int = 64,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.max_length = max_length
        self.batch_size = batch_size
        self._embed_dim = model.config.hidden_size
        self._zero = torch.zeros(2 * self._embed_dim)

    @property
    def embed_dim(self) -> int:
        return 2 * self._embed_dim

    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Return a [len(texts), D] tensor on CPU (fp32)."""
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            out = self.model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # Accumulate on-device; only move to CPU once per diff at the end.
            all_vecs.append(mean.detach())
        if not all_vecs:
            return torch.empty((0, self._embed_dim))
        return torch.cat(all_vecs, dim=0).float().cpu()

    def embed_diff(self, diff_text: str) -> torch.Tensor:
        """Embed a single diff text string -> [2*D] vector (MaxPool||MeanPool)."""
        if not diff_text or not diff_text.strip():
            return self._zero.clone()
        chunks = _make_text_chunks(
            self.tokenizer, diff_text, self.chunk_tokens, self.overlap,
        )
        if not chunks:
            return self._zero.clone()
        chunk_embs = self._embed_texts(chunks)  # [K, D] on CPU fp32
        return pool_file_chunks_maxmean(chunk_embs)  # [2*D]


# ---------------------------------------------------------------------------
# Data loading from SQLite DB
# ---------------------------------------------------------------------------

def load_commit_sequences_from_db(
    db_path: str,
    split_repo_ids: Optional[List[str]] = None,
    limit_repos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load per-repo commit sequences and assertions from the SQLite DB.

    Returns list of dicts, each with:
        repo_id: str
        commit_diffs: List[str]  (ordered by commit_index)
        assertions_by_commit: Dict[int, List[Tuple[str, str]]]
            mapping commit_index -> [(prefix, target), ...]
        max_commit_index: int
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if split_repo_ids is not None:
            placeholders = ",".join("?" for _ in split_repo_ids)
            repo_rows = conn.execute(
                f"SELECT DISTINCT repo_id FROM commits WHERE repo_id IN ({placeholders}) "
                "ORDER BY repo_id",
                split_repo_ids,
            ).fetchall()
        else:
            repo_rows = conn.execute(
                "SELECT DISTINCT repo_id FROM commits ORDER BY repo_id"
            ).fetchall()

        repo_ids = [r["repo_id"] for r in repo_rows]
        if limit_repos is not None:
            repo_ids = repo_ids[:limit_repos]

        items = []
        for repo_id in repo_ids:
            commits = conn.execute(
                "SELECT commit_index, production_code_diff FROM commits "
                "WHERE repo_id = ? ORDER BY commit_index",
                (repo_id,),
            ).fetchall()
            if not commits:
                continue

            commit_diffs = [row["production_code_diff"] for row in commits]
            commit_indices = [row["commit_index"] for row in commits]

            assertion_rows = conn.execute(
                "SELECT commit_index, assertion_prefix, assertion_target "
                "FROM assertions WHERE repo_id = ? ORDER BY commit_index",
                (repo_id,),
            ).fetchall()

            assertions_by_commit: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
            for ar in assertion_rows:
                ci = ar["commit_index"]
                prefix = ar["assertion_prefix"]
                target = ar["assertion_target"]
                if prefix and target and not target.lstrip().startswith(","):
                    assertions_by_commit[ci].append((prefix, target))

            if not assertions_by_commit:
                continue

            items.append({
                "repo_id": repo_id,
                "commit_diffs": commit_diffs,
                "commit_indices": commit_indices,
                "assertions_by_commit": dict(assertions_by_commit),
                "max_commit_index": commit_indices[-1],
            })

        return items
    finally:
        conn.close()


def get_assertions_up_to(
    assertions_by_commit: Dict[int, List[Tuple[str, str]]],
    current_commit_index: int,
    splits_by_commit: Optional[Dict[int, List[str]]] = None,
    keep_splits: Optional[List[str]] = None,
    mode: str = "cumulative",
) -> List[Tuple[str, str]]:
    """Collect assertions for the loss at ``current_commit_index``.

    Modes
    -----
    * ``cumulative``: all assertions with ``commit_index <= current_commit_index``
      (optionally filtered by in-repo split).
    * ``new``: only assertions with ``commit_index == current_commit_index``
      (the "delta" introduced at this kept commit).

    If ``splits_by_commit`` + ``keep_splits`` are both provided, assertions
    are filtered so that only those whose in-repo split is in ``keep_splits``
    are returned.
    """
    result: List[Tuple[str, str]] = []
    filter_splits: Optional[set] = (
        set(keep_splits) if (splits_by_commit and keep_splits) else None
    )

    def _emit(ci: int, pairs: List[Tuple[str, str]]) -> None:
        if filter_splits is None:
            result.extend(pairs)
            return
        splits = (splits_by_commit or {}).get(ci, [])
        if len(splits) != len(pairs):
            result.extend(pairs)
            return
        for (pair, s) in zip(pairs, splits):
            if s in filter_splits:
                result.append(pair)

    if mode == "new":
        if current_commit_index in assertions_by_commit:
            _emit(current_commit_index, assertions_by_commit[current_commit_index])
        return result

    # cumulative
    for ci, pairs in assertions_by_commit.items():
        if ci <= current_commit_index:
            _emit(ci, pairs)
    return result


def load_split_repo_ids(splits_dir: Path, split_name: str) -> List[str]:
    """Load repo IDs from a split JSON file (for train/val/test partitioning)."""
    import json
    for prefix in ("gru_", ""):
        path = splits_dir / f"{prefix}{split_name}"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return sorted(data.get("repositories", {}).keys())
    return []


# ---------------------------------------------------------------------------
# LM loss computation
# ---------------------------------------------------------------------------

def compute_lm_loss_batch(
    model: nn.Module,
    assertions: List[Tuple[str, str]],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
    backward: bool = False,
) -> Optional[float]:
    """Compute average cross-entropy loss over a set of (prefix, target) pairs.

    Processes one assertion at a time (batch=1 for the LM).
    When backward=True, calls .backward() on each assertion's scaled loss
    immediately so only one computation graph is alive at a time.
    Returns the average loss as a plain float (always detached).
    """
    if not assertions:
        return None

    n_total = len(assertions)
    total_loss_val = 0.0
    n = 0
    pad_id = tokenizer.pad_token_id

    for i, (prefix, target) in enumerate(assertions):
        tl = prepare_tokens_and_labels(prefix, target, tokenizer)
        tokens, labels = left_truncate_left_pad(
            tl["tokens"], tl["labels"], max_seq_len, pad_id,
        )
        attention_mask = [0 if t == pad_id else 1 for t in tokens]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        attn_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)
        lbl = torch.tensor([labels], dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=lbl)
        loss = out["loss"] if isinstance(out, dict) else out[0]

        if backward:
            is_last = (i == n_total - 1)
            (loss / n_total).backward(retain_graph=not is_last)

        total_loss_val += loss.item()
        n += 1

    if n == 0:
        return None
    return total_loss_val / n


def compute_lm_loss_micro_batched(
    model: nn.Module,
    tokenized_items: List[Dict[str, Any]],
    pad_id: int,
    device: torch.device,
    micro_batch: int,
    backward: bool = False,
) -> Optional[float]:
    """Padded + micro-batched version of :func:`compute_lm_loss_batch`.

    Each element of ``tokenized_items`` has ``tokens`` and ``labels`` as
    already-truncated Python lists / numpy arrays (see
    :func:`_precompute_repo_cache`). We pad each micro-batch to the longest
    sequence inside *that* batch so short assertions don't waste FLOPs on
    empty positions.

    The reported and back-propagated loss is the macro-average over
    assertions — i.e. per-assertion mean CE, then mean over assertions —
    to preserve the semantics of the old per-assertion implementation.
    """
    if not tokenized_items:
        return None

    n_total = len(tokenized_items)

    # Sort by length so each micro-batch has similar-length rows -> less padding.
    order = sorted(range(n_total), key=lambda i: len(tokenized_items[i]["tokens"]))
    items_sorted = [tokenized_items[i] for i in order]
    micro_batches = [
        items_sorted[i : i + max(micro_batch, 1)]
        for i in range(0, n_total, max(micro_batch, 1))
    ]
    n_micro = len(micro_batches)

    loss_accum_val = 0.0
    for bi, batch in enumerate(micro_batches):
        max_len = max(len(b["tokens"]) for b in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for r, b in enumerate(batch):
            toks = b["tokens"]
            lbls = b["labels"]
            L = len(toks)
            offset = max_len - L  # left-pad: target tokens stay at the right
            input_ids[r, offset:] = torch.as_tensor(
                toks, dtype=torch.long, device=device,
            )
            labels[r, offset:] = torch.as_tensor(
                lbls, dtype=torch.long, device=device,
            )
        attn_mask = (input_ids != pad_id).long()

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out["logits"] if isinstance(out, dict) else out.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        V = shift_logits.size(-1)

        per_tok = F.cross_entropy(
            shift_logits.reshape(-1, V).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, -1)
        valid = (shift_labels != -100).to(per_tok.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        per_ex_mean = (per_tok * valid).sum(dim=1) / denom

        if backward:
            is_last = (bi == n_micro - 1)
            (per_ex_mean.sum() / n_total).backward(retain_graph=not is_last)

        loss_accum_val += per_ex_mean.detach().sum().item()

    return loss_accum_val / n_total


# ---------------------------------------------------------------------------
# Per-repo precompute cache (diff embeddings + tokenized assertions)
# ---------------------------------------------------------------------------

def _precompute_repo_cache(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    diff_embedder: "DiffEmbedder",
    log_prefix: str = "",
) -> None:
    """Populate each repo in ``data`` with:

    * ``diff_embs``: a ``[N_commits, 2*D]`` fp16 tensor on CPU. Re-used
      for every epoch and every eval call (the embedder is frozen).
    * ``tokenized_by_commit``: same layout as ``assertions_by_commit``
      but each ``(prefix, target)`` tuple is replaced by a dict with
      ``tokens`` / ``labels`` as numpy int32 arrays, already truncated
      to ``max_seq_len``. Tokenization happens once per assertion
      instead of once per use.

    The code path in :func:`get_assertions_up_to` is generic over the
    per-commit payload, so passing ``tokenized_by_commit`` through the
    same helper works with no downstream changes.
    """
    import numpy as np  # local to avoid polluting the module namespace

    if not data:
        return

    total_commits = sum(len(r["commit_diffs"]) for r in data)
    total_pairs = sum(
        sum(len(v) for v in r["assertions_by_commit"].values()) for r in data
    )
    print(
        f"{log_prefix}[CACHE] precomputing {total_commits} diff "
        f"embeddings + {total_pairs} tokenized assertions..."
    )
    t0 = time.time()

    for repo in data:
        diffs = repo["commit_diffs"]
        embs = []
        with torch.no_grad():
            for d in diffs:
                embs.append(diff_embedder.embed_diff(d))  # [2*D] cpu fp32
        if embs:
            emb_tensor = torch.stack(embs, dim=0).to(torch.float16).contiguous()
        else:
            emb_tensor = torch.empty((0, diff_embedder.embed_dim), dtype=torch.float16)
        repo["diff_embs"] = emb_tensor

        tokenized_by_commit: Dict[int, List[Dict[str, Any]]] = {}
        for ci, pairs in repo["assertions_by_commit"].items():
            tok_list: List[Dict[str, Any]] = []
            for (p, t) in pairs:
                tl = prepare_tokens_and_labels(p, t, tokenizer)
                tokens = tl["tokens"]
                labels = tl["labels"]
                if len(tokens) > max_seq_len:
                    tokens = tokens[-max_seq_len:]
                    labels = labels[-max_seq_len:]
                tok_list.append({
                    "tokens": np.asarray(tokens, dtype=np.int32),
                    "labels": np.asarray(labels, dtype=np.int64),
                })
            tokenized_by_commit[ci] = tok_list
        repo["tokenized_by_commit"] = tokenized_by_commit

    elapsed = time.time() - t0
    print(f"{log_prefix}[CACHE] done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    code2lora_gru: Code2LoRAGRU,
    target_modules_dict: Dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    output_dir: str,
    filename: str = "code2lora_gru_state.pt",
    best_loss: Optional[float] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    payload = {
        "model_state_dict": code2lora_gru.state_dict(),
        "model_config": code2lora_gru.config_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "target_module_keys": list(target_modules_dict.keys()),
        "epoch": epoch,
        "global_step": global_step,
    }
    if best_loss is not None:
        payload["best_eval_loss"] = best_loss
    torch.save(payload, path)
    print(f"  Saved checkpoint -> {path}")


# ---------------------------------------------------------------------------
# Evaluation loop (commit-sequential)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_commit_sequential(
    code2lora_gru: Code2LoRAGRU,
    base_model: nn.Module,
    diff_embedder: DiffEmbedder,
    eval_data: List[Dict],
    target_modules_dict: Dict,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
    max_assertions_per_commit: int = 0,
    keep_splits: Optional[List[str]] = None,
    assertion_mode: str = "cumulative",
    lm_micro_batch: int = 4,
) -> Dict[str, float]:
    """Run commit-sequential evaluation, returning average loss.

    ``keep_splits`` controls assertion filtering by in-repo split
    (e.g. ``['val']`` for in-repo val eval, ``['test']`` for test, or
    ``None`` to use every assertion in the repo).
    ``assertion_mode`` is ``'cumulative'`` (default) or ``'new'``; see
    :func:`get_assertions_up_to` for semantics.
    """
    code2lora_gru.eval()
    base_model.eval()
    scaling = code2lora_gru.lora_generator.lora_scaling

    total_loss = 0.0
    total_repos = 0
    total_commits_with_loss = 0
    total_assertions_evaluated = 0

    pad_id = tokenizer.pad_token_id
    for repo_item in eval_data:
        h = code2lora_gru.compute_h0(
            batch_size=1, device=device, dtype=torch.float32,
        )
        repo_loss = 0.0
        repo_n = 0

        splits_by_commit = repo_item.get("assertion_splits")
        diff_embs = repo_item.get("diff_embs")  # precomputed, optional
        tok_by_commit = repo_item.get("tokenized_by_commit")

        for k, diff_text in enumerate(repo_item["commit_diffs"]):
            ci = repo_item["commit_indices"][k]
            if diff_embs is not None:
                diff_emb = diff_embs[k].unsqueeze(0).to(
                    device=device, dtype=torch.float32, non_blocking=True,
                )
            else:
                diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(
                    device=device, dtype=torch.float32,
                )
            h = code2lora_gru.encode_repository_commit(diff_emb, h)

            # Operate on pretokenized items when available; fall back to raw
            # (prefix, target) tuples + the legacy per-assertion loss fn.
            if tok_by_commit is not None:
                assertions = get_assertions_up_to(
                    tok_by_commit, ci,
                    splits_by_commit=splits_by_commit,
                    keep_splits=keep_splits,
                    mode=assertion_mode,
                )
            else:
                assertions = get_assertions_up_to(
                    repo_item["assertions_by_commit"], ci,
                    splits_by_commit=splits_by_commit,
                    keep_splits=keep_splits,
                    mode=assertion_mode,
                )
            if not assertions:
                continue

            if max_assertions_per_commit > 0 and len(assertions) > max_assertions_per_commit:
                assertions = random.sample(assertions, max_assertions_per_commit)

            lora_params = code2lora_gru.generate_lora_from_h(h)
            if lora_params is None:
                continue

            hooks = apply_lora_hooks(target_modules_dict, lora_params, scaling)
            if tok_by_commit is not None:
                avg_loss = compute_lm_loss_micro_batched(
                    base_model, assertions, pad_id, device,
                    micro_batch=lm_micro_batch, backward=False,
                )
            else:
                avg_loss = compute_lm_loss_batch(
                    base_model, assertions, tokenizer, max_seq_len, device,
                )
            remove_lora_hooks(hooks)

            if avg_loss is not None:
                repo_loss += avg_loss
                repo_n += 1
                total_assertions_evaluated += len(assertions)

        if repo_n > 0:
            total_loss += repo_loss / repo_n
            total_repos += 1
            total_commits_with_loss += repo_n

    avg_loss = total_loss / max(total_repos, 1)
    return {
        "eval_loss": avg_loss,
        "eval_repos": total_repos,
        "eval_commits_with_loss": total_commits_with_loss,
        "eval_assertions": total_assertions_evaluated,
    }


# ---------------------------------------------------------------------------
# Training loop (commit-sequential)
# ---------------------------------------------------------------------------

def _run_eval_suite(
    code2lora_gru: Code2LoRAGRU,
    base_model: nn.Module,
    diff_embedder: DiffEmbedder,
    eval_suites: Dict[str, Dict[str, Any]],
    target_modules_dict: Dict,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
    max_assertions_per_commit: int,
    tag_prefix: str,
    global_step: int,
    assertion_mode: str = "cumulative",
    lm_micro_batch: int = 4,
) -> Tuple[Dict[str, float], float]:
    """Evaluate each suite in ``eval_suites`` and return a flattened dict
    keyed like ``f"{tag_prefix}/{suite_name}/{metric}"`` together with the
    loss used for checkpoint selection (uses the first suite that has data).
    """
    metrics: Dict[str, float] = {}
    selection_loss = float("inf")
    for name, suite in eval_suites.items():
        data = suite.get("data") or []
        if not data:
            continue
        keep_splits = suite.get("keep_splits")
        m = evaluate_commit_sequential(
            code2lora_gru, base_model, diff_embedder, data,
            target_modules_dict, tokenizer, max_seq_len, device,
            max_assertions_per_commit=max_assertions_per_commit,
            keep_splits=keep_splits,
            assertion_mode=assertion_mode,
            lm_micro_batch=lm_micro_batch,
        )
        for k, v in m.items():
            metrics[f"{tag_prefix}/{name}/{k}"] = v
        # Selection: prefer in_repo_val, then cross_repo_val, then any
        if suite.get("is_primary", False) and m["eval_loss"] < selection_loss:
            selection_loss = m["eval_loss"]
    # fallback selection if no primary flagged
    if selection_loss == float("inf"):
        for name, suite in eval_suites.items():
            data = suite.get("data") or []
            if not data:
                continue
            selection_loss = metrics.get(f"{tag_prefix}/{name}/eval_loss", selection_loss)
            break
    return metrics, selection_loss


def _log_eval_metrics(metrics: Dict[str, float], global_step: int) -> None:
    if not metrics:
        return
    # Pretty print
    groups: Dict[str, Dict[str, float]] = defaultdict(dict)
    for k, v in metrics.items():
        parts = k.split("/")
        if len(parts) >= 3:
            groups[parts[1]][parts[2]] = v
        else:
            groups["_"][k] = v
    for name, d in groups.items():
        loss = d.get("eval_loss", float("nan"))
        repos = d.get("eval_repos", 0)
        ca = d.get("eval_commits_with_loss", 0)
        aa = d.get("eval_assertions", 0)
        print(
            f"    [{name}] loss={loss:.4f} repos={int(repos)} "
            f"commits_w_loss={int(ca)} assertions={int(aa)}"
        )
    wandb.log(metrics, step=global_step)


def train_commit_sequential(
    code2lora_gru: Code2LoRAGRU,
    base_model: nn.Module,
    diff_embedder: DiffEmbedder,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    train_data: List[Dict],
    eval_suites: Dict[str, Dict[str, Any]],
    target_modules_dict: Dict,
    tokenizer: AutoTokenizer,
    args,
    device: torch.device,
):
    scaling = code2lora_gru.lora_generator.lora_scaling
    global_step = 0
    best_eval_loss = float("inf")
    max_seq_len = args.max_seq_len
    max_assertions_per_commit = args.max_assertions_per_commit
    lm_micro_batch = max(int(getattr(args, "lm_micro_batch", 4)), 1)
    pad_id = tokenizer.pad_token_id
    train_keep_splits = (
        list(args.train_in_repo_splits) if args.train_in_repo_splits else None
    )

    total_repos_per_epoch = len(train_data)
    total_steps_est = total_repos_per_epoch * args.epochs

    print(f"\nTraining: {total_repos_per_epoch} repos/epoch, "
          f"{args.epochs} epochs, ~{total_steps_est} repo steps")
    print(f"  grad_accum={args.grad_accum} "
          f"max_assertions_per_commit={max_assertions_per_commit} "
          f"train_in_repo_splits={train_keep_splits}")

    for epoch in range(args.epochs):
        code2lora_gru.train()
        base_model.eval()

        random.shuffle(train_data)
        epoch_loss = 0.0
        epoch_repos = 0
        epoch_commits_with_loss = 0
        accum_loss_val = 0.0
        accum_count = 0
        t0 = time.time()

        for ri, repo_item in enumerate(train_data):
            repo_id = repo_item["repo_id"]
            commit_diffs = repo_item["commit_diffs"]
            commit_indices = repo_item["commit_indices"]
            assertions_by_commit = repo_item["assertions_by_commit"]
            splits_by_commit = repo_item.get("assertion_splits")
            diff_embs = repo_item.get("diff_embs")
            tok_by_commit = repo_item.get("tokenized_by_commit")
            n_commits = len(commit_diffs)

            h = code2lora_gru.compute_h0(
                batch_size=1, device=device, dtype=torch.float32,
            )

            repo_loss_val = 0.0
            repo_loss_n = 0

            for k in range(n_commits):
                ci = commit_indices[k]

                if diff_embs is not None:
                    diff_emb = diff_embs[k].unsqueeze(0).to(
                        device=device, dtype=torch.float32, non_blocking=True,
                    )
                else:
                    diff_text = commit_diffs[k]
                    diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(
                        device=device, dtype=torch.float32,
                    )
                h = code2lora_gru.encode_repository_commit(diff_emb, h)

                if tok_by_commit is not None:
                    assertions = get_assertions_up_to(
                        tok_by_commit, ci,
                        splits_by_commit=splits_by_commit,
                        keep_splits=train_keep_splits,
                        mode=args.assertion_mode,
                    )
                else:
                    assertions = get_assertions_up_to(
                        assertions_by_commit, ci,
                        splits_by_commit=splits_by_commit,
                        keep_splits=train_keep_splits,
                        mode=args.assertion_mode,
                    )
                if not assertions:
                    h = h.detach()
                    continue

                if max_assertions_per_commit > 0 and len(assertions) > max_assertions_per_commit:
                    assertions = random.sample(assertions, max_assertions_per_commit)

                lora_params = code2lora_gru.generate_lora_from_h(h)
                if lora_params is None:
                    h = h.detach()
                    continue

                hooks = apply_lora_hooks(target_modules_dict, lora_params, scaling)
                if tok_by_commit is not None:
                    avg_loss = compute_lm_loss_micro_batched(
                        base_model, assertions, pad_id, device,
                        micro_batch=lm_micro_batch, backward=True,
                    )
                else:
                    avg_loss = compute_lm_loss_batch(
                        base_model, assertions, tokenizer, max_seq_len, device,
                        backward=True,
                    )
                remove_lora_hooks(hooks)

                if avg_loss is not None:
                    accum_loss_val += avg_loss
                    accum_count += 1
                    repo_loss_val += avg_loss
                    repo_loss_n += 1

                h = h.detach()

            if repo_loss_n > 0:
                epoch_loss += repo_loss_val / repo_loss_n
                epoch_repos += 1
                epoch_commits_with_loss += repo_loss_n

            if accum_count > 0 and (ri + 1) % args.grad_accum == 0:
                for p in code2lora_gru.parameters():
                    if p.grad is not None:
                        p.grad /= accum_count

                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        code2lora_gru.parameters(), args.max_grad_norm,
                    )
                else:
                    grad_norm = 0.0

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg = accum_loss_val / accum_count
                    elapsed = time.time() - t0
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"  [epoch={epoch} step={global_step} "
                        f"repo={ri+1}/{total_repos_per_epoch}] "
                        f"loss={avg:.4f} grad_norm={grad_norm:.4f} "
                        f"lr={lr_now:.2e} elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                    wandb.log({
                        "train/loss": avg,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": lr_now,
                        "train/epoch": epoch + (ri + 1) / total_repos_per_epoch,
                        "train/global_step": global_step,
                    }, step=global_step)

                accum_loss_val = 0.0
                accum_count = 0

            if (
                args.eval_steps > 0
                and global_step > 0
                and global_step % args.eval_steps == 0
                and eval_suites
            ):
                print(f"\n  Eval at step {global_step}...")
                eval_metrics, sel_loss = _run_eval_suite(
                    code2lora_gru, base_model, diff_embedder, eval_suites,
                    target_modules_dict, tokenizer, max_seq_len, device,
                    max_assertions_per_commit, "eval", global_step,
                    assertion_mode=args.assertion_mode,
                    lm_micro_batch=lm_micro_batch,
                )
                _log_eval_metrics(eval_metrics, global_step)

                if sel_loss < best_eval_loss:
                    best_eval_loss = sel_loss
                    save_checkpoint(
                        code2lora_gru, target_modules_dict, optimizer,
                        epoch, global_step, args.output_dir,
                        filename="code2lora_gru_best.pt",
                        best_loss=best_eval_loss,
                    )
                code2lora_gru.train()

            if (
                args.save_steps > 0
                and global_step > 0
                and global_step % args.save_steps == 0
            ):
                ckpt_dir = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}",
                )
                save_checkpoint(
                    code2lora_gru, target_modules_dict, optimizer,
                    epoch, global_step, ckpt_dir,
                )

        # flush remaining accumulated gradients at end of epoch
        if accum_count > 0:
            for p in code2lora_gru.parameters():
                if p.grad is not None:
                    p.grad /= accum_count
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    code2lora_gru.parameters(), args.max_grad_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_loss_val = 0.0
            accum_count = 0

        epoch_avg_loss = epoch_loss / max(epoch_repos, 1)
        elapsed = time.time() - t0
        print(
            f"\n  Epoch {epoch} done: avg_loss={epoch_avg_loss:.4f} "
            f"repos={epoch_repos} commits_with_loss={epoch_commits_with_loss} "
            f"elapsed={elapsed:.0f}s",
        )
        wandb.log({
            "train/epoch_loss": epoch_avg_loss,
            "train/epoch": epoch + 1,
        }, step=global_step)

        if eval_suites:
            print(f"\n  End-of-epoch eval...")
            eval_metrics, sel_loss = _run_eval_suite(
                code2lora_gru, base_model, diff_embedder, eval_suites,
                target_modules_dict, tokenizer, max_seq_len, device,
                max_assertions_per_commit, "eval_epoch", global_step,
                assertion_mode=args.assertion_mode,
                lm_micro_batch=lm_micro_batch,
            )
            _log_eval_metrics(eval_metrics, global_step)
            if sel_loss < best_eval_loss:
                best_eval_loss = sel_loss
                save_checkpoint(
                    code2lora_gru, target_modules_dict, optimizer,
                    epoch, global_step, args.output_dir,
                    filename="code2lora_gru_best.pt",
                    best_loss=best_eval_loss,
                )
            code2lora_gru.train()

        save_checkpoint(
            code2lora_gru, target_modules_dict, optimizer,
            epoch, global_step, args.output_dir,
            filename=f"code2lora_gru_epoch{epoch}.pt",
        )

    save_checkpoint(
        code2lora_gru, target_modules_dict, optimizer,
        args.epochs - 1, global_step, args.output_dir,
        filename="code2lora_gru_final.pt",
        best_loss=best_eval_loss,
    )
    print(f"\nTraining complete. global_step={global_step} "
          f"best_eval_loss={best_eval_loss:.4f}")
    return global_step, best_eval_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train Code2LoRA-GRU (commit-sequential)",
    )

    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_parquet_dir = os.path.join(default_dataset, "commit_parquet")
    default_db = os.path.join(default_dataset, "commits_assertions.db")

    # Data source (mutually-exclusive at the semantic level)
    ap.add_argument("--parquet-dir", type=str, default=default_parquet_dir,
                    help="Directory containing commits.parquet + qna_pairs.parquet "
                         "(or shards/*.parquet). Preferred data source.")
    ap.add_argument("--commits-parquet", type=str, default=None,
                    help="Explicit path to commits.parquet (overrides --parquet-dir).")
    ap.add_argument("--qna-parquet", type=str, default=None,
                    help="Explicit path to qna_pairs.parquet (overrides --parquet-dir).")
    ap.add_argument("--parquet-prefer", type=str, default="auto",
                    choices=["auto", "concat", "shards", "hf"],
                    help="Which parquet layout to read from inside --parquet-dir.")
    ap.add_argument("--db-path", type=str, default=default_db,
                    help="[Legacy] Path to commits_assertions.db (SQLite).")
    ap.add_argument("--data-source", type=str, default="parquet",
                    choices=["parquet", "db"],
                    help="Which dataset backend to use.")

    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Split JSONs dir (only used by the legacy DB backend).")
    ap.add_argument("--limit-train-repos", type=int, default=None)
    ap.add_argument("--limit-eval-repos", type=int, default=None)

    # In-repo / cross-repo split controls (parquet backend)
    ap.add_argument("--train-in-repo-splits", nargs="+", default=["train"],
                    help="Assertions with these in-repo splits are used for the "
                         "training loss.")
    ap.add_argument("--in-repo-val-splits", nargs="+", default=["val"],
                    help="Assertions used for the in-repo val eval on training repos.")
    ap.add_argument("--in-repo-test-splits", nargs="+", default=["test"],
                    help="Assertions used for the in-repo test eval on training repos.")
    ap.add_argument("--cross-repo-eval-splits", nargs="+",
                    default=["cr_val", "cr_test"],
                    help="cross_repo_split values to form the held-out eval sets.")
    ap.add_argument("--no-in-repo-eval", action="store_true",
                    help="Disable the in-repo val/test evals (kept for ablations).")
    ap.add_argument("--no-cross-repo-eval", action="store_true",
                    help="Disable the cross-repo val/test evals.")
    ap.add_argument(
        "--assertion-mode", type=str, default="cumulative",
        choices=["cumulative", "new"],
        help=(
            "How to pick assertions at commit k. "
            "'cumulative' (default): all assertions with commit_index <= k "
            "matching the split filter. 'new': only assertions newly "
            "introduced at commit_index == k. 'new' is faster and closer "
            "to the dataset's semantics (kept commits == commits that add "
            "new assertions); 'cumulative' stresses long-horizon retention."
        ),
    )
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                    help="Embedding model for on-the-fly diff encoding")
    ap.add_argument(
        "--output-dir", type=str,
        default="/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU/commit_level",
    )

    # GRU architecture
    ap.add_argument("--gru-hidden-dim", type=int, default=1024)
    ap.add_argument("--gru-num-layers", type=int, default=1)
    ap.add_argument("--gru-dropout", type=float, default=0.0)
    ap.add_argument("--bptt-window", type=int, default=32)
    ap.add_argument(
        "--init-type", type=str, default="zeros",
        choices=["mamba2", "meanpool", "zeros"],
    )

    # LoRA generator
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora-hidden-dim", type=int, default=512)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--trunk-depth", type=int, default=2)

    # Embedding
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)

    # Training
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument(
        "--max-assertions-per-commit",
        type=int,
        default=0,
        help="Max assertions per commit step (0 = use all cumulative assertions at that commit)",
    )
    ap.add_argument(
        "--lm-micro-batch",
        type=int,
        default=4,
        help=(
            "Assertions forwarded together in one LM call. Each micro-batch "
            "is padded only to its longest member, so this controls the "
            "per-step throughput / memory trade-off. 1 = legacy behavior."
        ),
    )
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Number of repos between optimizer steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-steps", type=int, default=200,
                    help="Evaluate every N optimizer steps (0 to disable)")
    ap.add_argument(
        "--no-initial-eval",
        action="store_true",
        help="Skip eval on cr_val before training starts",
    )
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)

    target_modules_default = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ]
    ap.add_argument("--target-modules", nargs="+", default=target_modules_default)

    args = ap.parse_args()
    set_seed(args.seed)

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    eval_suites: Dict[str, Dict[str, Any]] = {}

    if args.data_source == "parquet":
        print("=" * 80)
        print("[DATA] Loading commit sequences from Parquet...")
        sources = resolve_parquet_sources(
            parquet_dir=args.parquet_dir,
            commits_path=args.commits_parquet,
            qna_path=args.qna_parquet,
            prefer=args.parquet_prefer,
        )
        print(f"  parquet source: kind={sources.source_kind} "
              f"commits={len(sources.commits_paths)} qna={len(sources.qna_paths)}")

        # Training set: cross_repo_split='train' repos, all kept commits,
        # but assertion loss filtered by --train-in-repo-splits.
        train_data = load_commit_sequences_from_parquet(
            sources,
            cross_repo_splits=["train"],
            in_repo_splits=None,  # load every assertion; filter at loss time
            limit_repos=args.limit_train_repos,
        )

        if not args.no_in_repo_eval:
            # Reuse training repos; cap to --limit-eval-repos for speed.
            if args.limit_eval_repos is not None:
                in_repo_val = train_data[: args.limit_eval_repos]
            else:
                in_repo_val = train_data
            eval_suites["in_repo_val"] = {
                "data": in_repo_val,
                "keep_splits": list(args.in_repo_val_splits),
                "is_primary": True,
            }
            eval_suites["in_repo_test"] = {
                "data": in_repo_val,  # same repo sequences, different keep_splits
                "keep_splits": list(args.in_repo_test_splits),
                "is_primary": False,
            }

        if not args.no_cross_repo_eval and args.cross_repo_eval_splits:
            for cr in args.cross_repo_eval_splits:
                data = load_commit_sequences_from_parquet(
                    sources,
                    cross_repo_splits=[cr],
                    in_repo_splits=None,
                    limit_repos=args.limit_eval_repos,
                )
                eval_suites[f"cross_repo_{cr}"] = {
                    "data": data,
                    "keep_splits": None,  # use every assertion on held-out repos
                    "is_primary": (cr == "cr_val" and args.no_in_repo_eval),
                }
    else:  # legacy SQLite DB backend
        db_path = Path(args.db_path).expanduser().resolve()
        if not db_path.exists():
            raise FileNotFoundError(
                f"DB not found: {db_path}\n"
                f"Run: python create_dataset/build_commit_assertion_db.py "
                f"--db-path {db_path}"
            )
        print("Loading split repo IDs...")
        train_repo_ids = load_split_repo_ids(splits_dir, "train.json")
        eval_repo_ids = load_split_repo_ids(splits_dir, "cr_val.json")
        print(f"  Split files: train={len(train_repo_ids)} repos, "
              f"eval={len(eval_repo_ids)} repos")
        print(f"Loading commit sequences from {db_path}...")
        train_data = load_commit_sequences_from_db(
            str(db_path),
            split_repo_ids=train_repo_ids if train_repo_ids else None,
            limit_repos=args.limit_train_repos,
        )
        eval_data = load_commit_sequences_from_db(
            str(db_path),
            split_repo_ids=eval_repo_ids if eval_repo_ids else None,
            limit_repos=args.limit_eval_repos,
        )
        if eval_data:
            eval_suites["cross_repo_cr_val"] = {
                "data": eval_data, "keep_splits": None, "is_primary": True,
            }

    # ── Summaries ──
    def _n_a(d: Dict) -> int:
        return sum(len(v) for v in d["assertions_by_commit"].values())

    total_train_commits = sum(len(d["commit_diffs"]) for d in train_data)
    total_train_assertions = sum(_n_a(d) for d in train_data)
    print(
        f"  Train:      {len(train_data)} repos, {total_train_commits} commits, "
        f"{total_train_assertions} assertions"
    )
    for name, suite in eval_suites.items():
        data = suite.get("data") or []
        if not data:
            print(f"  {name}: EMPTY (disabled)")
            continue
        nc = sum(len(d["commit_diffs"]) for d in data)
        na = sum(_n_a(d) for d in data)
        print(f"  {name}: {len(data)} repos, {nc} commits, {na} assertions "
              f"(keep={suite.get('keep_splits')})")

    if not train_data:
        raise ValueError(
            "No training data loaded. Check --parquet-dir / --db-path.",
        )

    # ── Config ──
    print("=" * 80)
    print("[CONFIG]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 80, flush=True)

    wandb.init(
        project="code2lora-gru-commits",
        name=args.output_dir.split("/")[-1],
        config=vars(args),
    )

    # ── Load embedding model (frozen) ──
    print(f"Loading embedding model: {args.embed_model}...")
    embed_tokenizer = AutoTokenizer.from_pretrained(
        args.embed_model, trust_remote_code=True,
    )
    embed_model = AutoModel.from_pretrained(
        args.embed_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    for p in embed_model.parameters():
        p.requires_grad = False

    diff_embedder = DiffEmbedder(
        model=embed_model,
        tokenizer=embed_tokenizer,
        device="cuda:0",
        chunk_tokens=args.chunk_tokens,
        overlap=args.chunk_overlap,
    )
    file_embed_dim = diff_embedder.embed_dim
    print(f"  Diff embed dim: {file_embed_dim} (2 * {diff_embedder._embed_dim})")

    # ── Load tokenizer and frozen base LLM ──
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading frozen base model: {args.model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    for p in base_model.parameters():
        p.requires_grad = False
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Base model: {total_base_params:,} params (frozen)")

    # ── Discover target modules ──
    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, args.target_modules,
    )
    print(f"  {len(target_modules_dict)} target modules across {num_layers} layers")

    # ── Build Code2LoRA-GRU ──
    device = torch.device("cuda:0")
    code2lora_gru = Code2LoRAGRU(
        file_embed_dim=file_embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        num_target_layers=num_layers,
        module_dims=module_dims,
        lora_hidden_dim=args.lora_hidden_dim,
        lora_rank=args.rank,
        lora_alpha=float(args.alpha),
        lora_num_bases=args.num_bases,
        lora_trunk_depth=args.trunk_depth,
        init_type=args.init_type,
        gru_dropout=args.gru_dropout,
        bptt_window=args.bptt_window,
    ).to(device=device, dtype=torch.float32)

    n_params = sum(p.numel() for p in code2lora_gru.parameters())
    print(f"\nCode2LoRA-GRU: {n_params:,} params ({n_params / 1e6:.2f}M)")

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        code2lora_gru.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps_est = len(train_data) * args.epochs // max(args.grad_accum, 1)
    warmup_steps = int(total_steps_est * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-2)
        progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"  Optimizer: AdamW lr={args.lr} wd={args.weight_decay}")
    print(f"  Scheduler: cosine with {warmup_steps} warmup / ~{total_steps_est} total steps")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Precompute per-repo caches ────────────────────────────────────────
    # 1) Diff embeddings: embedder is frozen, so running it once per diff
    #    (vs once per epoch + once per eval_steps) removes the biggest
    #    redundant compute in both train and eval.
    # 2) Tokenized assertions: prepare_tokens_and_labels is called once per
    #    (prefix, target) pair; in cumulative mode the same pair can appear
    #    in the loss at many commits, so tokenizing per-use was wasteful.
    _precompute_repo_cache(
        train_data, tok, args.max_seq_len, diff_embedder, log_prefix="  train "
    )
    seen_ids: set = set()
    seen_ids.update(id(r) for r in train_data)  # skip dups that share repos
    for name, suite in eval_suites.items():
        data = suite.get("data") or []
        fresh = [r for r in data if id(r) not in seen_ids]
        if fresh:
            _precompute_repo_cache(
                fresh, tok, args.max_seq_len, diff_embedder,
                log_prefix=f"  {name} ",
            )
            seen_ids.update(id(r) for r in fresh)

    # ── Initial eval ──
    if eval_suites and not args.no_initial_eval:
        print("\nInitial eval...")
        init_metrics, _ = _run_eval_suite(
            code2lora_gru, base_model, diff_embedder, eval_suites,
            target_modules_dict, tok, args.max_seq_len, device,
            args.max_assertions_per_commit, "eval_init", 0,
            assertion_mode=args.assertion_mode,
            lm_micro_batch=args.lm_micro_batch,
        )
        _log_eval_metrics(init_metrics, 0)
    elif eval_suites and args.no_initial_eval:
        print("\nSkipping initial eval (--no-initial-eval).")

    # ── Train ──
    global_step, best_loss = train_commit_sequential(
        code2lora_gru=code2lora_gru,
        base_model=base_model,
        diff_embedder=diff_embedder,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        eval_suites=eval_suites,
        target_modules_dict=target_modules_dict,
        tokenizer=tok,
        args=args,
        device=device,
    )

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
