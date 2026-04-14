#!/usr/bin/env python3
"""
Commit-level training for Code2LoRA-GRU: iterate over each commit in a
repository's history, embed the diff on-the-fly, update the GRU hidden state,
generate a LoRA, and compute loss on assertions available up to that commit.

Data source: SQLite database built by create_dataset/build_commit_assertion_db.py
    commits(repo_id, commit_index, commit_sha, production_code_diff)
    assertions(repo_id, commit_index, assertion_prefix, assertion_target)

Training loop per repository:
  1. Initialize h_0 (zeros or from initializer).
  2. For each commit k in chronological order:
     a. Embed production_code_diff via frozen Qwen3-Embedding (chunk + MaxPool||MeanPool).
     b. GRU step: h_k = encode_repository_commit(diff_emb_k, h_{k-1}).
     c. Collect assertions where commit_index <= k (cumulative).
     d. If assertions exist: generate LoRA from h_k, apply hooks, compute LM loss.
     e. Detach h_k for truncated BPTT.
  3. Backprop accumulated loss, step optimizer.

Usage:
    python hypernetwork/train_code2lora_gru_commits.py \\
        --db-path $SCRATCH/REPO_DATASET/commits_assertions.db
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
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_vecs.append(mean.detach().cpu())
        if not all_vecs:
            return torch.empty((0, self._embed_dim))
        return torch.cat(all_vecs, dim=0)

    def embed_diff(self, diff_text: str) -> torch.Tensor:
        """Embed a single diff text string -> [2*D] vector (MaxPool||MeanPool)."""
        if not diff_text or not diff_text.strip():
            return self._zero.clone()
        chunks = _make_text_chunks(
            self.tokenizer, diff_text, self.chunk_tokens, self.overlap,
        )
        if not chunks:
            return self._zero.clone()
        chunk_embs = self._embed_texts(chunks)  # [K, D]
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
) -> List[Tuple[str, str]]:
    """Collect all assertions with commit_index <= current_commit_index."""
    result = []
    for ci, pairs in assertions_by_commit.items():
        if ci <= current_commit_index:
            result.extend(pairs)
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
) -> Dict[str, float]:
    """Run commit-sequential evaluation, returning average loss."""
    code2lora_gru.eval()
    base_model.eval()
    scaling = code2lora_gru.lora_generator.lora_scaling

    total_loss = 0.0
    total_repos = 0
    total_commits_with_loss = 0

    for repo_item in eval_data:
        h = code2lora_gru.compute_h0(
            batch_size=1, device=device, dtype=torch.float32,
        )
        repo_loss = 0.0
        repo_n = 0

        for k, diff_text in enumerate(repo_item["commit_diffs"]):
            ci = repo_item["commit_indices"][k]
            diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(device=device, dtype=torch.float32)
            h = code2lora_gru.encode_repository_commit(diff_emb, h)

            assertions = get_assertions_up_to(
                repo_item["assertions_by_commit"], ci,
            )
            if not assertions:
                continue

            if max_assertions_per_commit > 0 and len(assertions) > max_assertions_per_commit:
                assertions = random.sample(assertions, max_assertions_per_commit)

            lora_params = code2lora_gru.generate_lora_from_h(h)
            if lora_params is None:
                continue

            hooks = apply_lora_hooks(target_modules_dict, lora_params, scaling)
            avg_loss = compute_lm_loss_batch(
                base_model, assertions, tokenizer, max_seq_len, device,
            )
            remove_lora_hooks(hooks)

            if avg_loss is not None:
                repo_loss += avg_loss
                repo_n += 1

        if repo_n > 0:
            total_loss += repo_loss / repo_n
            total_repos += 1
            total_commits_with_loss += repo_n

    avg_loss = total_loss / max(total_repos, 1)
    return {
        "eval_loss": avg_loss,
        "eval_repos": total_repos,
        "eval_commits_with_loss": total_commits_with_loss,
    }


# ---------------------------------------------------------------------------
# Training loop (commit-sequential)
# ---------------------------------------------------------------------------

def train_commit_sequential(
    code2lora_gru: Code2LoRAGRU,
    base_model: nn.Module,
    diff_embedder: DiffEmbedder,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    train_data: List[Dict],
    eval_data: Optional[List[Dict]],
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

    total_repos_per_epoch = len(train_data)
    total_steps_est = total_repos_per_epoch * args.epochs

    print(f"\nTraining: {total_repos_per_epoch} repos/epoch, "
          f"{args.epochs} epochs, ~{total_steps_est} repo steps")
    print(f"  grad_accum={args.grad_accum} "
          f"max_assertions_per_commit={max_assertions_per_commit}")

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
            n_commits = len(commit_diffs)

            h = code2lora_gru.compute_h0(
                batch_size=1, device=device, dtype=torch.float32,
            )

            repo_loss_val = 0.0
            repo_loss_n = 0

            for k in range(n_commits):
                ci = commit_indices[k]
                diff_text = commit_diffs[k]

                diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(device=device, dtype=torch.float32)
                h = code2lora_gru.encode_repository_commit(diff_emb, h)

                assertions = get_assertions_up_to(assertions_by_commit, ci)
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
                and eval_data
            ):
                print(f"\n  Eval at step {global_step}...")
                eval_metrics = evaluate_commit_sequential(
                    code2lora_gru, base_model, diff_embedder, eval_data,
                    target_modules_dict, tokenizer, max_seq_len, device,
                    max_assertions_per_commit=max_assertions_per_commit,
                )
                eval_loss = eval_metrics["eval_loss"]
                print(
                    f"  eval_loss={eval_loss:.4f} "
                    f"repos={eval_metrics['eval_repos']} "
                    f"commits_with_loss={eval_metrics['eval_commits_with_loss']}",
                )
                wandb.log(
                    {"eval/loss": eval_loss, **{f"eval/{k}": v for k, v in eval_metrics.items()}},
                    step=global_step,
                )

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
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

        if eval_data:
            print(f"\n  End-of-epoch eval...")
            eval_metrics = evaluate_commit_sequential(
                code2lora_gru, base_model, diff_embedder, eval_data,
                target_modules_dict, tokenizer, max_seq_len, device,
                max_assertions_per_commit=max_assertions_per_commit,
            )
            eval_loss = eval_metrics["eval_loss"]
            print(
                f"  eval_loss={eval_loss:.4f} "
                f"repos={eval_metrics['eval_repos']} "
                f"commits_with_loss={eval_metrics['eval_commits_with_loss']}",
            )
            wandb.log(
                {"eval/loss": eval_loss, **{f"eval/{k}": v for k, v in eval_metrics.items()}},
                step=global_step,
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
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
        description="Train Code2LoRA-GRU (commit-sequential, DB-backed)",
    )

    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_db = os.path.join(default_dataset, "commits_assertions.db")

    ap.add_argument("--db-path", type=str, default=default_db,
                    help="Path to commits_assertions.db")
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Split JSONs dir (for train/val/test repo partitioning)")
    ap.add_argument("--limit-train-repos", type=int, default=None)
    ap.add_argument("--limit-eval-repos", type=int, default=None)
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

    db_path = Path(args.db_path).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()

    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found: {db_path}\n"
            f"Run: python create_dataset/build_commit_assertion_db.py --db-path {db_path}"
        )

    # ── Load repo IDs from split files for train/eval partitioning ──
    print("Loading split repo IDs...")
    train_repo_ids = load_split_repo_ids(splits_dir, "train.json")
    eval_repo_ids = load_split_repo_ids(splits_dir, "cr_val.json")
    print(f"  Split files: train={len(train_repo_ids)} repos, eval={len(eval_repo_ids)} repos")

    # ── Load commit sequences from DB ──
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

    total_train_commits = sum(len(d["commit_diffs"]) for d in train_data)
    total_train_assertions = sum(
        sum(len(v) for v in d["assertions_by_commit"].values())
        for d in train_data
    )
    print(
        f"  Train: {len(train_data)} repos, {total_train_commits} commits, "
        f"{total_train_assertions} assertions"
    )
    if eval_data:
        total_eval_commits = sum(len(d["commit_diffs"]) for d in eval_data)
        print(f"  Eval:  {len(eval_data)} repos, {total_eval_commits} commits")

    if not train_data:
        raise ValueError("No training data loaded from DB. Check --db-path and --splits-dir.")

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
        torch_dtype=torch.float16,
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
        attn_implementation="flash_attention_2",
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

    # ── Initial eval ──
    if eval_data and not args.no_initial_eval:
        print("\nInitial eval...")
        init_metrics = evaluate_commit_sequential(
            code2lora_gru, base_model, diff_embedder, eval_data,
            target_modules_dict, tok, args.max_seq_len, device,
            max_assertions_per_commit=args.max_assertions_per_commit,
        )
        print(f"  init_eval_loss={init_metrics['eval_loss']:.4f}")
        wandb.log({"eval/init_loss": init_metrics["eval_loss"]})
    elif eval_data and args.no_initial_eval:
        print("\nSkipping initial eval (--no-initial-eval).")

    # ── Train ──
    global_step, best_loss = train_commit_sequential(
        code2lora_gru=code2lora_gru,
        base_model=base_model,
        diff_embedder=diff_embedder,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        eval_data=eval_data if eval_data else None,
        target_modules_dict=target_modules_dict,
        tokenizer=tok,
        args=args,
        device=device,
    )

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
