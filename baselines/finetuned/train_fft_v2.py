#!/usr/bin/env python3
"""Full fine-tune baseline trained on the **v2 smart-cap train QnAs**.

Trains Qwen2.5-Coder-1.5B on the same flat QnA pool the Code2LoRA-GRU v2
trainer consumes (``$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet``),
so that the FFT baseline sees an identical training distribution and is
directly comparable to Code2LoRA-static-v2 and Code2LoRA-GRU-v2 in Table 1.

Tokenization matches the v2 Code2LoRA trainers: **left-truncate, left-pad**
with the assertion target at the rightmost positions, loss masked on the
prefix tokens (-100).

Inputs:
    --qna-parquet  Default: $SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet

Outputs:
    --output-dir
        ├── checkpoint-{step}/ ...
        └── final/               # AutoModelForCausalLM-compatible directory
                                 # consumed by ``evaluation/run_baselines_v2.py``
                                 # via ``--method fft --ckpt .../final``.

Usage::

    sbatch scripts/slurm/train_fft_v2.sh
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
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_QNA = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet"


# ---------------------------------------------------------------------------
# Streaming parquet -> in-memory list of {prefix, target} (chars + minimal cols)
# ---------------------------------------------------------------------------

def load_train_qnas(path: Path, max_rows: int = 0) -> List[Dict[str, str]]:
    """Stream prefix/target from a v2 QnA parquet, optionally capped."""
    pf = pq.ParquetFile(str(path))
    print(f"[load] {path} -> {pf.metadata.num_rows:,} rows", flush=True)
    out: List[Dict[str, str]] = []
    for batch in pf.iter_batches(batch_size=8_192, columns=["prefix", "target"]):
        p = batch.column("prefix").to_pylist()
        t = batch.column("target").to_pylist()
        for pi, ti in zip(p, t):
            if not pi or not ti:
                continue
            ts = ti.lstrip()
            if not ts or ts.startswith(","):
                continue
            out.append({"prefix": pi, "target": ti})
            if max_rows and len(out) >= max_rows:
                return out
    return out


class QnaTorchDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, str]:
        return self.rows[i]


# ---------------------------------------------------------------------------
# Collator: left-truncate / left-pad, mask prefix
# ---------------------------------------------------------------------------

class LeftTruncLeftPadCollator:
    def __init__(self, tokenizer, max_seq_len: int = 4096):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id or 0
        self.eos = tokenizer.eos_token or ""

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        ids_list: List[List[int]] = []
        lbl_list: List[List[int]] = []
        for ex in batch:
            t_ids = self.tok(ex["target"] + self.eos,
                             add_special_tokens=False)["input_ids"]
            if not t_ids:
                continue
            budget = max(8, self.max_seq_len - len(t_ids))
            p_full = self.tok(ex["prefix"], add_special_tokens=False)["input_ids"]
            p_ids = p_full[-budget:] if len(p_full) > budget else p_full
            ids = p_ids + t_ids
            labels = ([-100] * len(p_ids)) + list(t_ids)
            ids_list.append(ids)
            lbl_list.append(labels)
        if not ids_list:
            return {}
        L = max(len(x) for x in ids_list)
        B = len(ids_list)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
            n = len(ids)
            input_ids[i, L - n:] = torch.tensor(ids, dtype=torch.long)
            labels[i, L - n:] = torch.tensor(lbl, dtype=torch.long)
            attn[i, L - n:] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qna-parquet", default=DEFAULT_QNA)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--save-every-steps", type=int, default=2000)
    ap.add_argument("--log-every-steps", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading model {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Trainable: every parameter (FFT).
    for p in model.parameters():
        p.requires_grad = True

    print(f"Loading QnAs from {args.qna_parquet} ...", flush=True)
    rows = load_train_qnas(Path(args.qna_parquet), max_rows=args.max_train_rows)
    print(f"  -> {len(rows):,} train QnAs", flush=True)

    ds = QnaTorchDataset(rows)
    collator = LeftTruncLeftPadCollator(tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
        pin_memory=True, drop_last=True,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, fused=True,
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"  total_steps={total_steps}  warmup={warmup_steps}", flush=True)

    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_tok = 0
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            if not batch:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            ntok = (batch["labels"] != -100).sum().item()
            loss = out.loss * ntok
            (loss / args.grad_accum).backward()
            running_loss += loss.detach().item()
            running_tok += ntok
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every_steps == 0:
                    avg = running_loss / max(running_tok, 1)
                    elapsed = (time.time() - t0) / 60
                    print(f"[ep{epoch} it{it+1}/{len(loader)} step{global_step}/"
                          f"{total_steps}] loss={avg:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} "
                          f"elapsed={elapsed:.1f}m", flush=True)
                    running_loss = 0.0
                    running_tok = 0
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    p = out_dir / f"checkpoint-{global_step}"
                    p.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(p)
                    tokenizer.save_pretrained(p)
                    print(f"  [ckpt] -> {p}", flush=True)
        # End of epoch.
        ep_dir = out_dir / f"checkpoint-ep{epoch}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ep_dir)
        tokenizer.save_pretrained(ep_dir)
        print(f"  [ckpt] end of epoch -> {ep_dir}", flush=True)

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nDone. Final FFT model -> {final_dir}")


if __name__ == "__main__":
    main()
