#!/usr/bin/env python3
"""Single LoRA baseline trained on the **v2 smart-cap train QnAs**.

Identical training distribution to ``train_fft_v2.py`` and to the Code2LoRA-GRU
v2 trainer, but only a PEFT LoRA (rank=16, alpha=32) is trained on top of a
frozen Qwen2.5-Coder-1.5B. The adapter is shared across all training repos
(hence "single" LoRA).

Tokenization matches the v2 trainers exactly: **left-truncate, left-pad**,
loss masked on the prefix tokens.

Output adapter directory is consumed by ``evaluation/run_baselines_v2.py``
via ``--method slora --ckpt <output-dir>/final``.

Usage::

    sbatch scripts/slurm/train_slora_v2.sh
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Reuse the dataset + collator from the FFT trainer (same data format).
sys.path.insert(0, str(_HERE.parent / "finetuned"))
from train_fft_v2 import (  # noqa: E402
    LeftTruncLeftPadCollator,
    QnaTorchDataset,
    load_train_qnas,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_QNA = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qna-parquet", default=DEFAULT_QNA)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)

    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
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

    print(f"Loading base model {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=list(args.target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        weight_decay=args.weight_decay, fused=True,
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
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
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
                    p = out_dir / f"adapter-{global_step}"
                    model.save_pretrained(p)
                    print(f"  [ckpt] -> {p}", flush=True)
        ep_dir = out_dir / f"adapter-ep{epoch}"
        model.save_pretrained(ep_dir)
        print(f"  [ckpt] end of epoch -> {ep_dir}", flush=True)

    final_dir = out_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nDone. SLoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
