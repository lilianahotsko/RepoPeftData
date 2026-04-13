#!/usr/bin/env python3
"""
Text2LoRA **SFT** training with pre-computed CODE embeddings on RepoPeftBench.

This is the strongest Text2LoRA variant: the hypernetwork generates LoRA
weights from a per-repo code embedding, and is trained end-to-end with
language-modeling loss on assertion-completion tasks (no oracle LoRAs needed).

Architecture (unchanged from T2L reconstruction variant):
  - TaskEncoder: Linear(2048 → 128) + LayerNorm
  - Layer-depth & layer-type embeddings
  - MLP trunk (mixer + 2 residual blocks + projection)
  - Per-module output heads  → LoRA A,B for all 7 module types × 28 layers

Training:
  - LoRA weights are generated once per example and applied via forward hooks
  - SFT loss (cross-entropy on target tokens only) backpropagates through
    hooks into the hypernetwork
  - Base model is frozen; only hypernetwork parameters are trained

Usage (from project root):
    python baselines/text2lora/train_code_sft.py \
        text2lora/configs/repopeft_code_sft.yaml
"""

import gc
import json
import os
import random
import shutil
import string
import sys
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import set_seed, get_scheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "text2lora", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.hyper_modulator import create_hypermod, save_hypermod_checkpoint
from hyper_llm_modulator.sft_trainer import (
    get_loss_batch,
    trl_activate_neftune,
    evaluating,
)
from hyper_llm_modulator.utils import (
    get_layers,
    get_num_params,
    create_logger,
    save_yaml,
    get_model_and_tokenizer,
)
from evaluation.data_utils import get_default_splits_dir, load_split

TARGET_MARKER = "### Target:"


# ---------------------------------------------------------------------------
# Dataset & sampling
# ---------------------------------------------------------------------------


class RepoPeftSFTDataset(Dataset):
    """Per-repo dataset: all examples share the same task embedding."""

    def __init__(self, examples: list[dict], task_emb: torch.Tensor):
        self.examples = examples
        self.task_emb = task_emb

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
            "task_emb": self.task_emb,
        }


class HierarchicalRepoSampler(Sampler):
    """Sample *n_tasks* repos then *n_points* examples per repo per batch."""

    def __init__(self, concat_dataset: ConcatDataset, n_tasks: int, n_points: int):
        self.concat_dataset = concat_dataset
        self.n_tasks = n_tasks
        self.n_points = n_points
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.n_datasets = len(self.cumulative_sizes)
        self.ds_sizes = [len(ds) for ds in concat_dataset.datasets]

    def __len__(self):
        return self.n_datasets // self.n_tasks

    def __iter__(self):
        task_order = torch.randperm(self.n_datasets)
        for i in range(0, self.n_datasets, self.n_tasks):
            if i + self.n_tasks > self.n_datasets:
                break
            batch = []
            for j in range(i, i + self.n_tasks):
                ds_idx = task_order[j].item()
                ds_size = self.ds_sizes[ds_idx]
                local_idx = torch.randint(0, ds_size, (self.n_points,))
                offset = self.cumulative_sizes[ds_idx] - ds_size
                batch.extend((local_idx + offset).tolist())
            yield batch


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def _find_marker(input_ids: list[int], marker_variants: list[list[int]]) -> int:
    """Return index of the first token AFTER the marker, or 0."""
    for marker in marker_variants:
        for i in range(len(input_ids) - len(marker) + 1):
            if input_ids[i : i + len(marker)] == marker:
                return i + len(marker)
    return 0


def tokenize_qa(prefix: str, target: str, tokenizer, max_length: int) -> dict | None:
    full_text = f"{prefix}\n{TARGET_MARKER}\n{target}"
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    ids = enc["input_ids"]
    if len(ids) < 10:
        return None

    marker_v1 = tokenizer.encode(f"\n{TARGET_MARKER}\n", add_special_tokens=False)
    marker_v2 = tokenizer.encode(f"{TARGET_MARKER}\n", add_special_tokens=False)
    tgt_start = _find_marker(ids, [marker_v1, marker_v2])

    labels = list(ids)
    for i in range(tgt_start):
        labels[i] = -100

    return {
        "input_ids": ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def slug(name: str) -> str:
    return name.replace("/", "__")


def collate_fn(inp_list, tokenizer):
    labels = [x.pop("labels") for x in inp_list]
    task_embs = torch.stack([x.pop("task_emb") for x in inp_list])

    padded = tokenizer.pad(inp_list, padding=True, pad_to_multiple_of=8, return_tensors="pt")
    lab_padded = tokenizer.pad({"input_ids": labels}, padding=True, pad_to_multiple_of=8, return_tensors="pt")[
        "input_ids"
    ]
    lab_padded = torch.where(padded["attention_mask"] == 0, -100, lab_padded)

    return {**padded, "labels": lab_padded, "task_embs": task_embs}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def build_datasets(split_items, code_embs, tokenizer, max_length):
    """Group items by repo, tokenize, return list of RepoPeftSFTDataset."""
    repo_groups: dict[str, list] = defaultdict(list)
    for item in split_items:
        repo_groups[item["repo"]].append(item)

    ds_list = []
    n_skipped = 0
    for repo, pairs in sorted(repo_groups.items()):
        repo_slug = slug(repo)
        if repo_slug not in code_embs:
            n_skipped += 1
            continue
        emb = code_embs[repo_slug].squeeze(0)  # [emb_dim]
        examples = []
        for p in pairs:
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target or target.lstrip().startswith(","):
                continue
            tok = tokenize_qa(prefix, target, tokenizer, max_length)
            if tok is not None:
                examples.append(tok)
        if examples:
            ds_list.append(RepoPeftSFTDataset(examples, emb))

    logger.info(f"  {len(ds_list)} repo datasets, {n_skipped} skipped (no embedding)")
    return ds_list


def build_loaders(args, code_embs, tokenizer, splits_dir):
    logger.info("Loading RepoPeftBench train split …")
    train_items = load_split(Path(splits_dir), "train")
    logger.info("Loading RepoPeftBench ir_val split …")
    val_items = load_split(Path(splits_dir), "ir_val")

    train_ds = build_datasets(train_items, code_embs, tokenizer, args.inp_max_len)
    val_ds = build_datasets(val_items, code_embs, tokenizer, args.inp_max_len)

    _col = partial(collate_fn, tokenizer=tokenizer)

    train_concat = ConcatDataset(train_ds)
    train_sampler = HierarchicalRepoSampler(train_concat, args.n_tasks_per_batch, args.n_points_per_task)
    train_loader = DataLoader(train_concat, batch_sampler=train_sampler, collate_fn=_col)

    val_loader = None
    if val_ds:
        val_concat = ConcatDataset(val_ds)
        val_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(val_concat), batch_size=args.val_batch_size, drop_last=False
        )
        val_loader = DataLoader(val_concat, batch_sampler=val_sampler, collate_fn=_col)

    return train_loader, {"val/seen": val_loader}


# ---------------------------------------------------------------------------
# Custom training loop (based on T2L sft_trainer.train)
# ---------------------------------------------------------------------------


def _validate(model, hypermod, val_loaders, _glb, curstep):
    with torch.no_grad(), evaluating(model, hypermod):
        out = {}
        for name, loader in val_loaders.items():
            if loader is None:
                continue
            info = defaultdict(list)
            for batch in loader:
                bl = _glb(batch, return_per_token_acc=True, return_entropy=True)
                info["sft_loss"].append(bl["sft_loss"].item())
                info["per_token_acc"].append(bl["per_token_acc"].item())
                info["entropy"].append(bl["entropy"].item())
            for k in info:
                info[k] = float(np.mean(info[k]))
                wandb.log({f"{name}/{k}": info[k]}, step=curstep)
            out[name] = info
    return out


def train_loop(
    args,
    save_dir,
    inp_dropout,
    accelerator,
    model,
    layer_indices,
    hypermod,
    train_loader,
    val_loaders,
    optimizer,
    num_training_steps,
    scheduler,
):
    model.train()
    hypermod.train()
    wandb.watch(hypermod, log="all", log_freq=1000)

    _glb = partial(
        get_loss_batch,
        model=model,
        target_modules=args.target_modules,
        inp_dropout=inp_dropout,
        layer_indices=layer_indices,
        use_hypernet=True,
        hypermod=hypermod,
        equally_weight_sample=args.equally_weight_sample,
    )
    _glb_train = partial(
        _glb,
        label_smoothing=args.label_smoothing,
        l2_reg_generated_w=args.l2_reg_generated_w,
    )

    neftune_handle = trl_activate_neftune(model, args.neftune_noise_alpha)

    # Initial validation
    _validate(model, hypermod, val_loaders, _glb, curstep=0)
    save_hypermod_checkpoint(save_dir, hypermod, curstep=0)

    curstep = 1
    avg = defaultdict(list)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        for batch in (pbar := tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            with accelerator.accumulate(model), accelerator.autocast():
                bl = _glb_train(batch)
                loss = bl["sft_loss"] + bl["generated_w_l2_loss"]
                avg["train/sft_loss"].append(bl["sft_loss"].item())
                avg["train/gen_w_l2"].append(bl["generated_w_l2_loss"].item())
                avg["train/total_loss"].append(loss.item())

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if curstep % args.logging_freq == 0 or curstep == num_training_steps:
                wandb.log(
                    {
                        "train/total_loss": np.mean(avg["train/total_loss"]),
                        "train/sft_loss": np.mean(avg["train/sft_loss"]),
                        "train/gen_w_l2": np.mean(avg["train/gen_w_l2"]),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": curstep / len(train_loader),
                        "train/step": curstep,
                    },
                    step=curstep,
                )
                logger.info(
                    f"step {curstep} | loss {np.mean(avg['train/total_loss']):.4f} "
                    f"| sft {np.mean(avg['train/sft_loss']):.4f}"
                )
                avg = defaultdict(list)

            if curstep % args.val_freq == 0 or curstep == num_training_steps:
                vi = _validate(model, hypermod, val_loaders, _glb, curstep)
                cp = save_hypermod_checkpoint(save_dir, hypermod, curstep)
                if "val/seen" in vi and vi["val/seen"]["sft_loss"] < best_val_loss:
                    best_val_loss = vi["val/seen"]["sft_loss"]
                    shutil.copy(cp, f"{save_dir}/hypermod.pt")
                    logger.info(f"  New best val loss: {best_val_loss:.4f}")

            curstep += 1

    # Final checkpoint
    last_cp = save_hypermod_checkpoint(save_dir, hypermod, curstep)
    if not os.path.isfile(f"{save_dir}/hypermod.pt"):
        shutil.copy(last_cp, f"{save_dir}/hypermod.pt")

    if args.keep_only_best:
        cp_dirs = sorted(
            [d for d in Path(f"{save_dir}/checkpoints").iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        for d in cp_dirs[:-1]:
            shutil.rmtree(d)

    wandb.unwatch(hypermod)
    neftune_handle.remove()
    accelerator.end_training()
    model.eval()
    hypermod.eval()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    args.use_hypernet = True
    args.training_task = "sft"
    if args.train_ds_names is None:
        args.train_ds_names = []
    if args.n_train_ds is None:
        args.n_train_ds = 0

    # Code embeddings
    logger.info(f"Loading code embeddings: {args.code_emb_path}")
    code_embs = torch.load(args.code_emb_path, map_location="cpu", weights_only=True)
    task_emb_size = next(iter(code_embs.values())).shape[-1]
    logger.info(f"  dim={task_emb_size}, repos={len(code_embs)}")

    save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)

    # Accelerator
    plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_steps, sync_with_dataloader=False)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=plugin, split_batches=True)

    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        config=vars(args),
        group=args.run_name,
        name=args.run_name,
        dir=wandb_dir,
        notes=args.notes,
    )
    device = accelerator.device

    # Model (cd into text2lora/ so chat template paths resolve)
    orig_cwd = os.getcwd()
    t2l_dir = os.path.join(os.path.dirname(__file__), "..", "..", "text2lora")
    os.chdir(t2l_dir)

    from peft import LoraConfig as _LoraConfig

    peft_config = _LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=list(args.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )
    peft_config.save_pretrained(os.path.join(orig_cwd, save_dir))

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=True,
        requires_grad=False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )

    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)

    hypermod = create_hypermod(args, "lora", device, model, layer_indices, task_emb_size)
    model.add_module("hypermod", hypermod)

    os.chdir(orig_cwd)

    n_total, n_train = get_num_params(model)
    logger.info(f"Trainable params: {n_train:,d} / {n_total:,d}")

    # Data
    splits_dir = os.environ.get(
        "SPLITS_DIR",
        os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "REPO_DATASET"),
    )
    train_loader, val_loaders = build_loaders(args, code_embs, tokenizer, splits_dir)
    logger.info(f"Train batches/epoch: {len(train_loader)}")

    # Optimizer (only hypernetwork params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model, hypermod, optimizer = accelerator.prepare(model, hypermod, optimizer)
    train_loader = accelerator.prepare(train_loader)
    for k, v in val_loaders.items():
        if v is not None:
            val_loaders[k] = accelerator.prepare(v)

    num_steps = args.epochs * len(train_loader)
    warmup = int(args.warmup_frac * num_steps / args.grad_accum_steps)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=int(num_steps / args.grad_accum_steps),
    )
    scheduler = accelerator.prepare(scheduler)
    inp_dropout = peft_config.lora_dropout

    train_loop(
        args,
        save_dir,
        inp_dropout,
        accelerator,
        model,
        layer_indices,
        hypermod,
        train_loader,
        val_loaders,
        optimizer,
        num_steps,
        scheduler,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "repopeft_text2lora_code_sft"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert args.exp_setup == "hyper_lora"

    if not hasattr(args, "code_emb_path") or not args.code_emb_path:
        print("ERROR: config must include 'code_emb_path'")
        sys.exit(1)

    uid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    args.run_name = "code_sft_" + time.strftime("%Y%m%d-%H%M%S") + f"_{uid}"
    args.save_dir = f"text2lora/train_outputs/sft/hyper_lora/{args.run_name}"

    global logger
    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(sys.argv)}")
    logger.debug(f"args: {args}")

    main(args)
