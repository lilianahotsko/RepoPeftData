#!/usr/bin/env python3
"""
Text2LoRA reconstruction training with pre-computed CODE embeddings
instead of text description embeddings.

Differences from the standard text2lora train_hyper_recon.py:
  - No embedding model loaded (saves GPU memory)
  - task_embs_dict is loaded from a pre-computed .pt file
  - task_emb_size is read from the embeddings (2048 for Qwen3-Embedding)

Usage (from project root, NOT from text2lora/):
    python baselines/text2lora/train_code_conditioned.py \
        text2lora/configs/repopeft_code.yaml
"""

import gc
import os
import random
import string
import shutil
import sys
import time
from copy import deepcopy
from math import ceil

import torch
import wandb
from transformers import get_scheduler, set_seed
from peft import get_peft_config, PeftConfig, load_peft_weights

# Add text2lora/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "text2lora", "src"))

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import get_recon_train_data
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.recon_trainer import train
from hyper_llm_modulator.utils import (
    create_logger,
    get_layers,
    get_num_params,
    save_yaml,
    get_model_and_tokenizer,
    get_target_lora_dirs,
)


def main(args):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.training_task = "recon"

    # ── Load pre-computed code embeddings ─────────────────────────────────────
    code_emb_path = args.code_emb_path
    logger.info(f"Loading pre-computed code embeddings from {code_emb_path}")
    all_code_embs = torch.load(code_emb_path, map_location="cpu", weights_only=True)

    # Filter to training repos only
    task_embs_dict = {}
    missing_embs = []
    for ds_name in args.train_ds_names:
        if ds_name in all_code_embs:
            task_embs_dict[ds_name] = all_code_embs[ds_name].to("cuda")
        else:
            missing_embs.append(ds_name)

    if missing_embs:
        logger.warning(f"Missing code embeddings for {len(missing_embs)} repos, removing from training")
        for m in missing_embs:
            args.train_ds_names.remove(m)
        args.n_train_ds = len(args.train_ds_names)

    # Determine embedding dimension from the data
    sample_emb = next(iter(task_embs_dict.values()))
    task_emb_size = sample_emb.shape[-1]  # 2048 for Qwen3-Embedding
    logger.info(f"Code embedding dim: {task_emb_size}, repos: {len(task_embs_dict)}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Oracle LoRAs ──────────────────────────────────────────────────────────
    # Need to cd into text2lora/ for get_target_lora_dirs to find the symlinks
    orig_cwd = os.getcwd()
    text2lora_dir = os.path.join(os.path.dirname(__file__), "..", "..", "text2lora")
    os.chdir(text2lora_dir)

    oracle_lora_dir = getattr(args, "oracle_lora_dir", None)
    lora_dirs = get_target_lora_dirs(args.train_ds_names, args.model_dir, oracle_lora_dir=oracle_lora_dir)

    lora_dir = list(lora_dirs.values())[0]
    adapter_config_path = f"{lora_dir}/adapter_config.json"
    peft_config = get_peft_config(PeftConfig.from_json_file(adapter_config_path))
    shutil.copy(adapter_config_path, f"{orig_cwd}/{save_dir}/adapter_config.json")

    # ── Load base model (frozen, just for layer info) ─────────────────────────
    # Stay in text2lora/ dir so get_tokenizer finds chat_templates/
    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
    )

    n_layers = len(get_layers(model))
    layer_indices = list(range(n_layers))

    # ── Create hypernetwork ───────────────────────────────────────────────────
    # Stay in text2lora/ dir — create_hypermod internally calls get_target_lora_dirs
    hypermod = create_hypermod(
        args, peft_config.peft_type.lower(), device, model, layer_indices, task_emb_size
    )
    hypermod.train()
    logger.info(f"Hypermod created with task_emb_size={task_emb_size}")

    # Free model memory — only needed for layer info
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ── Load target LoRA weights ──────────────────────────────────────────────
    # Still in text2lora/ dir
    target_loras = {task: load_peft_weights(d) for task, d in lora_dirs.items()}
    os.chdir(orig_cwd)

    if args.factorized and args.mt_lora_path:
        mt_lora = load_peft_weights(args.mt_lora_path)
        target_loras = {
            task: {k: v - mt_lora[k] for k, v in lora_sd.items()}
            for task, lora_sd in target_loras.items()
        }
        del mt_lora

    logger.info(f"# of target LoRAs: {len(target_loras)}")

    train_data = {}
    for task, state_dict in target_loras.items():
        train_data[task] = get_recon_train_data(state_dict, args.target_modules, layer_indices, device)

    wandb.watch(hypermod, log="all")

    logger.debug("Trainable hypernet parameters:")
    for name, p in hypermod.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")
    _, num_trainable_params = get_num_params(hypermod)
    logger.info(f"trainable params: {num_trainable_params:,d}")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(hypermod.parameters(), lr=args.lr, weight_decay=1e-3)
    tasks_per_batch = min(args.n_tasks_per_batch, len(args.train_ds_names))
    n_minibatches = ceil(len(args.train_ds_names) / tasks_per_batch)
    n_batches = n_minibatches * args.epochs

    num_warmup_steps = args.warmup_frac * n_batches
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=int(n_batches),
    )

    train(
        args,
        hypermod,
        train_data,
        task_embs_dict,
        layer_indices,
        n_batches,
        n_minibatches,
        tasks_per_batch,
        args.n_embs_per_sampled_task,
        optimizer,
        lr_scheduler,
        device,
        save_dir,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "repopeft_text2lora_code"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert args.exp_setup == "hyper_lora"

    # Extra arg: path to pre-computed code embeddings
    if not hasattr(args, "code_emb_path") or not args.code_emb_path:
        print("ERROR: config must include 'code_emb_path' pointing to code_embeddings.pt")
        sys.exit(1)

    uid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    args.run_name = "code_" + time.strftime("%Y%m%d-%H%M%S") + f"_{uid}"
    args.save_dir = f"text2lora/train_outputs/recon/{args.exp_setup}/{args.run_name}"

    global logger
    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(sys.argv)}")
    logger.debug(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main(args)
