#!/usr/bin/env python3
"""
Prepare the oracle LoRA directory structure expected by Text2LoRA's
reconstruction training pipeline.

Text2LoRA's get_target_lora_dirs() scans:
    train_outputs/sft/oracle_lora/*/adapter_model.safetensors
    train_outputs/sft/oracle_lora/*/args.yaml   (must contain train_ds_names + model_dir)

This script:
  1. Reads per-repo LoRA adapters from $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA/
  2. Creates symlinks inside text2lora/train_outputs/sft/oracle_lora/{repo_slug}/
  3. Writes a minimal args.yaml alongside each symlink
  4. Generates the final training config YAML (text2lora/configs/repopeft_text.yaml)
     with the full train_ds_names list

Usage:
    python baselines/text2lora/prepare_oracle_loras.py \
        --lora-root   $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA \
        --text2lora-dir text2lora \
        --model-dir   Qwen/Qwen2.5-Coder-1.5B \
        --splits-dir  $SCRATCH/REPO_DATASET
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def find_adapter(lora_root: Path, repo_name: str) -> Path | None:
    """Find adapter_model.safetensors for a given owner/repo."""
    owner, repo = repo_name.split("/", 1)
    candidate = lora_root / owner / repo / "adapter" / "adapter_model.safetensors"
    if candidate.exists():
        return candidate
    # Fallback: flat slug layout
    candidate2 = lora_root / slug(repo_name) / "adapter_model.safetensors"
    if candidate2.exists():
        return candidate2
    return None


def find_adapter_config(lora_root: Path, repo_name: str) -> Path | None:
    owner, repo = repo_name.split("/", 1)
    candidate = lora_root / owner / repo / "adapter" / "adapter_config.json"
    if candidate.exists():
        return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-root",      required=True, type=Path)
    ap.add_argument("--text2lora-dir",  required=True, type=Path)
    ap.add_argument("--model-dir",      default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--splits-dir",     required=True, type=Path,
                    help="Used to determine which repos are in the training split")
    args = ap.parse_args()

    # ── Training repos only (we only reconstruct train-split LoRAs) ───────────
    train_file = args.splits_dir / "train.json"
    train_repos: list[str] = sorted(
        json.loads(train_file.read_text()).get("repositories", {}).keys()
    )
    print(f"Training repos: {len(train_repos)}")

    oracle_dir = args.text2lora_dir / "train_outputs" / "sft" / "oracle_lora"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    found: list[str] = []
    missing: list[str] = []

    for repo_name in train_repos:
        adapter_src = find_adapter(args.lora_root, repo_name)
        if adapter_src is None:
            missing.append(repo_name)
            continue

        repo_slug = slug(repo_name)
        target_dir = oracle_dir / repo_slug
        target_dir.mkdir(exist_ok=True)

        # Symlink adapter_model.safetensors
        link = target_dir / "adapter_model.safetensors"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(adapter_src.resolve())

        # Symlink adapter_config.json (needed by Text2LoRA to read peft_config)
        cfg_src = find_adapter_config(args.lora_root, repo_name)
        if cfg_src:
            cfg_link = target_dir / "adapter_config.json"
            if cfg_link.exists() or cfg_link.is_symlink():
                cfg_link.unlink()
            cfg_link.symlink_to(cfg_src.resolve())

        # Write args.yaml (tells Text2LoRA which dataset this LoRA belongs to)
        args_yaml = {
            "train_ds_names": [repo_slug],
            "model_dir": args.model_dir,
        }
        (target_dir / "args.yaml").write_text(yaml.dump(args_yaml))

        found.append(repo_slug)

    print(f"  Linked: {len(found)}")
    print(f"  Missing LoRA: {len(missing)}")
    if missing:
        print("  Missing repos:")
        for r in missing[:10]:
            print(f"    {r}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    # ── Generate training config YAML ─────────────────────────────────────────
    # Read target_modules from one of the adapter configs
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    sample_cfg = find_adapter_config(args.lora_root, train_repos[0])
    if sample_cfg:
        import json as _json
        cfg_data = _json.loads(sample_cfg.read_text())
        target_modules = cfg_data.get("target_modules", target_modules)

    config = {
        # ── Experiment setup ──────────────────────────────────────────────────
        "exp_setup": "hyper_lora",
        "training_task": "recon",
        "model_dir": args.model_dir,
        # Explicit text embedding model (gte-large for text descriptions)
        "emb_model": "Alibaba-NLP/gte-large-en-v1.5",

        # ── LoRA target configuration ─────────────────────────────────────────
        "target_modules": target_modules,

        # ── Dataset ───────────────────────────────────────────────────────────
        "train_ds_names": found,      # all repos that have oracle LoRAs
        "n_train_ds":     len(found),
        "n_descs_per_ds": 5,          # one per description variant
        "n_embs_per_sampled_task": 1,
        "use_per_task_emb": True,
        "use_one_hot_task_emb": False,

        # ── Reconstruction training ────────────────────────────────────────────
        "factorized":   True,   # predict A and B matrices separately
        "pred_z_score": True,   # z-score normalise targets (stabler training)

        # ── Hypernetwork architecture ─────────────────────────────────────────
        "hypernet_latent_size": 256,
        "head_in_size":         512,
        "head_use_bias":        False,
        "encoder_type":         "linear",

        # ── Optimiser ─────────────────────────────────────────────────────────
        "lr":             1e-4,
        "weight_decay":   1e-3,
        "max_grad_norm":  1.0,
        "warmup_frac":    0.1,
        "grad_accum_steps": 1,

        # ── Training schedule ─────────────────────────────────────────────────
        "epochs":          500,   # reconstruction converges faster than SFT
        "n_tasks_per_batch": 8,
        "batch_size":      8,     # unused in recon but required by parser

        # ── Logging / saving ──────────────────────────────────────────────────
        "logging_freq": 50,
        "val_freq":     10000,
        "save_freq":    10000,
        "keep_only_best": True,
        "skip_eval":    True,   # no vllm eval needed during recon training
        "seed":         42,
    }

    cfg_path = args.text2lora_dir / "configs" / "repopeft_text.yaml"
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(yaml.dump(config, sort_keys=False, allow_unicode=True))

    print(f"\nConfig written to: {cfg_path}")
    print(f"  Repos in train_ds_names: {len(found)}")
    print(f"  Target modules: {target_modules}")
    print(f"\nNext step:")
    print(f"  cd {args.text2lora_dir}")
    print(f"  python scripts/train_hyper_recon.py configs/repopeft_text.yaml")


if __name__ == "__main__":
    main()
