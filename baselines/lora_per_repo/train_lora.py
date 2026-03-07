#!/usr/bin/env python3
"""
Per-repo LoRA trainer. Trains a LoRA adapter for a single repository.

Uses the same LoRA setup as baselines/lora/lora_trainer.py (model, config, collator).

Recommended: train on train.json (val from ir_val.json), then final test on ir_test.json:
  python train_lora.py --from-split train --limit-repos 1
  # Adapter saved to $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA/author/repo_name/adapter
  python baselines/lora_per_repo/test_lora.py --adapter $SCRATCH/.../adapter --limit-repos 1   # ir_test (final)
  python baselines/lora_per_repo/test_lora.py --adapter $SCRATCH/.../adapter --split ir_val --limit-repos 1  # validation

Other usage:
  python train_lora.py --repo /path/to/repo
  python train_lora.py --qna-path /path/to/repo/QNA_HYPERNET.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


TARGET_MARKER = "### Target:"


class PrefixTargetDataCollator(DataCollatorForLanguageModeling):
    """
    Masks labels for the prefix; only computes loss on the target part (after marker).
    """

    def __init__(self, tokenizer, target_marker: str = TARGET_MARKER, max_length: int = 2048, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.target_marker = target_marker
        self.max_length = max_length
        self.marker_tokens = tokenizer.encode(target_marker, add_special_tokens=False)
        self.marker_with_newline = tokenizer.encode(target_marker + "\n", add_special_tokens=False)

    def __call__(self, examples):
        if isinstance(examples[0], dict):
            formatted_texts = []
            for ex in examples:
                text = ex["prefix"] + "\n" + self.target_marker + "\n" + ex["target"]
                formatted_texts.append(text)
            examples = formatted_texts

        if isinstance(examples[0], str):
            # Left truncation: drop prefix tokens, keep target (right side)
            # Left padding: pad on left so last token is real content (needed for causal LM)
            # (truncation_side/padding_side set on tokenizer before collator creation)
            batch = self.tokenizer(
                examples,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        else:
            batch = super().__call__(examples)

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        for i in range(len(labels)):
            token_ids = input_ids[i].tolist()
            idx = self._find_target_start(token_ids)
            if idx > 0:
                labels[i, :idx] = -100
        return batch

    def _find_target_start(self, token_ids: List[int]) -> int:
        for marker_tokens in [self.marker_tokens, self.marker_with_newline]:
            if not marker_tokens:
                continue
            for i in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[i : i + len(marker_tokens)] == marker_tokens:
                    return i + len(marker_tokens)
        return 0

# --- LoRA setup (matches baselines/lora/lora_trainer.py) ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
QNA_HYPERNET = "QNA_HYPERNET.json"


def load_qna_dataset(qna_path: str) -> list[dict]:
    """Load QNA pairs from QNA_HYPERNET.json. Returns list of {prefix, target, ...}."""
    with open(qna_path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    if not pairs:
        raise ValueError(f"No pairs found in {qna_path}")
    return pairs


def load_from_split(splits_dir: Path, split_name: str, limit_repos: int = 1) -> tuple[list[dict], str]:
    """Load qna_pairs from split JSON. Returns (pairs, repo_full) e.g. repo_full='author/repo_name'."""
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if not repo_names:
        raise ValueError(f"No repositories in {path}")
    repo_names = repo_names[:limit_repos]
    pairs = []
    for repo in repo_names:
        r = repos[repo]
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if prefix and target:
                pairs.append({"prefix": prefix, "target": target})
    if not pairs:
        raise ValueError(f"No qna_pairs in first {limit_repos} repo(s) of {path}")
    repo_full = repo_names[0] if repo_names else "unknown"
    return pairs, repo_full


def load_val_for_repos(splits_dir: Path, val_split: str, repo_names: list[str]) -> list[dict]:
    """Load qna_pairs from val_split (e.g. ir_val.json) for the given repos. Returns flat list of pairs."""
    path = splits_dir / f"{val_split}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    pairs = []
    for repo in repo_names:
        r = repos.get(repo, {})
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if prefix and target:
                pairs.append({"prefix": prefix, "target": target})
    return pairs


def qna_to_examples(pairs: list[dict]) -> list[dict]:
    """Convert QNA pairs to {prefix, target} format."""
    return [{"prefix": p.get("prefix", ""), "target": p.get("target", "")} for p in pairs]


def create_train_val_split(examples: list[dict], val_ratio: float = 0.15, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Split examples into train and validation."""
    import random
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    n_train = len(shuffled) - n_val
    train_examples = shuffled[:n_train]
    val_examples = shuffled[n_train:]
    return Dataset.from_list(train_examples), Dataset.from_list(val_examples)


def formatting_func(example: dict) -> str:
    """Format as prefix + marker + target."""
    return example["prefix"] + "\n" + TARGET_MARKER + "\n" + example["target"]


def main():
    import os
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    parser = argparse.ArgumentParser(description="Train per-repo LoRA adapter")
    parser.add_argument(
        "--repo",
        type=str,
        help="Path to repository directory (must contain QNA_HYPERNET.json)",
    )
    parser.add_argument(
        "--qna-path",
        type=str,
        help="Direct path to QNA_HYPERNET.json (alternative to --repo)",
    )
    parser.add_argument(
        "--from-split",
        type=str,
        help="Load from split JSON (e.g. cr_test) instead of repo/qna-path",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=default_dataset,
        help=f"Dir with cr_test.json etc. (default: {default_dataset})",
    )
    parser.add_argument(
        "--limit-repos",
        type=int,
        default=1,
        help="When using --from-split: use first N repos (default: 1)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="ir_val",
        help="When using --from-split: validation split (default: ir_val)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the LoRA adapter (default: $SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA/author/repo_name)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    default_output_base = os.path.join(scratch, "TRAINING_CHECKPOINTS", "PER_REPO_LORA")

    # Resolve data source
    if args.from_split:
        splits_dir = Path(args.splits_dir).expanduser().resolve()
        print(f"Loading train from {args.from_split}.json (first {args.limit_repos} repo(s))")
        train_pairs, repo_full = load_from_split(splits_dir, args.from_split, limit_repos=args.limit_repos)
        examples = qna_to_examples(train_pairs)
        # Validation from ir_val.json for the same repos
        path = splits_dir / f"{args.from_split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        repo_names = sorted(data.get("repositories", {}).keys())[: args.limit_repos]
        val_pairs = load_val_for_repos(splits_dir, args.val_split, repo_names)
        val_examples = qna_to_examples(val_pairs)
    elif args.qna_path:
        qna_path = Path(args.qna_path)
        if not qna_path.exists():
            raise FileNotFoundError(f"QNA file not found: {qna_path}")
        parts = qna_path.parent.parts
        repo_full = "/".join(parts[-2:]) if len(parts) >= 2 else qna_path.parent.name
        print(f"Loading QNA from {qna_path}")
        pairs = load_qna_dataset(str(qna_path))
        examples = qna_to_examples(pairs)
        val_examples = []
    elif args.repo:
        repo_path = Path(args.repo).resolve()
        qna_path = repo_path / QNA_HYPERNET
        if not qna_path.exists():
            raise FileNotFoundError(f"QNA_HYPERNET.json not found at {qna_path}")
        parts = repo_path.parts
        repo_full = "/".join(parts[-2:]) if len(parts) >= 2 else repo_path.name
        print(f"Loading QNA from {qna_path}")
        pairs = load_qna_dataset(str(qna_path))
        examples = qna_to_examples(pairs)
        val_examples = []
    else:
        parser.error("Provide --repo, --qna-path, or --from-split")

    repo_slug = repo_full.replace("/", "_")

    # Output dir: TRAINING_CHECKPOINTS/PER_REPO_LORA/author/repo_name
    output_dir = args.output_dir or os.path.join(default_output_base, repo_full)
    output_dir = str(Path(output_dir).resolve())

    print(f"Loaded {len(examples)} training examples")

    if args.from_split and val_examples:
        train_dataset = Dataset.from_list(examples)
        val_dataset = Dataset.from_list(val_examples)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} (from {args.val_split}.json)")
    else:
        train_dataset, val_dataset = create_train_val_split(examples, val_ratio=args.val_ratio)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model & tokenizer (same as lora_trainer.py)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Left truncation: keep target (right), drop prefix (left) when over max_length
    # Left padding: pad on left so last token is real content (causal LM)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator (masks prefix, loss only on target; uses left truncation/padding)
    data_collator = PrefixTargetDataCollator(
        tokenizer=tokenizer,
        target_marker=TARGET_MARKER,
        max_length=args.max_seq_length,
        mlm=False,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none" if args.no_wandb else "wandb",
        remove_unused_columns=False,
    )

    if not args.no_wandb:
        import wandb
        repo_config = str(qna_path.parent) if not args.from_split else f"{args.from_split}:{repo_slug}"
        wandb.init(
            project="lora-per-repo-REPOPEFTDATA",
            name=f"lora-{repo_slug}",
            config={
                "model_name": MODEL_NAME,
                "repo": repo_config,
                "num_examples": len(examples),
                **vars(args),
            },
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=training_args,
    )

    print("\nTraining...")
    trainer.train()

    # Save
    save_path = Path(output_dir) / "adapter"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nLoRA adapter saved to {save_path}")

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
