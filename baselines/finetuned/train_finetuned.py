#!/usr/bin/env python3
"""
Full fine-tuning baseline: SFT Qwen2.5-Coder-1.5B on ALL training QnA pairs.
No repo embedding, no LoRA. Standard supervised fine-tuning.

This baseline shows what you get from seeing the training data distribution
without repo-specific adaptation.

Usage:
    python baselines/finetuned/train_finetuned.py
    python baselines/finetuned/train_finetuned.py --epochs 3 --lr 2e-5
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TARGET_MARKER = "### Target:"


class PrefixTargetDataCollator(DataCollatorForLanguageModeling):
    """Masks prefix labels; only compute loss on the target part."""

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
            batch = self.tokenizer(
                examples, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
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

    def _find_target_start(self, token_ids: list[int]) -> int:
        for marker_tokens in [self.marker_tokens, self.marker_with_newline]:
            if not marker_tokens:
                continue
            for i in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[i: i + len(marker_tokens)] == marker_tokens:
                    return i + len(marker_tokens)
        return 0


def load_all_training_pairs(
    splits_dir: Path,
    oracle_cache_dir: Path | None = None,
) -> list[dict]:
    """Load all QnA pairs from train.json (all repos).
    If *oracle_cache_dir* is given, prepend oracle context to each prefix."""
    if oracle_cache_dir:
        from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context, augment_prefix_with_oracle

    path = splits_dir / "train.json"
    if not path.exists():
        raise FileNotFoundError(f"train.json not found at {splits_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    pairs = []
    n_augmented = 0
    for repo_name, r in repos.items():
        oracle_contexts = load_oracle_cache(oracle_cache_dir, repo_name) if oracle_cache_dir else {}
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue
            if oracle_contexts:
                oracle_code = lookup_oracle_context(oracle_contexts, p.get("metadata", {}))
                if oracle_code:
                    prefix = augment_prefix_with_oracle(prefix, oracle_code)
                    n_augmented += 1
            pairs.append({"prefix": prefix, "target": target, "repo": repo_name})
    if oracle_cache_dir:
        print(f"  Oracle context: augmented {n_augmented}/{len(pairs)} pairs ({100*n_augmented/max(1,len(pairs)):.1f}%)")
    return pairs


def formatting_func(example: dict) -> str:
    return example["prefix"] + "\n" + TARGET_MARKER + "\n" + example["target"]


def main():
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_output = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "TRAINING_CHECKPOINTS", "FINETUNED",
    )

    ap = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder on all training QnA pairs")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--output-dir", type=str, default=default_output)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--val-split", type=str, default="cr_val")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Prepend oracle context to prefixes (increases effective seq length)")
    ap.add_argument("--oracle-cache-dir", type=str, default=None,
                    help="Dir with pre-built oracle contexts (default: $SCRATCH/ORACLE_CONTEXT_CACHE)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    oracle_cache_dir = None
    if args.use_oracle:
        from evaluation.oracle_utils import get_default_oracle_cache_dir, load_oracle_cache, lookup_oracle_context, augment_prefix_with_oracle
        oracle_cache_dir = Path(args.oracle_cache_dir or get_default_oracle_cache_dir()).expanduser().resolve()
        if not oracle_cache_dir.exists():
            raise FileNotFoundError(f"Oracle cache not found: {oracle_cache_dir}\n"
                                    "Run: python baselines/oracle_context/build_context.py")
        print(f"Using oracle context from {oracle_cache_dir}")

    print("Loading training data...")
    train_pairs = load_all_training_pairs(splits_dir, oracle_cache_dir=oracle_cache_dir)
    print(f"Loaded {len(train_pairs)} training pairs")

    # Load validation
    val_path = splits_dir / f"{args.val_split}.json"
    val_pairs = []
    if val_path.exists():
        val_data = json.loads(val_path.read_text(encoding="utf-8"))
        for repo_name, r in val_data.get("repositories", {}).items():
            oracle_contexts = load_oracle_cache(oracle_cache_dir, repo_name) if oracle_cache_dir else {}
            for p in r.get("qna_pairs", []):
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                if not prefix or not target or target.lstrip().startswith(","):
                    continue
                if oracle_contexts:
                    oracle_code = lookup_oracle_context(oracle_contexts, p.get("metadata", {}))
                    if oracle_code:
                        prefix = augment_prefix_with_oracle(prefix, oracle_code)
                val_pairs.append({"prefix": prefix, "target": target, "repo": repo_name})
    print(f"Loaded {len(val_pairs)} validation pairs")

    train_ds = Dataset.from_list(train_pairs)
    val_ds = Dataset.from_list(val_pairs) if val_pairs else None

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )

    data_collator = PrefixTargetDataCollator(
        tokenizer=tokenizer,
        target_marker=TARGET_MARKER,
        max_length=args.max_seq_length,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        seed=args.seed,
        max_grad_norm=1.0,
        report_to="none" if args.no_wandb else "wandb",
        remove_unused_columns=False,
    )

    if not args.no_wandb:
        import wandb
        wandb.init(project="finetuned-baseline-REPOPEFTDATA", config=vars(args))

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=training_args,
    )

    print("\nTraining...")
    trainer.train()

    save_path = Path(args.output_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nModel saved to {save_path}")

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
