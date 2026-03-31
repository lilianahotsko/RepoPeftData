#!/usr/bin/env python3
"""
Train FFT or Single LoRA using best hyperparameters from Optuna HPO.

Reads best_params_{method}.json from the HPO output directory and runs
a full training with those parameters.

Usage:
    python baselines/train_with_best_params.py --method fft \
        --hpo-dir $SCRATCH/TRAINING_CHECKPOINTS/HPO_FFT_DRC_V3 \
        --output-dir $SCRATCH/TRAINING_CHECKPOINTS/FFT_DRC_V3_8K \
        --use-oracle --oracle-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V3 \
        --max-oracle-tokens 6000 --max-seq-length 8192
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_MARKER = "### Target:"


class PrefixTargetDataCollator(DataCollatorForLanguageModeling):
    """Masks prefix labels; only compute loss on the target part."""

    def __init__(self, tokenizer, target_marker=TARGET_MARKER, max_length=2048, **kwargs):
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

    def _find_target_start(self, token_ids):
        for marker_tokens in [self.marker_tokens, self.marker_with_newline]:
            if not marker_tokens:
                continue
            for i in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[i: i + len(marker_tokens)] == marker_tokens:
                    return i + len(marker_tokens)
        return 0


def formatting_func(example):
    return example["prefix"] + "\n" + TARGET_MARKER + "\n" + example["target"]


def main():
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fft", "lora"])
    ap.add_argument("--hpo-dir", required=True, type=str,
                    help="Dir containing best_params_{method}.json from HPO")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--val-split", type=str, default="cr_val")
    ap.add_argument("--use-oracle", action="store_true")
    ap.add_argument("--oracle-cache-dir", type=str, default=None)
    ap.add_argument("--max-oracle-tokens", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load best params
    params_path = os.path.join(args.hpo_dir, f"best_params_{args.method}.json")
    if not os.path.exists(params_path):
        print(f"ERROR: {params_path} not found. Run HPO first.")
        sys.exit(1)

    hpo_results = json.load(open(params_path))
    best = hpo_results["best_params"]
    print(f"Best HPO params (eval_loss={hpo_results['best_eval_loss']:.4f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    lr = best["learning_rate"]
    num_epochs = best["num_epochs"]
    warmup_ratio = best["warmup_ratio"]
    weight_decay = best["weight_decay"]
    grad_accum = best["grad_accum_steps"]
    lr_scheduler = best["lr_scheduler"]

    # Load data
    from evaluation.oracle_utils import (
        load_oracle_cache, lookup_oracle_context,
        augment_prefix_with_oracle, augment_prefix_with_compressed_oracle,
    )
    from baselines.finetuned.train_finetuned import load_all_training_pairs

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    oracle_cache_dir = Path(args.oracle_cache_dir).expanduser().resolve() if args.use_oracle else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    print("Loading training data...")
    train_pairs = load_all_training_pairs(
        splits_dir, oracle_cache_dir=oracle_cache_dir,
        max_oracle_tokens=args.max_oracle_tokens, tokenizer=tokenizer,
    )
    print(f"Train: {len(train_pairs)}")

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
                        if args.max_oracle_tokens:
                            prefix = augment_prefix_with_compressed_oracle(
                                prefix, oracle_code, tokenizer, args.max_oracle_tokens,
                            )
                        else:
                            prefix = augment_prefix_with_oracle(prefix, oracle_code)
                val_pairs.append({"prefix": prefix, "target": target, "repo": repo_name})
    print(f"Val: {len(val_pairs)}")

    train_ds = Dataset.from_list(train_pairs)
    val_ds = Dataset.from_list(val_pairs) if val_pairs else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )

    if args.method == "lora":
        lora_rank = best.get("lora_rank", 16)
        lora_alpha = lora_rank * 2
        lora_dropout = best.get("lora_dropout", 0.0)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = PrefixTargetDataCollator(
        tokenizer=tokenizer,
        target_marker=TARGET_MARKER,
        max_length=args.max_seq_length,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        eval_accumulation_steps=1,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
        remove_unused_columns=False,
    )

    if not args.no_wandb:
        import wandb
        project = f"{'finetuned' if args.method == 'fft' else 'single-lora'}-baseline-REPOPEFTDATA"
        wandb.init(project=project, config={**best, "method": args.method, "max_seq_length": args.max_seq_length})

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=training_args,
    )

    print(f"\nTraining with best HPO params: lr={lr:.2e}, epochs={num_epochs}, "
          f"warmup={warmup_ratio:.2f}, wd={weight_decay:.3f}, "
          f"ga={grad_accum}, sched={lr_scheduler}")
    trainer.train()

    save_path = Path(args.output_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nModel saved to {save_path}")

    # Save the params used
    (Path(args.output_dir) / "training_params.json").write_text(
        json.dumps({"hpo_source": params_path, **best}, indent=2)
    )

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
