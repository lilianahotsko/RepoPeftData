#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for FFT and Single LoRA baselines.

Tunes: learning_rate, num_epochs, warmup_ratio, weight_decay, grad_accum_steps.
Uses eval_loss on cr_val as the objective.

Usage:
    python baselines/hpo_optuna.py --method fft --use-oracle \
        --oracle-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V3 \
        --max-oracle-tokens 6000 --max-seq-length 8192 \
        --n-trials 20
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import optuna
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


def load_data(args, tokenizer):
    """Load train and val datasets."""
    from evaluation.oracle_utils import (
        load_oracle_cache, lookup_oracle_context,
        augment_prefix_with_oracle, augment_prefix_with_compressed_oracle,
    )
    from baselines.finetuned.train_finetuned import load_all_training_pairs

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    oracle_cache_dir = None
    if args.use_oracle:
        oracle_cache_dir = Path(args.oracle_cache_dir).expanduser().resolve()

    train_pairs = load_all_training_pairs(
        splits_dir, oracle_cache_dir=oracle_cache_dir,
        max_oracle_tokens=args.max_oracle_tokens, tokenizer=tokenizer,
    )

    # Load val
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

    return Dataset.from_list(train_pairs), Dataset.from_list(val_pairs) if val_pairs else None


def create_objective(args, train_ds, val_ds, tokenizer):
    """Create Optuna objective function."""

    def objective(trial):
        # Hyperparameters to tune
        lr = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
        num_epochs = trial.suggest_int("num_epochs", 1, 5)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        grad_accum = trial.suggest_categorical("grad_accum_steps", [8, 16, 32])
        lr_scheduler = trial.suggest_categorical("lr_scheduler", ["cosine", "linear", "constant_with_warmup"])

        if args.method == "lora":
            lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32, 64])
            lora_alpha = lora_rank * 2
            lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)

        output_dir = os.path.join(args.output_dir, f"trial_{trial.number}")

        # Load model fresh each trial
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": "cuda:0"},
        )

        if args.method == "lora":
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
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  LoRA r={lora_rank}, trainable={trainable:,d}")

        data_collator = PrefixTargetDataCollator(
            tokenizer=tokenizer,
            target_marker=TARGET_MARKER,
            max_length=args.max_seq_length,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
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
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=1,
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            formatting_func=formatting_func,
            data_collator=data_collator,
            args=training_args,
        )

        trainer.train()

        # Get best eval loss
        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]

        print(f"\n  Trial {trial.number}: eval_loss={eval_loss:.4f} "
              f"(lr={lr:.2e}, epochs={num_epochs}, warmup={warmup_ratio:.2f}, "
              f"wd={weight_decay:.3f}, ga={grad_accum}, sched={lr_scheduler})")

        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

        # Remove trial checkpoints to save space
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        return eval_loss

    return objective


def main():
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )

    ap = argparse.ArgumentParser(description="Optuna HPO for FFT / Single LoRA")
    ap.add_argument("--method", required=True, choices=["fft", "lora"],
                    help="Training method")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Directory for trial outputs and study DB")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--val-split", type=str, default="cr_val")
    ap.add_argument("--use-oracle", action="store_true")
    ap.add_argument("--oracle-cache-dir", type=str, default=None)
    ap.add_argument("--max-oracle-tokens", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--study-name", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Oracle: {args.use_oracle}, max_tokens={args.max_oracle_tokens}")

    # Load tokenizer and data once
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    print("Loading data...")
    train_ds, val_ds = load_data(args, tokenizer)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}")

    # Create Optuna study
    study_name = args.study_name or f"hpo_{args.method}_drc_v3_8k"
    db_path = os.path.join(args.output_dir, f"{study_name}.db")
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    objective = create_objective(args, train_ds, val_ds, tokenizer)

    print(f"\nStarting {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)

    # Report results
    print("\n" + "=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best eval_loss: {study.best_trial.value:.4f}")
    print(f"Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save best params
    results = {
        "method": args.method,
        "best_trial": study.best_trial.number,
        "best_eval_loss": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    results_path = os.path.join(args.output_dir, f"best_params_{args.method}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
