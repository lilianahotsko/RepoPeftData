#!/usr/bin/env python3
"""
Training script for Code2LoRA-GRU: a hypernetwork that generates
repository-specific LoRA adapters by processing source files sequentially
through a GRU, trained over GitHub commit history.

Training loop per batch element:
  1. Initialize h_0 via Mamba2/MeanPool on the repo preamble (or zeros).
  2. Feed files in order through the GRU, one step per file, until all files
     present at the training commit have been processed.
  3. Use the final hidden state h_t to generate a LoRA via the LoRA generator.
  4. Apply the LoRA to the frozen LLM.
  5. Compute cross-entropy loss on the assertion target given the prefix.
  6. Backprop through the hypernetwork and GRU (truncated BPTT).

Usage:
    python hypernetwork/train_code2lora_gru.py --splits-dir $SCRATCH/REPO_DATASET
    python hypernetwork/train_code2lora_gru.py --init-type mamba2 --bptt-window 32
    python hypernetwork/train_code2lora_gru.py --init-type zeros --gru-hidden-dim 1024
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code2lora_gru import Code2LoRAGRU

_DEBUG_PRINT_EVERY = 100
_DEBUG_FIRST_N = 3
_DEBUG_STEP = {"global": 0}


def _should_debug_print():
    step = _DEBUG_STEP["global"]
    return step < _DEBUG_FIRST_N or step % _DEBUG_PRINT_EVERY == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading: GRU-aware splits (with commit history)
# ---------------------------------------------------------------------------

def load_gru_splits(
    splits_dir: Path,
    limit_train_repos: Optional[int] = None,
    limit_eval_repos: Optional[int] = None,
    limit_test_repos: Optional[int] = None,
    oracle_cache_dir: Optional[Path] = None,
    file_order: str = "chronological",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load GRU-format split JSONs (gru_train.json etc.), falling back to
    standard splits if GRU versions are unavailable.

    Each returned item has:
        repo, prefix, target, file_embeddings (ordered list), preamble_embeddings,
        repo_embedding, file_lengths, preamble_lengths
    """
    if oracle_cache_dir:
        from evaluation.oracle_utils import (
            augment_prefix_with_oracle,
            load_oracle_cache,
            lookup_oracle_context,
        )

    def _resolve_file_sequence(
        rdata: Dict,
        file_order: str,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Build ordered file embedding sequences and preamble from repo data."""
        file_embs_raw = rdata.get("file_embeddings", [])
        if not file_embs_raw:
            return [], []

        emb_by_path = {}
        for fe in file_embs_raw:
            emb_by_path[fe["path"]] = fe["embedding"]

        history = rdata.get("commit_history")

        if history and history.get("file_order"):
            ordered_paths = [fo["path"] for fo in history["file_order"]]
        else:
            ordered_paths = [fe["path"] for fe in file_embs_raw]

        if file_order == "reverse":
            ordered_paths = list(reversed(ordered_paths))
        elif file_order == "random":
            ordered_paths = list(ordered_paths)
            random.shuffle(ordered_paths)
        elif file_order == "alphabetical":
            ordered_paths = sorted(ordered_paths)

        all_file_embs = []
        for p in ordered_paths:
            if p in emb_by_path:
                all_file_embs.append(emb_by_path[p])

        preamble_embs = []
        if history:
            preamble_files = set(history.get("preamble_files", []))
            for p in ordered_paths:
                if p in preamble_files and p in emb_by_path:
                    preamble_embs.append(emb_by_path[p])

        return all_file_embs, preamble_embs

    def load_split(
        path: Path,
        limit: Optional[int],
    ) -> List[Dict]:
        gru_path = path.parent / f"gru_{path.name}"
        actual_path = gru_path if gru_path.exists() else path
        if not actual_path.exists():
            return []

        data = json.loads(actual_path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})
        repo_names = sorted(repos.keys())
        if limit is not None:
            repo_names = repo_names[:limit]

        items = []
        n_with_files = 0
        n_augmented = 0

        for repo in repo_names:
            r = repos[repo]
            pairs = r.get("qna_pairs", [])
            emb = r.get("embedding")
            if emb is None:
                continue

            oracle_contexts = (
                load_oracle_cache(oracle_cache_dir, repo) if oracle_cache_dir else {}
            )
            file_embs, preamble_embs = _resolve_file_sequence(r, file_order)
            has_files = len(file_embs) > 0
            if has_files:
                n_with_files += 1

            for p in pairs:
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                if not prefix or not target:
                    continue
                if target.lstrip().startswith(","):
                    continue

                if oracle_contexts:
                    oracle_code = lookup_oracle_context(
                        oracle_contexts, p.get("metadata", {})
                    )
                    if oracle_code:
                        prefix = augment_prefix_with_oracle(prefix, oracle_code)
                        n_augmented += 1

                items.append({
                    "repo": repo,
                    "prefix": prefix,
                    "target": target,
                    "repo_embedding": emb,
                    "file_embeddings": file_embs if has_files else None,
                    "preamble_embeddings": preamble_embs if preamble_embs else None,
                })

        if oracle_cache_dir and items:
            print(
                f"  Oracle: augmented {n_augmented}/{len(items)} "
                f"({100*n_augmented/len(items):.1f}%) in {actual_path.name}"
            )
        print(
            f"  {actual_path.name}: {len(items)} items from "
            f"{len(repo_names)} repos ({n_with_files} with file embeddings)"
        )
        return items

    train = load_split(splits_dir / "train.json", limit_train_repos)
    val = load_split(splits_dir / "cr_val.json", limit_eval_repos)
    test = load_split(splits_dir / "cr_test.json", limit_test_repos)
    return train, val, test


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def get_bos_id(tok):
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return tok.pad_token_id


def prepare_tokens_and_labels(prefix, target, tok, add_eos=True):
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    target_ids = tok.encode(target, add_special_tokens=False)
    if add_eos and tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]
    bos = get_bos_id(tok)
    tokens = [bos] + prefix_ids + target_ids
    labels = [-100] * (1 + len(prefix_ids)) + target_ids
    return {"tokens": tokens, "labels": labels}


def left_truncate_left_pad(tokens, labels, max_len, pad_token_id):
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]
        labels = labels[-max_len:]
    if len(tokens) < max_len:
        pad_len = max_len - len(tokens)
        tokens = [pad_token_id] * pad_len + tokens
        labels = [-100] * pad_len + labels
    return tokens, labels


def prepare_training_items(raw_items, tok):
    out = []
    for it in raw_items:
        tl = prepare_tokens_and_labels(it["prefix"], it["target"], tok)
        item = {
            "repo": it["repo"],
            "tokens": tl["tokens"],
            "labels": tl["labels"],
            "repo_embedding": it["repo_embedding"],
            "file_embeddings": it.get("file_embeddings"),
            "preamble_embeddings": it.get("preamble_embeddings"),
        }
        out.append(item)
    return out


def to_hf_dataset(items, seed, shuffle):
    ds = Dataset.from_list(items)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.add_column("text", [""] * len(ds))
    return ds


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

@dataclass
class GRUDataCollator:
    """Collator for Code2LoRA-GRU training.

    Handles file embedding sequences of variable length by padding, and
    falls back to the repo-level embedding when file embeddings are absent.
    """

    pad_token_id: int
    max_seq_len: int = 8192
    max_files: int = 512

    def __call__(self, examples):
        ex = examples[0]

        tokens, labels = left_truncate_left_pad(
            ex["tokens"], ex["labels"], self.max_seq_len, self.pad_token_id,
        )
        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]

        file_embs = ex.get("file_embeddings")
        preamble_embs = ex.get("preamble_embeddings")

        batch = {
            "repo_name": ex.get("repo", ""),
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }

        if file_embs and len(file_embs) > 0:
            file_embs = file_embs[: self.max_files]
            batch["file_embeddings"] = torch.tensor(
                [file_embs], dtype=torch.float32
            )
            batch["file_lengths"] = torch.tensor(
                [len(file_embs)], dtype=torch.long
            )
        else:
            repo_emb = ex["repo_embedding"]
            batch["file_embeddings"] = torch.tensor(
                [[repo_emb]], dtype=torch.float32
            )  # [1, 1, D] -- treat repo embedding as single "file"
            batch["file_lengths"] = torch.tensor([1], dtype=torch.long)

        if preamble_embs and len(preamble_embs) > 0:
            batch["preamble_embeddings"] = torch.tensor(
                [preamble_embs], dtype=torch.float32
            )
            batch["preamble_lengths"] = torch.tensor(
                [len(preamble_embs)], dtype=torch.long
            )

        return batch


# ---------------------------------------------------------------------------
# Hook-based LoRA injection (from hypernetwork_paw.py)
# ---------------------------------------------------------------------------

def discover_target_modules(model, target_module_names):
    target_modules_dict = {}
    module_dims = {}
    max_layer = -1
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for target in target_module_names:
            if not name.endswith(f".{target}"):
                continue
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    target_modules_dict[(layer_idx, target)] = module
                    if target not in module_dims:
                        module_dims[target] = (
                            module.in_features,
                            module.out_features,
                        )
                    max_layer = max(max_layer, layer_idx)
                    break
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    return target_modules_dict, module_dims, num_layers


def apply_lora_hooks(target_modules_dict, lora_params, scaling):
    handles = []
    for (layer_idx, module_name), module in target_modules_dict.items():
        key = (layer_idx, module_name)
        if key not in lora_params:
            continue
        A, B = lora_params[key]

        def _make_hook(_A, _B, _scaling):
            def hook(mod, inp, out):
                x = inp[0]
                _Ac = _A.to(dtype=x.dtype)
                _Bc = _B.to(dtype=x.dtype)
                delta = torch.bmm(
                    torch.bmm(x, _Ac.transpose(1, 2)),
                    _Bc.transpose(1, 2),
                ) * _scaling
                return out + delta
            return hook

        h = module.register_forward_hook(_make_hook(A, B, scaling))
        handles.append(h)
    return handles


def remove_lora_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Code2LoRAGRUTrainer(SFTTrainer):
    """Custom trainer that runs the Code2LoRA-GRU forward pass:
    preamble -> h_0 -> GRU(files) -> LoRA generator -> inject -> LM loss.
    """

    def __init__(
        self,
        *args,
        code2lora_gru: Code2LoRAGRU,
        target_modules_dict: Dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.code2lora_gru = code2lora_gru
        self._target_modules_dict = target_modules_dict
        self._active_hooks = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device

        file_embs = inputs["file_embeddings"].to(device=device, dtype=torch.float32)
        file_lengths = inputs["file_lengths"].to(device=device)

        preamble_embs = inputs.get("preamble_embeddings")
        preamble_lengths = inputs.get("preamble_lengths")
        if preamble_embs is not None:
            preamble_embs = preamble_embs.to(device=device, dtype=torch.float32)
            preamble_lengths = preamble_lengths.to(device=device)

        h_final, lora_params = self.code2lora_gru(
            file_embeddings=file_embs,
            file_lengths=file_lengths,
            preamble_embeddings=preamble_embs,
            preamble_lengths=preamble_lengths,
        )

        self._active_hooks = apply_lora_hooks(
            self._target_modules_dict,
            lora_params,
            self.code2lora_gru.lora_generator.lora_scaling,
        )

        out = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            labels=inputs["labels"].to(device),
        )
        loss = out["loss"] if isinstance(out, dict) else out[0]

        if _should_debug_print():
            n_files = file_lengths[0].item()
            has_preamble = preamble_embs is not None
            print(
                f"[step={_DEBUG_STEP['global']}] loss={loss.item():.4f} "
                f"repo={inputs.get('repo_name', '?')!r} "
                f"files={n_files} preamble={has_preamble}",
                flush=True,
            )

        return (loss, out) if return_outputs else loss

    def _remove_active_hooks(self):
        hooks = getattr(self, "_active_hooks", None)
        if hooks:
            remove_lora_hooks(hooks)
            self._active_hooks = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        do_dbg = _should_debug_print()
        model.train()
        self.code2lora_gru.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        self._remove_active_hooks()

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.code2lora_gru.parameters(), self.args.max_grad_norm
            )
            if do_dbg:
                print(
                    f"[step={_DEBUG_STEP['global']}] grad_norm={total_norm:.4f}",
                    flush=True,
                )

        _DEBUG_STEP["global"] += 1
        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        result = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        self._remove_active_hooks()
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.code2lora_gru.training
        self.code2lora_gru.eval()
        result = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if was_training:
            self.code2lora_gru.train()
        return result

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if _internal_call:
            return


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class SaveGRUCallback(TrainerCallback):
    def __init__(self, code2lora_gru, target_modules_dict, filename="code2lora_gru_state.pt"):
        self.code2lora_gru = code2lora_gru
        self.target_modules_dict = target_modules_dict
        self.filename = filename

    def _payload(self):
        return {
            "model_state_dict": self.code2lora_gru.state_dict(),
            "model_config": self.code2lora_gru.config_dict(),
            "target_module_keys": list(self.target_modules_dict.keys()),
        }

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved Code2LoRA-GRU -> {path}")

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved final Code2LoRA-GRU -> {path}")


class SaveBestGRUCallback(TrainerCallback):
    def __init__(self, code2lora_gru, target_modules_dict, filename="code2lora_gru_best.pt"):
        self.code2lora_gru = code2lora_gru
        self.target_modules_dict = target_modules_dict
        self.filename = filename
        self.best_eval_loss = float("inf")

    def _payload(self):
        return {
            "model_state_dict": self.code2lora_gru.state_dict(),
            "model_config": self.code2lora_gru.config_dict(),
            "target_module_keys": list(self.target_modules_dict.keys()),
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, self.filename)
            torch.save(self._payload(), path)
            print(f"Saved best Code2LoRA-GRU (eval_loss={eval_loss:.4f}) -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Code2LoRA-GRU")

    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--limit-train-repos", type=int, default=None)
    ap.add_argument("--limit-eval-repos", type=int, default=None)
    ap.add_argument("--limit-test-repos", type=int, default=None)
    ap.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B"
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU/full_repos",
    )

    # GRU architecture
    ap.add_argument("--gru-hidden-dim", type=int, default=1024)
    ap.add_argument("--gru-num-layers", type=int, default=1)
    ap.add_argument("--gru-dropout", type=float, default=0.0)
    ap.add_argument("--bptt-window", type=int, default=32)
    ap.add_argument(
        "--init-type",
        type=str,
        default="mamba2",
        choices=["mamba2", "meanpool", "zeros"],
    )
    ap.add_argument(
        "--file-order",
        type=str,
        default="chronological",
        choices=["chronological", "reverse", "random", "alphabetical"],
    )
    ap.add_argument("--max-files", type=int, default=512)

    # LoRA generator
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora-hidden-dim", type=int, default=512)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--trunk-depth", type=int, default=2)

    # Training
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--use-oracle", action="store_true")
    ap.add_argument("--oracle-cache-dir", type=str, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)

    target_modules_default = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ]
    ap.add_argument("--target-modules", nargs="+", default=target_modules_default)

    args = ap.parse_args()
    set_seed(args.seed)

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    oracle_cache_dir = None
    if args.use_oracle:
        from evaluation.oracle_utils import get_default_oracle_cache_dir

        oracle_cache_dir = Path(
            args.oracle_cache_dir or get_default_oracle_cache_dir()
        ).expanduser().resolve()
        if not oracle_cache_dir.exists():
            raise FileNotFoundError(f"Oracle cache not found: {oracle_cache_dir}")
        print(f"Using oracle context from {oracle_cache_dir}")

    # ── Load data ──
    print("Loading data...")
    train_items, eval_items, test_items = load_gru_splits(
        splits_dir=splits_dir,
        limit_train_repos=args.limit_train_repos,
        limit_eval_repos=args.limit_eval_repos,
        limit_test_repos=args.limit_test_repos,
        oracle_cache_dir=oracle_cache_dir,
        file_order=args.file_order,
    )

    print("=" * 80)
    print("[CONFIG]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 80, flush=True)

    wandb.init(
        project="code2lora-gru-REPOPEFTDATA",
        name=args.output_dir.split("/")[-1],
    )

    # ── Load tokenizer and base model ──
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading frozen base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    total_base_params = sum(p.numel() for p in model.parameters())
    print(
        f"Base model: {total_base_params:,} params (frozen), "
        f"dtype={next(model.parameters()).dtype}"
    )

    # ── Discover target modules ──
    target_modules_dict, module_dims, num_layers = discover_target_modules(
        model, args.target_modules
    )
    print(
        f"Discovered {len(target_modules_dict)} target modules "
        f"across {num_layers} layers:"
    )
    for m, (d_in, d_out) in module_dims.items():
        print(f"  {m}: in={d_in}, out={d_out}")

    # ── Tokenize ──
    train_items = prepare_training_items(train_items, tok)
    eval_items = prepare_training_items(eval_items, tok)
    test_items = prepare_training_items(test_items, tok)
    print(
        f"After tokenization: train={len(train_items)} "
        f"eval={len(eval_items)} test={len(test_items)}"
    )

    if not train_items:
        raise ValueError("No training items. Check --splits-dir.")

    # ── Determine file embedding dimensionality ──
    file_embed_dim = None
    for it in train_items:
        if it.get("file_embeddings") and len(it["file_embeddings"]) > 0:
            file_embed_dim = len(it["file_embeddings"][0])
            break
    if file_embed_dim is None:
        file_embed_dim = len(train_items[0]["repo_embedding"])
        print(
            f"WARNING: No file embeddings found, using repo embedding dim={file_embed_dim}. "
            f"Run create_dataset/extract_commit_history.py first for full GRU training."
        )
    print(f"File embedding dim: {file_embed_dim}")

    # ── Stats ──
    for split_name, items in [
        ("train", train_items),
        ("eval", eval_items),
        ("test", test_items),
    ]:
        if items:
            lengths = [len(it["tokens"]) for it in items]
            repos = set(it["repo"] for it in items)
            n_with_files = sum(
                1 for it in items
                if it.get("file_embeddings") and len(it["file_embeddings"]) > 1
            )
            print(
                f"  {split_name}: n={len(items)} repos={len(repos)} "
                f"seq_len: min={min(lengths)} max={max(lengths)} "
                f"mean={np.mean(lengths):.0f} "
                f"with_file_embs={n_with_files}/{len(items)}"
            )

    # ── Build HF datasets ──
    train_ds = to_hf_dataset(train_items, seed=args.seed, shuffle=True)
    eval_ds = to_hf_dataset(eval_items, seed=args.seed, shuffle=False)

    collator = GRUDataCollator(
        pad_token_id=tok.pad_token_id,
        max_seq_len=args.max_seq_len,
        max_files=args.max_files,
    )

    # ── Build Code2LoRA-GRU ──
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
    ).to(device="cuda:0", dtype=torch.float32)

    n_gru_params = sum(p.numel() for p in code2lora_gru.parameters())
    print(f"\nCode2LoRA-GRU: {n_gru_params:,} params ({n_gru_params/1e6:.2f}M)")

    gru_params = sum(p.numel() for p in code2lora_gru.gru.parameters())
    init_params = (
        sum(p.numel() for p in code2lora_gru.initializer.parameters())
        if code2lora_gru.initializer is not None
        else 0
    )
    gen_params = (
        sum(p.numel() for p in code2lora_gru.lora_generator.parameters())
        if code2lora_gru.lora_generator is not None
        else 0
    )
    print(f"  GRU: {gru_params:,}  Initializer: {init_params:,}  Generator: {gen_params:,}")

    # ── SFT config ──
    sft_cfg = SFTConfig(
        dataset_text_field="text",
        label_names=["labels"],
        remove_unused_columns=False,
        max_length=args.max_seq_len,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_safetensors=True,
        seed=args.seed,
        bf16=True,
        output_dir=args.output_dir,
        report_to="wandb",
        max_grad_norm=5.0,
        prediction_loss_only=True,
    )

    opt = torch.optim.AdamW(
        code2lora_gru.parameters(), lr=sft_cfg.learning_rate, weight_decay=0.01
    )

    total_steps = len(train_items) * args.epochs // args.grad_accum
    print(f"\nOptimizer: AdamW lr={sft_cfg.learning_rate} wd=0.01")
    print(
        f"Training: epochs={args.epochs} grad_accum={args.grad_accum} "
        f"~{total_steps} steps"
    )

    save_cb = SaveGRUCallback(code2lora_gru, target_modules_dict)
    save_best_cb = SaveBestGRUCallback(code2lora_gru, target_modules_dict)

    trainer = Code2LoRAGRUTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_cfg,
        optimizers=(opt, None),
        code2lora_gru=code2lora_gru,
        target_modules_dict=target_modules_dict,
        callbacks=[save_cb, save_best_cb],
    )

    # ── Train ──
    print("\nInitial eval...")
    model.eval()
    init_val = trainer.evaluate()
    print(f"init_val_loss = {init_val['eval_loss']:.4f}")
    wandb.log({"init_val_loss": init_val["eval_loss"]})

    _DEBUG_STEP["global"] = 0
    print("\nTraining...")
    trainer.train()
    print(f"Training complete. Steps executed: {_DEBUG_STEP['global']}")

    print("\nFinal eval...")
    final_val = trainer.evaluate()
    print(f"final_val_loss = {final_val['eval_loss']:.4f}")
    wandb.log({"final_val_loss": final_val["eval_loss"]})

    if test_items:
        test_ds = to_hf_dataset(test_items, seed=args.seed, shuffle=False)
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        test_loss = test_metrics.get("test_eval_loss")
        if test_loss is not None:
            print(f"test_loss = {test_loss:.4f}")
            wandb.log({"test_loss": test_loss})

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
