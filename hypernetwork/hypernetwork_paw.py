#!/usr/bin/env python3
"""
PAW-style hypernetwork: shared-basis LoRA generation with per-layer weights.

Key differences from hypernetwork_sampled.py:
  - Shared basis: learnable A_bases/B_bases per module type, combined via mixing coefficients
  - Per-layer LoRA: each transformer layer gets distinct (A, B) weights
  - Hook-based injection: forward hooks instead of module replacement
  - Residual MLP trunk: deeper processing with LayerNorm + residual connections
  - ~8.5x fewer parameters than direct-projection hypernetwork

Usage:
    python hypernetwork/hypernetwork_paw.py --splits-dir $SCRATCH/REPO_DATASET
    python hypernetwork/hypernetwork_paw.py --use-oracle --max-seq-len 8192
"""

import os
import sys
import json
import re
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_METADATA = "REPO_METADATA.json"
QNA_HYPERNET = "QNA_HYPERNET.json"

_DEBUG_PRINT_EVERY = 100
_DEBUG_FIRST_N = 3
_DEBUG_STEP = {"global": 0}


def _should_debug_print():
    step = _DEBUG_STEP["global"]
    return step < _DEBUG_FIRST_N or step % _DEBUG_PRINT_EVERY == 0


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading (same as hypernetwork_sampled.py)
# ---------------------------------------------------------------------------

def load_from_splits(
    splits_dir: Path,
    limit_train_repos: Optional[int],
    limit_eval_repos: Optional[int],
    limit_test_repos: Optional[int],
    oracle_cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load from train.json, cr_val.json, cr_test.json."""
    if oracle_cache_dir:
        from evaluation.oracle_utils import load_oracle_cache, lookup_oracle_context, augment_prefix_with_oracle

    def load_split(path: Path, limit: Optional[int]) -> List[Dict]:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})
        repo_names = sorted(repos.keys())
        if limit is not None:
            repo_names = repo_names[:limit]
        items = []
        n_augmented = 0
        for repo in repo_names:
            r = repos[repo]
            pairs = r.get("qna_pairs", [])
            emb = r.get("embedding")
            if emb is None:
                continue
            oracle_contexts = load_oracle_cache(oracle_cache_dir, repo) if oracle_cache_dir else {}
            for p in pairs:
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
                items.append({
                    "repo": repo,
                    "repo_name": repo,
                    "prefix": prefix,
                    "target": target,
                    "repo_embedding": emb,
                })
        if oracle_cache_dir and items:
            print(f"  Oracle: augmented {n_augmented}/{len(items)} ({100*n_augmented/len(items):.1f}%) in {path.name}")
        return items

    train_items = load_split(splits_dir / "train.json", limit_train_repos)
    eval_items = load_split(splits_dir / "cr_val.json", limit_eval_repos)
    test_items = load_split(splits_dir / "cr_test.json", limit_test_repos)
    return train_items, eval_items, test_items


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
    return {
        "tokens": tokens,
        "labels": labels,
        "prefix_len": len(prefix_ids),
        "target_len": len(target_ids),
    }


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
        repo = it.get("repo", "")
        prefix = it.get("prefix", "")
        target = it.get("target", "")
        if not repo or not prefix or not target:
            continue
        if it.get("repo_embedding") is None:
            continue
        tl = prepare_tokens_and_labels(prefix, target, tok, add_eos=True)
        out.append({
            "repo": repo,
            "repo_name": repo,
            "repo_embedding": it.get("repo_embedding"),
            "tokens": tl["tokens"],
            "labels": tl["labels"],
            "prefix_len": tl["prefix_len"],
            "target_len": tl["target_len"],
        })
    return out


def to_hf_dataset(items, seed, shuffle):
    ds = Dataset.from_list(items)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.add_column("text", [""] * len(ds))
    return ds


@dataclass
class HypernetDataCollator:
    pad_token_id: int
    max_seq_len: int = 8192

    def __call__(self, examples):
        ex = examples[0]
        ctx = torch.tensor(ex["repo_embedding"], dtype=torch.float32).unsqueeze(0)
        tokens, labels = left_truncate_left_pad(
            tokens=ex["tokens"],
            labels=ex["labels"],
            max_len=self.max_seq_len,
            pad_token_id=self.pad_token_id,
        )
        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]
        return {
            "repo_name": ex.get("repo_name", ""),
            "ctx": ctx,
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# PAW-style architecture: shared-basis LoRA generation
# ---------------------------------------------------------------------------

class _ResidualMLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> residual add."""
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


def discover_target_modules(
    model: nn.Module,
    target_module_names: list[str],
) -> Tuple[Dict, Dict, int]:
    """
    Walk model and find target Linear layers.

    Returns:
        target_modules_dict:  {(layer_idx, module_name): nn.Linear}
        module_dims:          {module_name: (in_features, out_features)}
        num_layers:           number of transformer layers
    """
    target_modules_dict: Dict = {}
    module_dims: Dict = {}
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
                        module_dims[target] = (module.in_features, module.out_features)
                    max_layer = max(max_layer, layer_idx)
                    break
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    return target_modules_dict, module_dims, num_layers


def apply_lora_hooks(
    target_modules_dict: Dict,
    lora_params: Dict,
    scaling: float,
) -> list:
    """Register forward hooks that add LoRA delta to each target module's output."""
    handles = []
    for (layer_idx, module_name), module in target_modules_dict.items():
        key = (layer_idx, module_name)
        if key not in lora_params:
            continue
        A, B = lora_params[key]

        def _make_hook(_A, _B, _scaling):
            def hook(mod, inp, out):
                x = inp[0]
                _A_cast = _A.to(dtype=x.dtype)
                _B_cast = _B.to(dtype=x.dtype)
                delta = torch.bmm(
                    torch.bmm(x, _A_cast.transpose(1, 2)),
                    _B_cast.transpose(1, 2),
                ) * _scaling
                return out + delta
            return hook

        h = module.register_forward_hook(_make_hook(A, B, scaling))
        handles.append(h)
    return handles


def remove_lora_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


class LoraMapper(nn.Module):
    """
    Shared-basis LoRA generation: learnable basis vectors combined via
    input-conditioned mixing coefficients.

    For each module type (e.g. q_proj), maintains:
      - A_bases: (num_bases, rank, d_in)  -- normal init
      - B_bases: (num_bases, d_out, rank) -- zero init (LoRA convention)

    A coefficient head maps the trunk output to per-layer, per-module mixing
    weights that linearly combine bases into final per-layer LoRA (A, B).
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        module_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int = 512,
        rank: int = 16,
        alpha: float = 32.0,
        num_bases: int = 16,
        trunk_depth: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.lora_scaling = alpha / rank
        self.module_names = sorted(module_dims.keys())
        self.module_dims = module_dims
        self.num_bases = num_bases

        layers: list = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, trunk_depth - 1)):
            layers.append(_ResidualMLP(hidden_dim))
        self.trunk = nn.Sequential(*layers)

        for m in self.module_names:
            d_in, d_out = module_dims[m]
            self.register_parameter(
                f"A_bases_{m}",
                nn.Parameter(torch.randn(num_bases, rank, d_in) * 0.02),
            )
            self.register_parameter(
                f"B_bases_{m}",
                nn.Parameter(torch.zeros(num_bases, d_out, rank)),
            )

        M = len(self.module_names)
        num_coeff = num_layers * M * num_bases * 2
        self.coeff_head = nn.Linear(hidden_dim, num_coeff)
        nn.init.normal_(self.coeff_head.weight, std=0.01)
        nn.init.zeros_(self.coeff_head.bias)

    def forward(self, ctx: torch.Tensor) -> Dict:
        """
        Args:
            ctx: (B, input_dim) repo embedding, already L2-normalized.
        Returns:
            {(layer_idx, module_name): (A, B)} where A: (B, rank, d_in), B: (B, d_out, rank)
        """
        L = self.num_layers
        M = len(self.module_names)
        N = self.num_bases

        h = self.trunk(ctx.float())
        coeffs = self.coeff_head(h)
        BK = coeffs.shape[0]
        coeffs = coeffs.view(BK, L, M, N, 2)

        lora_params: Dict = {}
        for mi, m in enumerate(self.module_names):
            A_bases = getattr(self, f"A_bases_{m}")
            B_bases = getattr(self, f"B_bases_{m}")
            for layer_idx in range(L):
                alpha_A = coeffs[:, layer_idx, mi, :, 0]  # (BK, N)
                alpha_B = coeffs[:, layer_idx, mi, :, 1]  # (BK, N)
                A = torch.einsum("bn,nrd->brd", alpha_A, A_bases)
                B = torch.einsum("bn,ndr->bdr", alpha_B, B_bases)
                lora_params[(layer_idx, m)] = (A, B)

        return lora_params

    def config_dict(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "lora_scaling": self.lora_scaling,
            "module_names": self.module_names,
            "module_dims": self.module_dims,
            "num_bases": self.num_bases,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HypernetPawTrainer(SFTTrainer):
    def __init__(self, *args, lora_mapper, target_modules_dict, mapper_fp32, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_mapper = lora_mapper
        self._target_modules_dict = target_modules_dict
        self._mapper_fp32 = mapper_fp32

    def _run_mapper(self, ctx):
        if self._mapper_fp32:
            lora_params = self.lora_mapper(ctx.float())
            model_dtype = next(self.model.parameters()).dtype
            return {k: (A.to(model_dtype), B.to(model_dtype)) for k, (A, B) in lora_params.items()}
        return self.lora_mapper(ctx)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ctx = inputs["ctx"].to(device=model.device, dtype=torch.float32)
        ctx = F.normalize(ctx, p=2, dim=-1)
        lora_params = self._run_mapper(ctx)

        # Hooks must survive through backward for gradient-checkpointing
        # recomputation. Store on self so training_step can clean up after
        # backward() completes.
        self._active_hooks = apply_lora_hooks(
            self._target_modules_dict, lora_params, self.lora_mapper.lora_scaling,
        )

        out = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            labels=inputs["labels"].to(model.device),
        )
        loss = out["loss"] if isinstance(out, dict) else out[0]

        if _should_debug_print():
            print(f"[step={_DEBUG_STEP['global']}] loss={loss.item():.4f} repo={inputs.get('repo_name','?')!r}", flush=True)

        return (loss, out) if return_outputs else loss

    def _remove_active_hooks(self):
        hooks = getattr(self, "_active_hooks", None)
        if hooks:
            remove_lora_hooks(hooks)
            self._active_hooks = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        do_dbg = _should_debug_print()
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        # Safe to remove hooks now — backward (including checkpoint recomputation) is done
        self._remove_active_hooks()

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.lora_mapper.parameters(), self.args.max_grad_norm)
            if do_dbg:
                print(f"[step={_DEBUG_STEP['global']}] grad_norm={total_norm:.4f}", flush=True)

        _DEBUG_STEP["global"] += 1
        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to clean up hooks after each eval forward (no backward)."""
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        self._remove_active_hooks()
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.lora_mapper.training
        self.lora_mapper.eval()
        result = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                  metric_key_prefix=metric_key_prefix)
        if was_training:
            self.lora_mapper.train()
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

class SaveMapperCallback(TrainerCallback):
    def __init__(self, lora_mapper: LoraMapper, target_modules_dict, filename="lora_mapper_state.pt"):
        self.lora_mapper = lora_mapper
        self.target_modules_dict = target_modules_dict
        self.filename = filename

    def _payload(self):
        return {
            "lora_mapper_state_dict": self.lora_mapper.state_dict(),
            "lora_mapper_config": self.lora_mapper.config_dict(),
            "target_module_keys": list(self.target_modules_dict.keys()),
        }

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved lora_mapper -> {path}")

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved final lora_mapper -> {path}")


class SaveBestMapperCallback(TrainerCallback):
    def __init__(self, lora_mapper: LoraMapper, target_modules_dict, filename="lora_mapper_best.pt"):
        self.lora_mapper = lora_mapper
        self.target_modules_dict = target_modules_dict
        self.filename = filename
        self.best_eval_loss = float("inf")

    def _payload(self):
        return {
            "lora_mapper_state_dict": self.lora_mapper.state_dict(),
            "lora_mapper_config": self.lora_mapper.config_dict(),
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
            print(f"Saved best lora_mapper (eval_loss={eval_loss:.4f}) -> {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="PAW-style hypernetwork with shared-basis LoRA")

    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--limit-train-repos", type=int, default=None)
    ap.add_argument("--limit-eval-repos", type=int, default=None)
    ap.add_argument("--limit-test-repos", type=int, default=None)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--output-dir", type=str,
                    default="/scratch/lhotsko/TRAINING_CHECKPOINTS/HYPERNET_PAW/full_repos")

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-bases", type=int, default=16,
                    help="Number of LoRA basis vectors per module type")
    ap.add_argument("--trunk-depth", type=int, default=2,
                    help="1=shallow, 2+=adds residual MLP blocks")
    ap.add_argument("--lora-mapper-fp32", action="store_true",
                    help="Keep LoRA mapper in fp32 for numerical stability")

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

    target_modules_default = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    ap.add_argument("--target-modules", nargs="+", default=target_modules_default)

    args = ap.parse_args()
    set_seed(args.seed)

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    oracle_cache_dir = None
    if args.use_oracle:
        from evaluation.oracle_utils import get_default_oracle_cache_dir
        oracle_cache_dir = Path(args.oracle_cache_dir or get_default_oracle_cache_dir()).expanduser().resolve()
        if not oracle_cache_dir.exists():
            raise FileNotFoundError(f"Oracle cache not found: {oracle_cache_dir}")
        print(f"Using oracle context from {oracle_cache_dir}")

    train_items, eval_items, test_items = load_from_splits(
        splits_dir=splits_dir,
        limit_train_repos=args.limit_train_repos,
        limit_eval_repos=args.limit_eval_repos,
        limit_test_repos=args.limit_test_repos,
        oracle_cache_dir=oracle_cache_dir,
    )

    print("=" * 80)
    print("[CONFIG]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 80, flush=True)

    wandb.init(project="hypernet-paw-REPOPEFTDATA", name=args.output_dir.split("/")[-1])

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: {total_params:,} params (frozen), dtype={next(model.parameters()).dtype}")

    target_modules_dict, module_dims, num_layers = discover_target_modules(model, args.target_modules)
    print(f"Discovered {len(target_modules_dict)} target modules across {num_layers} layers:")
    for m, (d_in, d_out) in module_dims.items():
        print(f"  {m}: in={d_in}, out={d_out}")

    print(f"Loaded splits: train={len(train_items)} eval={len(eval_items)} test={len(test_items)}")

    train_items = prepare_training_items(train_items, tok)
    eval_items = prepare_training_items(eval_items, tok)
    test_items = prepare_training_items(test_items, tok)
    print(f"After tokenization: train={len(train_items)} eval={len(eval_items)} test={len(test_items)}")

    if not train_items:
        raise ValueError("No training items. Check --splits-dir.")

    embedding_dim = len(train_items[0]["repo_embedding"])
    print(f"Embedding dim: {embedding_dim}")

    import numpy as np
    for split_name, items in [("train", train_items), ("eval", eval_items), ("test", test_items)]:
        if items:
            lengths = [len(it["tokens"]) for it in items]
            repos = set(it["repo"] for it in items)
            print(f"  {split_name}: n={len(items)} repos={len(repos)} "
                  f"seq_len: min={min(lengths)} max={max(lengths)} "
                  f"mean={np.mean(lengths):.0f} >{args.max_seq_len}: {sum(1 for l in lengths if l > args.max_seq_len)}")

    train_ds = to_hf_dataset(train_items, seed=args.seed, shuffle=True)
    eval_ds = to_hf_dataset(eval_items, seed=args.seed, shuffle=False)

    collator = HypernetDataCollator(pad_token_id=tok.pad_token_id, max_seq_len=args.max_seq_len)

    mapper_dtype = torch.float32 if args.lora_mapper_fp32 else torch.bfloat16
    lora_mapper = LoraMapper(
        input_dim=embedding_dim,
        num_layers=num_layers,
        module_dims=module_dims,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        alpha=float(args.alpha),
        num_bases=args.num_bases,
        trunk_depth=args.trunk_depth,
    ).to(device="cuda:0", dtype=mapper_dtype)

    n_mapper_params = sum(p.numel() for p in lora_mapper.parameters())
    print(f"\nLoraMapper: {n_mapper_params:,} params ({n_mapper_params/1e6:.2f}M), dtype={mapper_dtype}")
    basis_params = sum(
        getattr(lora_mapper, f"A_bases_{m}").numel() + getattr(lora_mapper, f"B_bases_{m}").numel()
        for m in lora_mapper.module_names
    )
    trunk_params = sum(p.numel() for p in lora_mapper.trunk.parameters())
    coeff_params = sum(p.numel() for p in lora_mapper.coeff_head.parameters())
    print(f"  Bases: {basis_params:,}  Trunk: {trunk_params:,}  Coeff head: {coeff_params:,}")
    lora_per_example = 0
    for m in lora_mapper.module_names:
        d_in, d_out = module_dims[m]
        lora_per_example += num_layers * (args.rank * d_in + d_out * args.rank)
    print(f"  Per-example LoRA program: {lora_per_example:,} params ({lora_per_example/1e3:.1f}K)")
    if args.lora_mapper_fp32:
        print(f"  [fp32 mode] Mapper runs in fp32, LoRA outputs cast to model dtype")

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

    opt = torch.optim.AdamW(lora_mapper.parameters(), lr=sft_cfg.learning_rate, weight_decay=0.01)

    total_steps = len(train_items) * args.epochs // args.grad_accum
    print(f"\nOptimizer: AdamW lr={sft_cfg.learning_rate} wd=0.01")
    print(f"Training: epochs={args.epochs} grad_accum={args.grad_accum} ~{total_steps} steps")

    save_cb = SaveMapperCallback(lora_mapper, target_modules_dict)
    save_best_cb = SaveBestMapperCallback(lora_mapper, target_modules_dict)

    trainer = HypernetPawTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_cfg,
        optimizers=(opt, None),
        lora_mapper=lora_mapper,
        target_modules_dict=target_modules_dict,
        mapper_fp32=args.lora_mapper_fp32,
        callbacks=[save_cb, save_best_cb],
    )

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
