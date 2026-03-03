#!/usr/bin/env python3
"""
Composable hypernetwork: generates LoRAs from individual file embeddings
and composes them into a repo-level adapter.

Composition strategies:
  - additive: average of per-file LoRAs
  - weighted: learned or similarity-based weighting of per-file LoRAs
  - gated: small gating network selects/weights file LoRAs based on the query

Training: same as hypernetwork_sampled.py but with file-level embeddings.
The hypernetwork processes each file embedding independently, then composes.
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypernetwork.hypernetwork_sampled import (
    LoRA, get_module_specs, replace_with_lora, inject_lora_weights,
    get_bos_id, prepare_tokens_and_labels, left_truncate_left_pad,
    set_seed, SaveHypernetCallback, SaveBestHypernetCallback,
)


class ComposableHypernetwork(nn.Module):
    """
    Hypernetwork that generates LoRA weights from individual file embeddings
    and supports multiple composition strategies.
    """

    def __init__(self, input_dim, module_specs, hidden_dim, rank,
                 composition="weighted", max_files=50):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.composition = composition
        self.max_files = max_files

        # Per-file encoder (shared across files)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        type_shapes = {}
        for _, _, t, in_f, out_f in module_specs:
            if t not in type_shapes:
                type_shapes[t] = (in_f, out_f)

        self.types = sorted(type_shapes.keys())
        self.type_shapes = type_shapes

        self.heads_A = nn.ModuleDict({
            t: nn.Linear(hidden_dim, rank * type_shapes[t][0])
            for t in self.types
        })
        self.heads_B = nn.ModuleDict({
            t: nn.Linear(hidden_dim, type_shapes[t][1] * rank)
            for t in self.types
        })

        self.log_scale_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(-3.5)) for t in self.types
        })
        self.log_scale_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(-3.5)) for t in self.types
        })

        # Composition modules
        if composition == "weighted":
            # Learnable weight predictor from file embedding
            self.weight_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif composition == "gated":
            # Gate conditioned on both file embedding and query context
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim + hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.query_proj = nn.Linear(input_dim, hidden_dim)

    def forward_single(self, ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate LoRA weights from a single file embedding. ctx: [B, dim]"""
        h = self.trunk(ctx.float())
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)

        A, B = {}, {}
        for t in self.types:
            in_f, out_f = self.type_shapes[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)

            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)

            A[t] = torch.tanh(A_raw) * scale_A
            B[t] = torch.tanh(B_raw) * scale_B

        return {"A": A, "B": B}

    def compose_loras(self, file_loras: List[Dict], weights: Optional[torch.Tensor] = None) -> Dict:
        """
        Compose multiple per-file LoRAs into a single adapter.
        file_loras: list of {"A": {t: tensor}, "B": {t: tensor}}
        weights: [N] tensor of composition weights (sum to 1)
        """
        if not file_loras:
            raise ValueError("No file LoRAs to compose")

        if len(file_loras) == 1:
            return file_loras[0]

        n = len(file_loras)
        if weights is None:
            weights = torch.ones(n, device=file_loras[0]["A"][self.types[0]].device) / n
        else:
            weights = weights / weights.sum().clamp(min=1e-8)

        composed_A, composed_B = {}, {}
        for t in self.types:
            stacked_A = torch.stack([fl["A"][t] for fl in file_loras], dim=0)  # [N, B, r, in_f]
            stacked_B = torch.stack([fl["B"][t] for fl in file_loras], dim=0)  # [N, B, out_f, r]
            w = weights.view(-1, 1, 1, 1)
            composed_A[t] = (stacked_A * w).sum(dim=0)
            composed_B[t] = (stacked_B * w).sum(dim=0)

        return {"A": composed_A, "B": composed_B}

    def forward(self, ctx: torch.Tensor, file_embeddings: Optional[torch.Tensor] = None,
                query_ctx: Optional[torch.Tensor] = None) -> Dict:
        """
        ctx: [B, dim] repo-level embedding (fallback)
        file_embeddings: [B, K, dim] per-file embeddings (optional)
        query_ctx: [B, dim] query/prefix embedding for gated composition (optional)
        """
        if file_embeddings is None or file_embeddings.shape[1] == 0:
            return self.forward_single(ctx)

        K = min(file_embeddings.shape[1], self.max_files)
        file_embeddings = file_embeddings[:, :K, :]

        # Generate per-file LoRAs
        B_size = file_embeddings.shape[0]
        file_loras = []
        for k in range(K):
            file_emb = F.normalize(file_embeddings[:, k, :], p=2, dim=-1)
            lora = self.forward_single(file_emb)
            file_loras.append(lora)

        # Compute composition weights
        if self.composition == "additive":
            weights = torch.ones(K, device=file_embeddings.device) / K
        elif self.composition == "weighted":
            raw_weights = []
            for k in range(K):
                w = self.weight_net(file_embeddings[:, k, :].float())  # [B, 1]
                raw_weights.append(w)
            weights = torch.softmax(torch.cat(raw_weights, dim=-1), dim=-1)  # [B, K]
            weights = weights.mean(dim=0)  # [K] average across batch
        elif self.composition == "gated":
            if query_ctx is None:
                query_ctx = ctx
            q = self.query_proj(query_ctx.float())  # [B, hidden_dim]
            raw_gates = []
            for k in range(K):
                combined = torch.cat([file_embeddings[:, k, :].float(), q], dim=-1)
                gate = self.gate_net(combined)  # [B, 1]
                raw_gates.append(gate)
            weights = torch.cat(raw_gates, dim=-1).mean(dim=0)  # [K]
            weights = weights / weights.sum().clamp(min=1e-8)
        else:
            weights = torch.ones(K, device=file_embeddings.device) / K

        return self.compose_loras(file_loras, weights)


@dataclass
class ComposableDataCollator:
    """Data collator that handles file-level embeddings."""
    pad_token_id: int
    max_seq_len: int = 8192
    max_files: int = 50

    def __call__(self, examples):
        ex = examples[0]
        ctx = torch.tensor(ex["repo_embedding"], dtype=torch.float32).unsqueeze(0)

        file_embs = ex.get("file_embeddings_tensor")
        if file_embs is not None:
            file_ctx = torch.tensor(file_embs, dtype=torch.float32).unsqueeze(0)  # [1, K, dim]
            if file_ctx.shape[1] > self.max_files:
                file_ctx = file_ctx[:, :self.max_files, :]
        else:
            file_ctx = None

        tokens, labels = left_truncate_left_pad(
            tokens=ex["tokens"],
            labels=ex["labels"],
            max_len=self.max_seq_len,
            pad_token_id=self.pad_token_id,
        )

        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]

        result = {
            "repo_name": ex.get("repo_name", ""),
            "ctx": ctx,
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }
        if file_ctx is not None:
            result["file_ctx"] = file_ctx

        return result


class ComposableHypernetTrainer(SFTTrainer):
    """Trainer for composable hypernetwork."""

    def __init__(self, *args, hypernet, module_specs, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypernet = hypernet
        self._module_specs = module_specs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ctx = inputs["ctx"].to(device=model.device, dtype=torch.float32)
        ctx = F.normalize(ctx, p=2, dim=-1)

        file_ctx = inputs.get("file_ctx")
        if file_ctx is not None:
            file_ctx = file_ctx.to(device=model.device, dtype=torch.float32)

        h = self.hypernet(ctx, file_embeddings=file_ctx)
        inject_lora_weights(model, self._module_specs, h, batch_index=0)

        out = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            labels=inputs["labels"].to(model.device),
        )
        loss = out["loss"] if isinstance(out, dict) else out[0]
        return (loss, out) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.hypernet.training
        self.hypernet.eval()
        result = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                  metric_key_prefix=metric_key_prefix)
        if was_training:
            self.hypernet.train()
        return result

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if _internal_call:
            return


def prepare_composable_items(raw_items, tok, max_files=50):
    """Prepare training items with file-level embeddings."""
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

        item = {
            "repo": repo,
            "repo_name": repo,
            "repo_embedding": it["repo_embedding"],
            "tokens": tl["tokens"],
            "labels": tl["labels"],
            "prefix_len": tl["prefix_len"],
            "target_len": tl["target_len"],
        }

        # Add file-level embeddings if available
        file_embs = it.get("file_embeddings")
        if file_embs and isinstance(file_embs, list):
            emb_list = [fe["embedding"] for fe in file_embs[:max_files] if "embedding" in fe]
            if emb_list:
                item["file_embeddings_tensor"] = emb_list

        out.append(item)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(os.environ.get("SCRATCH", "~/scratch"),
                                         "TRAINING_CHECKPOINTS/HYPERNET_COMPOSABLE"))
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--composition", type=str, default="weighted",
                    choices=["additive", "weighted", "gated"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit-train-repos", type=int, default=None)
    ap.add_argument("--limit-eval-repos", type=int, default=None)
    args = ap.parse_args()

    from hypernetwork.hypernetwork_sampled import load_from_splits

    set_seed(args.seed)
    splits_dir = Path(args.splits_dir).expanduser().resolve()

    # Load data with file embeddings
    def load_split_with_files(path, limit):
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        repos = data.get("repositories", {})
        repo_names = sorted(repos.keys())
        if limit is not None:
            repo_names = repo_names[:limit]
        items = []
        for repo in repo_names:
            r = repos[repo]
            pairs = r.get("qna_pairs", [])
            emb = r.get("embedding")
            file_embs = r.get("file_embeddings")
            if emb is None:
                continue
            for p in pairs:
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                if not prefix or not target:
                    continue
                items.append({
                    "repo": repo,
                    "prefix": prefix,
                    "target": target,
                    "repo_embedding": emb,
                    "file_embeddings": file_embs,
                })
        return items

    train_items = load_split_with_files(splits_dir / "train.json", args.limit_train_repos)
    eval_items = load_split_with_files(splits_dir / "cr_val.json", args.limit_eval_repos)

    print(f"Loaded: train={len(train_items)} eval={len(eval_items)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    replace_with_lora(model, specs, r=args.rank, alpha=args.alpha)

    train_items = prepare_composable_items(train_items, tok, max_files=args.max_files)
    eval_items = prepare_composable_items(eval_items, tok, max_files=args.max_files)

    n_with_files = sum(1 for it in train_items if "file_embeddings_tensor" in it)
    print(f"Training items with file embeddings: {n_with_files}/{len(train_items)}")

    embedding_dim = len(train_items[0]["repo_embedding"])
    # For file embeddings, use the file embedding dim (half of repo dim if concat)
    file_dim = embedding_dim
    if train_items[0].get("file_embeddings_tensor"):
        file_dim = len(train_items[0]["file_embeddings_tensor"][0])

    print(f"Repo embedding dim: {embedding_dim}, File embedding dim: {file_dim}")

    hypernet = ComposableHypernetwork(
        input_dim=file_dim,
        module_specs=specs,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        composition=args.composition,
        max_files=args.max_files,
    ).cuda()

    print(f"Composable Hypernetwork params: {sum(p.numel() for p in hypernet.parameters()):,}")
    print(f"Composition strategy: {args.composition}")

    train_ds = Dataset.from_list(train_items)
    train_ds = train_ds.shuffle(seed=args.seed)
    train_ds = train_ds.add_column("text", [""] * len(train_ds))
    eval_ds = Dataset.from_list(eval_items)
    eval_ds = eval_ds.add_column("text", [""] * len(eval_ds))

    collator = ComposableDataCollator(
        pad_token_id=tok.pad_token_id,
        max_seq_len=args.max_seq_len,
        max_files=args.max_files,
    )

    output_dir = f"{args.output_dir}_{args.composition}"

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
        save_total_limit=1,
        seed=args.seed,
        bf16=True,
        output_dir=output_dir,
        report_to="wandb",
        max_grad_norm=5.0,
        prediction_loss_only=True,
    )

    opt = torch.optim.AdamW(hypernet.parameters(), lr=sft_cfg.learning_rate, weight_decay=0.01)

    save_cb = SaveHypernetCallback(hypernet, specs)
    save_best_cb = SaveBestHypernetCallback(hypernet, specs)

    trainer = ComposableHypernetTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_cfg,
        optimizers=(opt, None),
        hypernet=hypernet,
        module_specs=specs,
        callbacks=[save_cb, save_best_cb],
    )

    import wandb
    wandb.init(project=f"hypernetwork-composable-{args.composition}")

    print("\nTraining composable hypernetwork...")
    trainer.train()

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
