#!/usr/bin/env python3

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
import wandb


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def load_embeddings_dict(emb_dir):
    out = {}
    for p in sorted(emb_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not obj.get("ok", True):
                continue
            v = obj.get("repo_embedding")
            if v is None:
                continue
            out[p.stem] = v
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    return out


def repo_to_embedding_key(repo):
    return repo.replace("/", "__")


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


def smart_truncate_keep_two_part_prefix(tokens, labels, prefix_len, max_len, head_keep= 256, tail_keep_min = 512):
    """

      tokens = [BOS] + prefix + target
      labels = [-100]*(1+prefix_len) + target

    """
    if len(tokens) <= max_len:
        return tokens, labels

    bos = tokens[0]
    prefix_tokens = tokens[1 : 1 + prefix_len]
    target_tokens = tokens[1 + prefix_len :]
    target_labels = labels[1 + prefix_len :]

    keep_target_len = 1 + len(target_tokens)

    if keep_target_len >= max_len:
        new_tokens = tokens[-max_len:]
        new_labels = labels[-max_len:]
        return new_tokens, new_labels

    space_for_prefix = max_len - keep_target_len

    head = prefix_tokens[: min(head_keep, len(prefix_tokens))]
    head_labels = [-100] * len(head)

    remaining = space_for_prefix - len(head)
    if remaining < 0:
        head = head[:space_for_prefix]
        head_labels = [-100] * len(head)
        tail = []
        tail_labels = []
    else:
        tail_take = min(len(prefix_tokens) - len(head), remaining)
        tail = prefix_tokens[-tail_take:] if tail_take > 0 else []
        tail_labels = [-100] * len(tail)

    new_prefix = head + tail
    new_tokens = [bos] + new_prefix + target_tokens
    new_labels = [-100] + ([-100] * len(new_prefix)) + target_labels

    return new_tokens, new_labels


@dataclass
class HypernetDataCollator:
    pad_token_id: int
    max_seq_len: int = 8192
    head_keep: int = 256
    tail_keep_min: int = 512

    def __call__(self, examples):
        ex = examples[0]  # batch_size=1

        ctx = torch.tensor(ex["repo_embedding"], dtype=torch.float32).unsqueeze(0)  # [1, dim]

        tokens, labels = smart_truncate_keep_two_part_prefix(
            tokens=ex["tokens"],
            labels=ex["labels"],
            prefix_len=ex["prefix_len"],
            max_len=self.max_seq_len,
            head_keep=self.head_keep,
            tail_keep_min=self.tail_keep_min,
        )

        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]

        return {
            "repo_name": ex.get("repo_name", ""),
            "ctx": ctx,  # [1, dim] fp32 CPU
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }


class LoRA(nn.Module):
    def __init__(self, base, in_f, out_f, r, alpha):
        super().__init__()
        self.base = base
        self.in_f, self.out_f, self.r = in_f, out_f, r
        self.scale = float(alpha) / float(max(1, r))
        self.A: Optional[torch.Tensor] = None  # [r, in_f]
        self.B: Optional[torch.Tensor] = None  # [out_f, r]

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor):
        self.A, self.B = A, B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is None or self.B is None:
            return y

        x2 = x.detach().to(torch.float32)
        A = self.A.to(torch.float32)
        B = self.B.to(torch.float32)

        delta = (x2.reshape(-1, self.in_f) @ A.transpose(0, 1)) @ B.transpose(0, 1)
        delta = delta.view(*x.shape[:-1], self.out_f) * self.scale

        return y + delta.to(dtype=y.dtype)


def get_module_specs(model, target_modules):
    specs = []
    pat = re.compile(r"\bmodel\.layers\.(\d+)\.")
    for name, m in model.named_modules():
        if any(tm in name for tm in target_modules):
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                m_layer = pat.search(name)
                layer_idx = int(m_layer.group(1)) if m_layer else -1
                m_type = next(tm for tm in target_modules if tm in name)
                specs.append((name, layer_idx, m_type, int(m.in_features), int(m.out_features)))
    specs.sort(key=lambda t: (t[1], t[0]))
    return specs


def replace_with_lora(model: nn.Module, module_specs, r: int, alpha: int):
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for full_name, _, _, in_f, out_f in module_specs:
        parent, attr = full_name.rsplit(".", 1)
        orig = getattr(named[parent], attr)
        if isinstance(orig, LoRA):
            continue
        assert isinstance(orig, nn.Linear), f"{full_name} is not nn.Linear"
        wrapped = LoRA(orig, in_f, out_f, r, alpha).to(device=device, dtype=dtype)
        setattr(named[parent], attr, wrapped)


def inject_lora_weights(model: nn.Module, module_specs, hyper_out: Dict[str, Any], batch_index: int = 0):
    Ashared, Bshared = hyper_out["A"], hyper_out["B"]
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    for full_name, _, t, _, _ in module_specs:
        A0 = Ashared[t][batch_index].to(device=device)  # [r, in_f]
        B0 = Bshared[t][batch_index].to(device=device)  # [out_f, r]
        named[full_name].set_lora_weights(A0, B0)


class Hypernetwork(nn.Module):
    def __init__(self, input_dim, module_specs, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim

        self.ctx_norm = nn.LayerNorm(input_dim)

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

        self.heads_A = nn.ModuleDict({t: nn.Linear(hidden_dim, rank * type_shapes[t][0]) for t in self.types})
        self.heads_B = nn.ModuleDict({t: nn.Linear(hidden_dim, type_shapes[t][1] * rank) for t in self.types})

        self.log_scale_A = nn.ParameterDict({t: nn.Parameter(torch.tensor(-3.5)) for t in self.types})
        self.log_scale_B = nn.ParameterDict({t: nn.Parameter(torch.tensor(-3.5)) for t in self.types})

    def forward(self, ctx: torch.Tensor):
        # ctx: [B, dim] or [B, K, dim] (if you later pass multiple ctx vectors)
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values

        ctx = self.ctx_norm(ctx.float())
        h = self.trunk(ctx.float())

        A = {}
        B = {}

        for t in self.types:
            in_f, out_f = self.type_shapes[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)

            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)

            A[t] = torch.tanh(A_raw) * scale_A
            B[t] = torch.tanh(B_raw) * scale_B

        return {"A": A, "B": B}


class HypernetTrainer(SFTTrainer):
    def __init__(self, *args, hypernet, module_specs, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypernet = hypernet
        self._module_specs = module_specs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ctx = inputs["ctx"].to(device=model.device, dtype=torch.float32)
        ctx = F.normalize(ctx, p=2, dim=-1)

        h = self.hypernet(ctx)

        # for key in h["A"]:
        #     if torch.isnan(h["A"][key]).any() or torch.isnan(h["B"][key]).any():
        #         loss = torch.tensor(10.0, device=model.device, dtype=torch.float32)
        #         return (loss, None) if return_outputs else loss

        inject_lora_weights(model, self._module_specs, h, batch_index=0)

        out = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            labels=inputs["labels"].to(model.device),
        )
        loss = out["loss"] if isinstance(out, dict) else out[0]
        # if torch.isnan(loss) or torch.isinf(loss):
        #     loss = torch.tensor(10.0, device=model.device, dtype=torch.float32)

        return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.hypernet.parameters(), self.args.max_grad_norm)

        return loss.detach()



    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset
    
    def save_model(self, output_dir=None, _internal_call=False):
        if _internal_call: 
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)


class SaveHypernetCallback(TrainerCallback):
    def __init__(self, hypernet: Hypernetwork, module_specs, filename="hypernet_state.pt"):
        self.hypernet = hypernet
        self.module_specs = module_specs
        self.filename = filename

    def _payload(self):
        return {
            "hypernet_state_dict": self.hypernet.state_dict(),
            "module_specs": self.module_specs,
            "hypernet_config": {
                "input_dim": self.hypernet.input_dim,
                "rank": self.hypernet.rank,
                "types": self.hypernet.types,
                "type_shapes": self.hypernet.type_shapes,
            }
        }

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(ckpt_dir):
            path = os.path.join(ckpt_dir, self.filename)
            torch.save(self._payload(), path)
            print(f"Saved hypernet -> {path}")

    def on_train_end(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved final hypernet -> {path}")


def prepare_training_items(raw_items, tok):
    out = []
    for it in raw_items:
        repo = it.get("repo", "")
        prefix = it.get("prefix", "")
        target = it.get("target", "")
        if not repo or not prefix or not target:
            continue

        tl = prepare_tokens_and_labels(prefix, target, tok, add_eos=True)

        out.append({
            "repo": repo,
            "repo_name": repo,
            "tokens": tl["tokens"],
            "labels": tl["labels"],
            "prefix_len": tl["prefix_len"],
            "target_len": tl["target_len"],
            "task": it.get("task", ""),
            "framework": it.get("framework", ""),
            "metadata": it.get("metadata", {}),
        })
    return out


def attach_embeddings(items, emb):
    kept = []
    missing = 0
    for it in items:
        key = repo_to_embedding_key(it["repo"])
        v = emb.get(key)
        if v is None:
            missing += 1
            continue
        it["repo_embedding"] = v
        kept.append(it)
    if missing:
        print(f"!!! Dropped {missing} examples missing embeddings")
    return kept


def to_hf_dataset(items, seed, shuffle):
    ds = Dataset.from_list(items)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.add_column("text", [""] * len(ds))
    return ds


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-jsonl", type=str, default="/scratch/lhotsko/train_qna_pairs/test_next_block.jsonl")
    ap.add_argument("--val-jsonl", type=str, default="/scratch/lhotsko/val_qna_pairs/test_next_block.jsonl")
    ap.add_argument("--test-jsonl", type=str, default="/scratch/lhotsko/test_qna_pairs/test_next_block.jsonl")

    ap.add_argument("--emb-dir", type=str, default="/scratch/lhotsko/repo_embeddings")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--output-dir", type=str, default="/scratch/lhotsko/model_checkpoints/hypernet_repo_splits_v2")

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)

    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--head-keep", type=int, default=256)
    ap.add_argument("--tail-keep-min", type=int, default=512)

    ap.add_argument("--hidden-dim", type=int, default=512)

    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)

    args = ap.parse_args()

    set_seed(args.seed)

    wandb.init(project="hypernetwork-REPOPEFTDATA", name=args.output_dir.split("/")[-1])

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

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    replace_with_lora(model, specs, r=args.rank, alpha=args.alpha)
    print(f"Replaced {len(specs)} modules with LoRA wrappers (rank={args.rank})")

    emb_dir = Path(args.emb_dir).expanduser().resolve()
    emb = load_embeddings_dict(emb_dir)
    print(f"Loaded {len(emb)} repo embeddings from {emb_dir}")

    train_raw = load_jsonl(Path(args.train_jsonl))
    val_raw = load_jsonl(Path(args.val_jsonl))
    test_raw = load_jsonl(Path(args.test_jsonl))
    print(f"QnA sizes: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")

    train_items = attach_embeddings(prepare_training_items(train_raw, tok), emb)
    val_items = attach_embeddings(prepare_training_items(val_raw, tok), emb)
    test_items = attach_embeddings(prepare_training_items(test_raw, tok), emb)

    embedding_dim = len(train_items[0]["repo_embedding"])
    print(f"Embedding dim: {embedding_dim}")

    train_ds = to_hf_dataset(train_items, seed=args.seed, shuffle=True)
    val_ds = to_hf_dataset(val_items, seed=args.seed, shuffle=False)
    test_ds = to_hf_dataset(test_items, seed=args.seed, shuffle=False)

    collator = HypernetDataCollator(
        pad_token_id=tok.pad_token_id,
        max_seq_len=args.max_seq_len,
        head_keep=args.head_keep,
        tail_keep_min=args.tail_keep_min,
    )

    hypernet = Hypernetwork(
        input_dim=embedding_dim,
        module_specs=specs,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).cuda()

    print(f"Hypernet params: {sum(p.numel() for p in hypernet.parameters()):,}")

    sft_cfg = SFTConfig(
        dataset_text_field="text",
        label_names=["labels"],
        remove_unused_columns=False,
        max_length=args.max_seq_len,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        seed=args.seed,
        bf16=True,
        output_dir=args.output_dir,
        report_to="wandb",
        max_grad_norm=0.3,
        prediction_loss_only=True,
    )

    opt = torch.optim.AdamW(hypernet.parameters(), lr=sft_cfg.learning_rate, weight_decay=sft_cfg.weight_decay)
    save_cb = SaveHypernetCallback(hypernet, specs)

    trainer = HypernetTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        args=sft_cfg,
        optimizers=(opt, None),
        hypernet=hypernet,
        module_specs=specs,
        callbacks=[save_cb],
    )

    print("\nInitial eval...")
    model.eval()
    init_val = trainer.evaluate()
    print("init_val_loss =", init_val["eval_loss"])
    wandb.log({"init_val_loss": init_val["eval_loss"]})


    print("\nTraining...")
    trainer.train()


    print("\nFinal eval...")
    final_val = trainer.evaluate()
    print("final_val_loss =", final_val["eval_loss"])
    wandb.log({"final_val_loss": final_val["eval_loss"]})

    print("\nTEST eval (only once, after training)...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    test_loss = test_metrics.get("test_eval_loss", None)
    print("test_loss =", test_loss)
    if test_loss is not None:
        wandb.log({"test_loss": test_loss})

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    main()
