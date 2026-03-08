#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_METADATA = "REPO_METADATA.json"
QNA_HYPERNET = "QNA_HYPERNET.json"


def _check_repo_embeddings_impl(repos_root: Path, min_examples: int = 30) -> Dict[str, Any]:
    """Scan all repos and report embedding status. No heavy deps."""
    total_repos = 0
    with_meta = 0
    with_qna = 0
    with_embedding = 0
    with_enough_examples = 0
    missing_embedding = []
    missing_qna = []
    embedding_none_or_empty = []

    for author_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
        author = author_dir.name
        for repo_dir in sorted(p for p in author_dir.iterdir() if p.is_dir()):
            repo_name = repo_dir.name
            total_repos += 1
            meta_path = repo_dir / REPO_METADATA
            qna_path = repo_dir / QNA_HYPERNET
            repo_full = f"{author}/{repo_name}"

            if not meta_path.exists():
                continue
            with_meta += 1

            if not qna_path.exists():
                missing_qna.append(repo_full)
                continue
            with_qna += 1

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                emb = meta.get("embedding")
            except (json.JSONDecodeError, OSError):
                missing_embedding.append(repo_full)
                continue

            if emb is None or (isinstance(emb, list) and len(emb) == 0):
                embedding_none_or_empty.append(repo_full)
                continue
            if not isinstance(emb, list):
                missing_embedding.append(repo_full)
                continue
            with_embedding += 1

            try:
                qna = json.loads(qna_path.read_text(encoding="utf-8"))
                pairs = qna.get("pairs", [])
            except (json.JSONDecodeError, OSError):
                continue
            if len(pairs) >= min_examples:
                with_enough_examples += 1

    return {
        "total_repos": total_repos,
        "with_meta": with_meta,
        "with_qna": with_qna,
        "with_embedding": with_embedding,
        "with_enough_examples": with_enough_examples,
        "missing_embedding": missing_embedding,
        "missing_qna": missing_qna,
        "embedding_none_or_empty": embedding_none_or_empty,
    }


# Early exit for --check-embeddings (no torch/trl/transformers needed)
if "--check-embeddings" in sys.argv:
    import argparse
    ap = argparse.ArgumentParser()
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )
    ap.add_argument("--repos-root", type=str, default=default_repos)
    ap.add_argument("--min-examples", type=int, default=30)
    ap.add_argument("--check-embeddings", action="store_true")
    args = ap.parse_args()
    repos_root = Path(args.repos_root).expanduser().resolve()
    report = _check_repo_embeddings_impl(repos_root, min_examples=args.min_examples)
    print("\n[Embedding check]")
    print(f"  total_repos: {report['total_repos']}")
    print(f"  with REPO_METADATA: {report['with_meta']}")
    print(f"  with QNA_HYPERNET: {report['with_qna']}")
    print(f"  with valid embedding: {report['with_embedding']}")
    print(f"  with >= {args.min_examples} examples: {report['with_enough_examples']}")
    if report["embedding_none_or_empty"]:
        n = len(report["embedding_none_or_empty"])
        print(f"  embedding None/empty ({n}): {report['embedding_none_or_empty'][:10]}{'...' if n > 10 else ''}")
    if report["missing_embedding"]:
        n = len(report["missing_embedding"])
        print(f"  missing/invalid embedding ({n}): {report['missing_embedding'][:10]}{'...' if n > 10 else ''}")
    if report["missing_qna"]:
        n = len(report["missing_qna"])
        print(f"  missing QNA_HYPERNET ({n}): {report['missing_qna'][:10]}{'...' if n > 10 else ''}")
    print()
    sys.exit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Heavy imports (torch, trl, transformers) ---
import random
from dataclasses import dataclass
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
import wandb


_DEBUG_STEP = {"global": 0, "collator_calls": 0, "eval_calls": 0, "in_eval": False}
_DEBUG_PRINT_EVERY = 50  # print debug info every N steps
_DEBUG_FIRST_N = 5       # always print first N steps
_DEBUG_EVAL_FIRST_N = 3  # always print first N eval batches

# Store batch fingerprints + losses for train/eval comparison
_DEBUG_BATCH_LOG = {"train": {}, "eval": {}}


def _should_debug_print():
    """Return True if we should print debug info for the current step."""
    if _DEBUG_STEP["in_eval"]:
        idx = _DEBUG_STEP["eval_calls"]
        return idx < _DEBUG_EVAL_FIRST_N or idx % _DEBUG_PRINT_EVERY == 0
    step = _DEBUG_STEP["global"]
    return step < _DEBUG_FIRST_N or step % _DEBUG_PRINT_EVERY == 0


def _batch_fingerprint(input_ids, labels):
    """Create a deterministic fingerprint of a batch for matching train vs eval."""
    ids = input_ids.flatten()
    labs = labels.flatten()
    # Use first 8 + last 8 token ids, seq len, and label count as fingerprint
    first = tuple(ids[:8].tolist())
    last = tuple(ids[-8:].tolist())
    seq_len = int(ids.shape[0])
    n_labels = int((labs != -100).sum().item())
    return f"len={seq_len}|labs={n_labels}|first={first}|last={last}"


def _dbg(tag, msg):
    """Debug print with step info."""
    step = _DEBUG_STEP["global"]
    print(f"[DEBUG step={step}][{tag}] {msg}", flush=True)


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


def load_from_splits(
    splits_dir: Path,
    limit_train_repos: Optional[int],
    limit_eval_repos: Optional[int],
    limit_test_repos: Optional[int],
    oracle_cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load from create_splits output: train.json, cr_val.json, cr_test.json.
    Each has repositories: {repo: {qna_pairs, embedding}}.
    limit_*_repos: take first N repos from each split (None = all).
    If *oracle_cache_dir* is given, prepend oracle context to each prefix.
    Returns (train_items, eval_items, test_items) as flat lists of {repo, prefix, target, repo_embedding}.
    """
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


def load_repos_with_embeddings_and_qnas(
    repos_root: Path,
    min_examples: int,
    max_examples_per_repo: int,
    n_train_repos: int,
    n_eval_repos: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load from repos: REPO_METADATA.json (embedding) + QNA_HYPERNET.json (pairs).
    Returns (train_items, eval_items) where each item has repo, prefix, target, repo_embedding.
    - Only repos with >= min_examples pairs
    - Up to max_examples_per_repo first examples per repo
    - n_train_repos for training, n_eval_repos for evaluation (repo-level split)
    """
    by_repo = {}
    for author_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
        author = author_dir.name
        for repo_dir in sorted(p for p in author_dir.iterdir() if p.is_dir()):
            repo_name = repo_dir.name
            meta_path = repo_dir / REPO_METADATA
            qna_path = repo_dir / QNA_HYPERNET
            if not meta_path.exists() or not qna_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                emb = meta.get("embedding")
                if not emb or not isinstance(emb, list):
                    continue
            except (json.JSONDecodeError, OSError):
                continue
            try:
                qna = json.loads(qna_path.read_text(encoding="utf-8"))
                pairs = qna.get("pairs", [])
            except (json.JSONDecodeError, OSError):
                continue
            if len(pairs) < min_examples:
                continue
            pairs = pairs[:max_examples_per_repo]
            repo_full = f"{author}/{repo_name}"
            items = []
            for p in pairs:
                prefix = p.get("prefix", "")
                target = p.get("target", "")
                if not prefix or not target:
                    continue
                items.append({
                    "repo": repo_full,
                    "repo_name": repo_full,
                    "prefix": prefix,
                    "target": target,
                    "repo_embedding": emb,
                })
            if items:
                by_repo[repo_full] = items
            if len(by_repo) >= n_train_repos + n_eval_repos:
                break
        if len(by_repo) >= n_train_repos + n_eval_repos:
            break

    repo_names = sorted(by_repo.keys())
    random.seed(seed)
    random.shuffle(repo_names)
    n_available = len(repo_names)
    if n_available < n_train_repos + n_eval_repos:
        print(f"Warning: only {n_available} repos with >= {min_examples} examples "
              f"(requested {n_train_repos} train + {n_eval_repos} eval)")
    train_repos = repo_names[:min(n_train_repos, n_available)]
    eval_repos = repo_names[n_train_repos : min(n_train_repos + n_eval_repos, n_available)]

    train_items = []
    for r in train_repos:
        train_items.extend(by_repo[r])
    eval_items = []
    for r in eval_repos:
        eval_items.extend(by_repo[r])

    return train_items, eval_items


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


def left_truncate_left_pad(tokens, labels, max_len, pad_token_id):
    """
    Left truncate: keep rightmost max_len tokens (drop prefix from left when too long).
    Left pad: pad on left when too short.
    tokens = [BOS] + prefix + target, labels = [-100]*(1+prefix_len) + target_ids
    Target is always kept; prefix may be truncated from the left.
    """
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]
        labels = labels[-max_len:]
    if len(tokens) < max_len:
        pad_len = max_len - len(tokens)
        tokens = [pad_token_id] * pad_len + tokens
        labels = [-100] * pad_len + labels
    return tokens, labels


@dataclass
class HypernetDataCollator:
    pad_token_id: int
    max_seq_len: int = 8192

    def __call__(self, examples):
        ex = examples[0]  # batch_size=1
        call_idx = _DEBUG_STEP["collator_calls"]
        _DEBUG_STEP["collator_calls"] += 1

        ctx = torch.tensor(ex["repo_embedding"], dtype=torch.float32).unsqueeze(0)  # [1, dim]

        orig_len = len(ex["tokens"])
        tokens, labels = left_truncate_left_pad(
            tokens=ex["tokens"],
            labels=ex["labels"],
            max_len=self.max_seq_len,
            pad_token_id=self.pad_token_id,
        )

        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]
        n_label_tokens = sum(1 for l in labels if l != -100)

        if _should_debug_print():
            print(f"[DEBUG collator call={call_idx}] repo={ex.get('repo_name','?')!r}  "
                  f"seq_len={orig_len}->{len(tokens)}  "
                  f"prefix={ex['prefix_len']}  target={ex.get('target_len','?')}  "
                  f"labels={n_label_tokens}  ctx_norm={ctx.norm().item():.4f}", flush=True)

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
    do_dbg = _should_debug_print()
    n_injected = 0
    a_norms, b_norms = [], []
    for full_name, _, t, _, _ in module_specs:
        A0 = Ashared[t][batch_index].to(device=device)  # [r, in_f]
        B0 = Bshared[t][batch_index].to(device=device)  # [out_f, r]
        named[full_name].set_lora_weights(A0, B0)
        n_injected += 1
        if do_dbg:
            a_norms.append(A0.norm().item())
            b_norms.append(B0.norm().item())
    if do_dbg:
        _dbg("inject_lora", f"injected {n_injected} modules  "
             f"A_norm: min={min(a_norms):.6f} max={max(a_norms):.6f} mean={sum(a_norms)/len(a_norms):.6f}  "
             f"B_norm: min={min(b_norms):.6f} max={max(b_norms):.6f} mean={sum(b_norms)/len(b_norms):.6f}")


class Hypernetwork(nn.Module):
    def __init__(self, input_dim, module_specs, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 

        # self.ctx_norm = nn.LayerNorm(input_dim)

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
        do_dbg = _should_debug_print()

        # ctx: [B, dim] or [B, K, dim] (if you later pass multiple ctx vectors)
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values

        # ctx = self.ctx_norm(ctx.float())
        
        h = self.trunk(ctx.float())
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)

        if do_dbg:
            _dbg("Hypernet.fwd", f"ctx_normed={ctx.norm().item():.4f}  "
                 f"trunk_h: norm={h.norm().item():.4f}  range=[{h.min().item():.4f},{h.max().item():.4f}]  "
                 f"nan={torch.isnan(h).any().item()}")

        A = {}
        B = {}
        any_nan = False

        for t in self.types:
            in_f, out_f = self.type_shapes[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)

            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)

            A[t] = torch.tanh(A_raw) * scale_A
            B[t] = torch.tanh(B_raw) * scale_B

            if torch.isnan(A[t]).any() or torch.isnan(B[t]).any():
                any_nan = True

        if do_dbg:
            # One compact summary line
            scales = {t: (torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3).item(),
                          torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3).item()) for t in self.types}
            a_means = {t: A[t].abs().mean().item() for t in self.types}
            b_means = {t: B[t].abs().mean().item() for t in self.types}
            _dbg("Hypernet.fwd", f"scales={{{','.join(f'{t}:{s[0]:.4f}/{s[1]:.4f}' for t,s in scales.items())}}}  "
                 f"A_abs_mean=[{min(a_means.values()):.6f}..{max(a_means.values()):.6f}]  "
                 f"B_abs_mean=[{min(b_means.values()):.6f}..{max(b_means.values()):.6f}]  "
                 f"nan={any_nan}")

        if any_nan:
            _dbg("Hypernet.fwd", "WARNING: NaN in A/B outputs!")

        return {"A": A, "B": B}


class HypernetTrainer(SFTTrainer):
    def __init__(self, *args, hypernet, module_specs, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypernet = hypernet
        self._module_specs = module_specs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Determine if we're in train or eval mode
        is_training = model.training
        mode_str = "TRAIN" if is_training else "EVAL"
        do_dbg = _should_debug_print()

        ctx = inputs["ctx"].to(device=model.device, dtype=torch.float32)
        fingerprint = _batch_fingerprint(inputs["input_ids"], inputs["labels"])

        if do_dbg:
            _dbg("compute_loss", f"[{mode_str}] repo={inputs.get('repo_name','?')!r}  "
                 f"batch_fingerprint={fingerprint}  "
                 f"model.training={model.training}  hypernet.training={self.hypernet.training}  "
                 f"ctx_raw_norm={ctx.norm().item():.4f}  "
                 f"input_ids_shape={list(inputs['input_ids'].shape)}  "
                 f"labels_non_masked={int((inputs['labels'] != -100).sum())}")
            # Warn if hypernet mode doesn't match model mode
            if model.training != self.hypernet.training:
                _dbg("compute_loss", f"WARNING: mode mismatch! model.training={model.training} "
                     f"but hypernet.training={self.hypernet.training}")

        ctx = F.normalize(ctx, p=2, dim=-1)

        if do_dbg:
            repo_name = inputs.get('repo_name', '?')
            _dbg("compute_loss", f"[{mode_str}] ctx after L2-norm: repo={repo_name!r}  "
                 f"norm={ctx.norm().item():.4f}  "
                 f"min={ctx.min().item():.4f}  max={ctx.max().item():.4f}")

        h = self.hypernet(ctx)

        inject_lora_weights(model, self._module_specs, h, batch_index=0)

        out = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            labels=inputs["labels"].to(model.device),
        )
        loss = out["loss"] if isinstance(out, dict) else out[0]

        if do_dbg:
            _dbg("compute_loss", f"[{mode_str}] loss={loss.item():.4f}  "
                 f"is_nan={torch.isnan(loss).item()}  is_inf={torch.isinf(loss).item()}")

        if torch.isnan(loss) or torch.isinf(loss):
            _dbg("compute_loss", f"[{mode_str}] WARNING: loss is {'NaN' if torch.isnan(loss) else 'Inf'}!  "
                 f"repo={inputs.get('repo_name','?')!r}")

        # Record for train/eval comparison
        record_key = "train" if is_training else "eval"
        repo = inputs.get("repo_name", "?")
        _DEBUG_BATCH_LOG[record_key][fingerprint] = {
            "repo": repo,
            "loss": loss.item(),
            "mode": mode_str,
            "hypernet_training": self.hypernet.training,
        }

        if not is_training:
            _DEBUG_STEP["eval_calls"] += 1
            # Check if we've seen this exact batch during training
            if fingerprint in _DEBUG_BATCH_LOG["train"]:
                train_rec = _DEBUG_BATCH_LOG["train"][fingerprint]
                delta = loss.item() - train_rec["loss"]
                _dbg("compute_loss", f"[COMPARE] Same batch seen in TRAIN!  "
                     f"repo={repo!r}  train_loss={train_rec['loss']:.4f}  eval_loss={loss.item():.4f}  "
                     f"delta={delta:+.4f}  "
                     f"train_hypernet_mode={'train' if train_rec['hypernet_training'] else 'eval'}  "
                     f"eval_hypernet_mode={'train' if self.hypernet.training else 'eval'}")

        return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        step = _DEBUG_STEP["global"]
        do_dbg = _should_debug_print()

        if do_dbg:
            _dbg("training_step", f"=== START step={step} ===")

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Check gradient stats BEFORE clipping
        if do_dbg:
            grad_norms = []
            n_no_grad = 0
            n_nan_grad = 0
            for name, p in self.hypernet.named_parameters():
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    grad_norms.append(gn)
                    if torch.isnan(p.grad).any():
                        n_nan_grad += 1
                        _dbg("training_step", f"  NaN grad in param: {name}")
                else:
                    n_no_grad += 1
            if grad_norms:
                _dbg("training_step", f"PRE-clip grad norms: "
                     f"min={min(grad_norms):.6f}  max={max(grad_norms):.6f}  "
                     f"mean={sum(grad_norms)/len(grad_norms):.6f}  "
                     f"n_params_with_grad={len(grad_norms)}  n_no_grad={n_no_grad}  n_nan_grad={n_nan_grad}")
            else:
                _dbg("training_step", "WARNING: No hypernet params have gradients!")

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.hypernet.parameters(), self.args.max_grad_norm)
            if do_dbg:
                _dbg("training_step", f"POST-clip total_grad_norm={total_norm:.6f}  "
                     f"max_grad_norm={self.args.max_grad_norm}")

        if do_dbg:
            # Check hypernet param stats
            param_norms = {name: p.norm().item() for name, p in self.hypernet.named_parameters()}
            max_pn = max(param_norms.values())
            min_pn = min(param_norms.values())
            mean_pn = sum(param_norms.values()) / len(param_norms)
            _dbg("training_step", f"Hypernet param norms: min={min_pn:.6f}  max={max_pn:.6f}  mean={mean_pn:.6f}")
            _dbg("training_step", f"=== END step={step}  loss={loss.item():.4f} ===\n")

        _DEBUG_STEP["global"] += 1
        return loss.detach() / self.args.gradient_accumulation_steps



    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Wrap evaluate to set hypernet to eval mode and log comparison."""
        _DEBUG_STEP["eval_calls"] = 0
        _DEBUG_STEP["in_eval"] = True
        _DEBUG_BATCH_LOG["eval"] = {}

        was_training = self.hypernet.training
        self.hypernet.eval()
        print(f"\n[DEBUG][evaluate] === EVAL START ===  "
              f"hypernet set to eval mode (was training={was_training})  "
              f"global_train_step={_DEBUG_STEP['global']}  "
              f"metric_key_prefix={metric_key_prefix!r}", flush=True)

        result = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                  metric_key_prefix=metric_key_prefix)
        _DEBUG_STEP["in_eval"] = False

        # Print comparison summary
        n_eval = len(_DEBUG_BATCH_LOG["eval"])
        n_matched = sum(1 for fp in _DEBUG_BATCH_LOG["eval"] if fp in _DEBUG_BATCH_LOG["train"])
        print(f"[DEBUG][evaluate] === EVAL END ===  "
              f"eval_batches={n_eval}  matched_with_train={n_matched}", flush=True)

        if n_matched > 0:
            deltas = []
            for fp in _DEBUG_BATCH_LOG["eval"]:
                if fp in _DEBUG_BATCH_LOG["train"]:
                    d = _DEBUG_BATCH_LOG["eval"][fp]["loss"] - _DEBUG_BATCH_LOG["train"][fp]["loss"]
                    deltas.append(d)
                    repo = _DEBUG_BATCH_LOG["eval"][fp]["repo"]
                    print(f"[DEBUG][evaluate] MATCH repo={repo!r}  "
                          f"train_loss={_DEBUG_BATCH_LOG['train'][fp]['loss']:.4f}  "
                          f"eval_loss={_DEBUG_BATCH_LOG['eval'][fp]['loss']:.4f}  "
                          f"delta={d:+.4f}", flush=True)
            if deltas:
                import numpy as np
                print(f"[DEBUG][evaluate] MATCH SUMMARY: n={len(deltas)}  "
                      f"mean_delta={np.mean(deltas):+.6f}  "
                      f"std_delta={np.std(deltas):.6f}  "
                      f"max_abs_delta={max(abs(d) for d in deltas):.6f}", flush=True)
        else:
            print(f"[DEBUG][evaluate] No overlapping batches between train and eval "
                  f"(train has {len(_DEBUG_BATCH_LOG['train'])} recorded fingerprints)", flush=True)

        print(f"[DEBUG][evaluate] result={result}", flush=True)

        # Restore hypernet to training mode if it was before
        if was_training:
            self.hypernet.train()
            print(f"[DEBUG][evaluate] hypernet restored to train mode", flush=True)

        return result

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset
    
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if _internal_call:
            return


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
                "hidden_dim": self.hypernet.hidden_dim,
                "rank": self.hypernet.rank,
                "types": self.hypernet.types,
                "type_shapes": self.hypernet.type_shapes,
            }
        }

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved hypernet -> {path}")

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, self.filename)
        torch.save(self._payload(), path)
        print(f"Saved final hypernet -> {path}")


class SaveBestHypernetCallback(TrainerCallback):
    """Saves the hypernet whenever eval_loss improves."""
    def __init__(self, hypernet: Hypernetwork, module_specs, filename="hypernet_best.pt"):
        self.hypernet = hypernet
        self.module_specs = module_specs
        self.filename = filename
        self.best_eval_loss = float("inf")

    def _payload(self):
        return {
            "hypernet_state_dict": self.hypernet.state_dict(),
            "module_specs": self.module_specs,
            "hypernet_config": {
                "input_dim": self.hypernet.input_dim,
                "hidden_dim": self.hypernet.hidden_dim,
                "rank": self.hypernet.rank,
                "types": self.hypernet.types,
                "type_shapes": self.hypernet.type_shapes,
            }
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
            print(f"Saved best hypernet (eval_loss={eval_loss:.4f}) -> {path}")


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
            "task": it.get("task", ""),
            "framework": it.get("framework", ""),
            "metadata": it.get("metadata", {}),
        })
    return out


def to_hf_dataset(items, seed, shuffle):
    ds = Dataset.from_list(items)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.add_column("text", [""] * len(ds))
    return ds


def main():
    ap = argparse.ArgumentParser()

    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Dir with train.json, cr_val.json, etc.")
    ap.add_argument("--limit-train-repos", type=int, default=None,
                    help="Use only first N repos from train.json (default: all)")
    ap.add_argument("--limit-eval-repos", type=int, default=None,
                    help="Use only first N repos from cr_val.json (default: all)")
    ap.add_argument("--limit-test-repos", type=int, default=None,
                    help="Use only first N repos from cr_test.json (default: all)")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--output-dir", type=str, default="/scratch/lhotsko/TRAINING_CHECKPOINTS/HYPERNET/full_repos")

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)

    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--use-oracle", action="store_true",
                    help="Prepend oracle context to prefixes")
    ap.add_argument("--oracle-cache-dir", type=str, default=None,
                    help="Dir with pre-built oracle contexts")

    ap.add_argument("--hidden-dim", type=int, default=512)

    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--check-embeddings", action="store_true",
                    help="Only check embedding status across repos and exit")

    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = splits_dir / "repositories"
    emb_report = _check_repo_embeddings_impl(repos_root, min_examples=30)
    print("\n[Embedding check]")
    print(f"  total_repos: {emb_report['total_repos']}")
    print(f"  with REPO_METADATA: {emb_report['with_meta']}")
    print(f"  with QNA_HYPERNET: {emb_report['with_qna']}")
    print(f"  with valid embedding: {emb_report['with_embedding']}")
    print(f"  with >= 30 examples: {emb_report['with_enough_examples']}")
    if emb_report["embedding_none_or_empty"]:
        print(f"  embedding None/empty ({len(emb_report['embedding_none_or_empty'])}): "
              f"{emb_report['embedding_none_or_empty'][:10]}{'...' if len(emb_report['embedding_none_or_empty']) > 10 else ''}")
    if emb_report["missing_embedding"]:
        print(f"  missing/invalid embedding ({len(emb_report['missing_embedding'])}): "
              f"{emb_report['missing_embedding'][:10]}{'...' if len(emb_report['missing_embedding']) > 10 else ''}")
    if emb_report["missing_qna"]:
        print(f"  missing QNA_HYPERNET ({len(emb_report['missing_qna'])}): "
              f"{emb_report['missing_qna'][:10]}{'...' if len(emb_report['missing_qna']) > 10 else ''}")
    print()

    if args.check_embeddings:
        print("--check-embeddings: exiting.")
        return

    set_seed(args.seed)

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
    print("[DEBUG] CONFIGURATION:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 80, flush=True)

    wandb.init(project="hypernetwork-REPOPEFTDATA_full_repos", name=args.output_dir.split("/")[-1])

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[DEBUG] Tokenizer: vocab_size={tok.vocab_size}  "
          f"pad_token_id={tok.pad_token_id}  eos_token_id={tok.eos_token_id}  "
          f"bos_token_id={tok.bos_token_id}", flush=True)

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
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[DEBUG] Base model: total_params={total_params:,}  frozen={frozen_params:,}  "
          f"dtype={next(model.parameters()).dtype}  device={next(model.parameters()).device}", flush=True)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    replace_with_lora(model, specs, r=args.rank, alpha=args.alpha)
    print(f"Replaced {len(specs)} modules with LoRA wrappers (rank={args.rank})")

    # Print breakdown by module type
    from collections import Counter
    type_counts = Counter(s[2] for s in specs)
    for mtype, cnt in sorted(type_counts.items()):
        shape = next((s[3], s[4]) for s in specs if s[2] == mtype)
        print(f"  [DEBUG] LoRA type={mtype}: count={cnt}  in_features={shape[0]}  out_features={shape[1]}")
    print(f"  [DEBUG] LoRA alpha={args.alpha}  scale={float(args.alpha)/float(args.rank):.2f}", flush=True)

    print(f"Loaded from {splits_dir}: train={len(train_items)} eval={len(eval_items)} test={len(test_items)} examples")

    train_items = prepare_training_items(train_items, tok)
    eval_items = prepare_training_items(eval_items, tok)
    test_items = prepare_training_items(test_items, tok)
    print(f"[DEBUG] After prepare_training_items: train={len(train_items)}  eval={len(eval_items)}  test={len(test_items)}")

    all_items = train_items + eval_items + test_items
    if all_items:
        missing = [it["repo"] for it in all_items if it.get("repo_embedding") is None]
        if missing:
            raise ValueError(f"{len(missing)} items missing embedding: {missing[:5]}...")
    if not train_items:
        raise ValueError("No training items loaded. Run create_splits.py first, check --splits-dir.")

    embedding_dim = len(train_items[0]["repo_embedding"])
    print(f"Embedding dim: {embedding_dim}")

    # Print token length statistics
    import numpy as np
    for split_name, items in [("train", train_items), ("eval", eval_items), ("test", test_items)]:
        if items:
            lengths = [len(it["tokens"]) for it in items]
            repos = set(it["repo"] for it in items)
            print(f"[DEBUG] {split_name}: n_examples={len(items)}  n_unique_repos={len(repos)}  "
                  f"seq_len: min={min(lengths)}  max={max(lengths)}  "
                  f"mean={np.mean(lengths):.0f}  median={np.median(lengths):.0f}  "
                  f">{args.max_seq_len}: {sum(1 for l in lengths if l > args.max_seq_len)}", flush=True)

    train_ds = to_hf_dataset(train_items, seed=args.seed, shuffle=True)
    eval_ds = to_hf_dataset(eval_items, seed=args.seed, shuffle=False)
    test_ds = to_hf_dataset(test_items, seed=args.seed, shuffle=False)

    collator = HypernetDataCollator(
        pad_token_id=tok.pad_token_id,
        max_seq_len=args.max_seq_len,
    )

    hypernet = Hypernetwork(
        input_dim=embedding_dim,
        module_specs=specs,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).cuda()

    n_hypernet_params = sum(p.numel() for p in hypernet.parameters())
    print(f"Hypernet params: {n_hypernet_params:,}")
    print(f"[DEBUG] Hypernet architecture:")
    for name, p in hypernet.named_parameters():
        print(f"  {name}: shape={list(p.shape)}  numel={p.numel():,}  dtype={p.dtype}  device={p.device}")
    print(f"[DEBUG] Hypernet memory: ~{n_hypernet_params * 4 / 1024**2:.1f} MB (fp32)", flush=True)

    sft_cfg = SFTConfig(
        dataset_text_field="text",
        label_names=["labels"],
        remove_unused_columns=False,
        max_length=args.max_seq_len,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4,
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

    # opt = torch.optim.AdamW(hypernet.parameters(), lr=sft_cfg.learning_rate, weight_decay=sft_cfg.weight_decay)
    opt = torch.optim.AdamW(hypernet.parameters(), lr=sft_cfg.learning_rate, weight_decay=0.01)

    print(f"[DEBUG] Optimizer: AdamW  lr={sft_cfg.learning_rate}  weight_decay={sft_cfg.weight_decay}")
    print(f"[DEBUG] SFT config: batch_size=1  grad_accum={args.grad_accum}  effective_batch={args.grad_accum}  "
          f"epochs={args.epochs}  max_grad_norm={sft_cfg.max_grad_norm}  bf16={sft_cfg.bf16}")
    total_steps = len(train_items) * args.epochs // args.grad_accum
    print(f"[DEBUG] Estimated total optimization steps: ~{total_steps}", flush=True)
    save_cb = SaveHypernetCallback(hypernet, specs)
    save_best_cb = SaveBestHypernetCallback(hypernet, specs)

    trainer = HypernetTrainer(
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

    print("\nInitial eval...")
    model.eval()
    init_val = trainer.evaluate()
    print(f"init_val_loss = {init_val['eval_loss']}")
    print(f"[DEBUG] Full init eval metrics: {init_val}", flush=True)
    wandb.log({"init_val_loss": init_val["eval_loss"]})

    # Reset debug step counter for training
    _DEBUG_STEP["global"] = 0
    print(f"\n[DEBUG] Starting training. Debug prints every {_DEBUG_PRINT_EVERY} steps "
          f"(always first {_DEBUG_FIRST_N} steps)", flush=True)
    print("\nTraining...")
    trainer.train()

    print(f"\n[DEBUG] Training complete. Total steps executed: {_DEBUG_STEP['global']}", flush=True)

    print("\nFinal eval...")
    final_val = trainer.evaluate()
    print("final_val_loss =", final_val["eval_loss"])
    wandb.log({"final_val_loss": final_val["eval_loss"]})

    print("\nTest eval (cr_test)...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    test_loss = test_metrics.get("test_eval_loss")
    if test_loss is not None:
        print(f"test_loss = {test_loss}")
        wandb.log({"test_loss": test_loss})

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    main()
