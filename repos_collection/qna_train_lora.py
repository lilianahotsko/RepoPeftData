#!/usr/bin/env python3
"""
repos_collection/qna_train_lora.py

End-to-end pipeline for a *single* repository:
  1. Pick a repo from scratch/pulled_repos/
  2. Analyse test files – resolve imports to regular_files, compress the
     imported source into a context block that preserves signatures, types,
     docstrings and the information each specific QnA actually needs.
  3. Generate N QnA pairs  (default 20)  — task = "complete the next block
     of test code given the repo context".
  4. Split 80 / 20  train / validation.
  5. Train a repo-specific LoRA adapter on Qwen2.5-Coder-1.5B.

Usage:
    # List available repos
    python repos_collection/qna_train_lora.py --list-repos

    # Run full pipeline for one repo
    python repos_collection/qna_train_lora.py --repo "owner/repo"

    # Generate QnA only (skip training)
    python repos_collection/qna_train_lora.py --repo "owner/repo" --qna-only

    # Override number of pairs / output dir
    python repos_collection/qna_train_lora.py --repo "owner/repo" --num-pairs 30
"""

import os
import sys
import ast
import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

PULLED_REPOS_ROOT = "/home/lhotsko/scratch/pulled_repos"
OUTPUT_ROOT = "/home/lhotsko/scratch/repo_lora"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"

SEED = 42
NUM_PAIRS = 20
TRAIN_RATIO = 0.8

MAX_CONTEXT_LINES = 150   # compressed context budget (lines)
MAX_PREFIX_LINES = 200    # test-file prefix budget
MAX_TARGET_LINES = 15     # target code block
MIN_TEST_LINES = 4        # minimum test-function length to be useful


# ═══════════════════════════════════════════════════════════════════════
# Part 1 — Repo discovery
# ═══════════════════════════════════════════════════════════════════════

def list_available_repos(root: str) -> List[str]:
    """Return owner/repo names that have both test_files/ and regular_files/."""
    repos = []
    if not os.path.isdir(root):
        return repos
    for owner in sorted(os.listdir(root)):
        owner_path = os.path.join(root, owner)
        if not os.path.isdir(owner_path):
            continue
        for repo in sorted(os.listdir(owner_path)):
            repo_path = os.path.join(owner_path, repo)
            if (os.path.isdir(os.path.join(repo_path, "test_files"))
                    and os.path.isdir(os.path.join(repo_path, "regular_files"))):
                repos.append(f"{owner}/{repo}")
    return repos


def get_repo_dirs(root: str, repo_name: str) -> Tuple[str, str]:
    """Return (test_files_dir, regular_files_dir)."""
    base = os.path.join(root, repo_name)
    return os.path.join(base, "test_files"), os.path.join(base, "regular_files")


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — File indexing & module resolution
# ═══════════════════════════════════════════════════════════════════════

def collect_python_files(root_dir: str) -> Dict[str, str]:
    """Return {relative_path: absolute_path} for every .py under *root_dir*."""
    result: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".py"):
                abspath = os.path.join(dirpath, fn)
                relpath = os.path.relpath(abspath, root_dir)
                result[relpath] = abspath
    return result


def build_module_index(regular_files: Dict[str, str]) -> Dict[str, str]:
    """Map dotted module names → absolute paths.

    ``mypackage/utils.py``  →  ``mypackage.utils``
    ``mypackage/__init__.py`` → ``mypackage``
    """
    index: Dict[str, str] = {}
    for relpath, abspath in regular_files.items():
        parts = list(Path(relpath).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            module = ".".join(parts)
            index[module] = abspath
    return index


# ═══════════════════════════════════════════════════════════════════════
# Part 3 — Import analysis
# ═══════════════════════════════════════════════════════════════════════

def extract_imports(source: str) -> List[Dict]:
    """Return a list of ``{module, names, alias}`` dicts from import stmts."""
    imports: List[Dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "names": None, "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = [a.name for a in node.names] if node.names else None
                imports.append({"module": node.module, "names": names, "alias": None})
    return imports


def resolve_import(imp: Dict, module_index: Dict[str, str]) -> Optional[str]:
    """Try to map an import dict to an absolute path in regular_files."""
    module = imp["module"]

    # Exact match
    if module in module_index:
        return module_index[module]

    # Try progressively shorter prefixes
    parts = module.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in module_index:
            return module_index[candidate]

    # Suffix match (handles cases where top-level package name differs)
    for mod_name, path in module_index.items():
        if mod_name.endswith("." + module) or module.endswith("." + mod_name):
            return path

    return None


# ═══════════════════════════════════════════════════════════════════════
# Part 4 — Source compression
# ═══════════════════════════════════════════════════════════════════════

def _get_docstring_end(node: ast.AST) -> Optional[int]:
    """Return the 0-indexed end line of the node's docstring, or None."""
    if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), (ast.Constant,))):
        return getattr(node.body[0], "end_lineno", None)
    return None


def _compress_function(node: ast.AST, lines: List[str],
                       max_body_lines: int = 5) -> str:
    """Signature + docstring + first few body lines, then ``...``."""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    func_lines = lines[start:end]

    if not func_lines:
        return ""

    # How many lines for the signature (including decorators)?
    body_start_offset = (node.body[0].lineno - 1 - start) if node.body else 1
    sig_lines = func_lines[:max(body_start_offset, 1)]

    # Docstring
    ds_end = _get_docstring_end(node)
    ds_offset = (ds_end - start) if ds_end else body_start_offset
    doc_lines = func_lines[body_start_offset:ds_offset]

    # Remaining body
    remaining = func_lines[ds_offset:]
    if len(remaining) > max_body_lines:
        remaining = remaining[:max_body_lines] + ["        ...  # truncated"]

    return "\n".join(sig_lines + doc_lines + remaining)


def _compress_class(node: ast.ClassDef, lines: List[str],
                    max_method_body: int = 3) -> str:
    """Class header + docstring + method signatures with truncated bodies."""
    segments: List[str] = []

    # Class definition line (possibly multi-line with bases)
    cls_start = node.lineno - 1
    body_first = node.body[0].lineno - 1 if node.body else cls_start + 1
    segments.extend(lines[cls_start:body_first])

    # Docstring
    ds_end = _get_docstring_end(node)
    if ds_end:
        segments.extend(lines[body_first:ds_end])

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segments.append(_compress_function(child, lines, max_body_lines=max_method_body))
        elif isinstance(child, ast.Assign):
            a_start = child.lineno - 1
            a_end = getattr(child, "end_lineno", child.lineno)
            seg = "\n".join(lines[a_start:a_end])
            if len(seg) < 200:
                segments.append(seg)

    return "\n".join(segments)


def compress_source(source: str,
                    wanted_names: Optional[List[str]] = None,
                    max_lines: int = 80) -> str:
    """Return a compressed version of *source* keeping only *wanted_names*.

    If *wanted_names* is ``None`` every top-level definition is included
    (with bodies truncated).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(source.splitlines()[:max_lines])

    lines = source.splitlines()
    segments: List[str] = []

    # Always keep imports
    import_lines: List[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s = node.lineno - 1
            e = getattr(node, "end_lineno", node.lineno)
            import_lines.extend(lines[s:e])
    if import_lines:
        segments.append("\n".join(import_lines))

    # Definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if wanted_names and node.name not in wanted_names:
                continue
            segments.append(_compress_function(node, lines))

        elif isinstance(node, ast.ClassDef):
            if wanted_names:
                # Include class if its name is wanted OR any of its methods are
                methods = {n.name for n in ast.walk(node)
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if node.name not in wanted_names and not (set(wanted_names) & methods):
                    continue
            segments.append(_compress_class(node, lines))

        elif isinstance(node, ast.Assign):
            s = node.lineno - 1
            e = getattr(node, "end_lineno", node.lineno)
            seg = "\n".join(lines[s:e])
            if len(seg) < 200:
                segments.append(seg)

    result = "\n\n".join(s for s in segments if s.strip())
    result_lines = result.splitlines()
    if len(result_lines) > max_lines:
        result = "\n".join(result_lines[:max_lines]) + "\n    # ... (truncated)"
    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 5 — Context building (per test file)
# ═══════════════════════════════════════════════════════════════════════

def build_context_for_test(
    test_source: str,
    test_rel_path: str,
    regular_files: Dict[str, str],
    module_index: Dict[str, str],
    test_files: Dict[str, str],
    max_lines: int = MAX_CONTEXT_LINES,
) -> str:
    """Build a compressed cross-file context block for a test file.

    Includes:
    * Compressed versions of every imported regular-file module.
    * Relevant ``conftest.py`` fixtures.
    """
    imports = extract_imports(test_source)

    parts: List[str] = []
    seen: Set[str] = set()

    # ── imported regular files ──────────────────────────────────────
    for imp in imports:
        resolved = resolve_import(imp, module_index)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)

        try:
            src = Path(resolved).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Determine nice relative path for the header comment
        rel = resolved
        for reg_rel, reg_abs in regular_files.items():
            if reg_abs == resolved:
                rel = reg_rel
                break

        compressed = compress_source(src, wanted_names=imp.get("names"), max_lines=60)
        if compressed.strip():
            parts.append(f"# ── Source: {rel} ──\n{compressed}")

    # ── conftest fixtures ───────────────────────────────────────────
    test_dir = str(Path(test_rel_path).parent)
    for t_rel, t_abs in test_files.items():
        if Path(t_rel).name != "conftest.py":
            continue
        conftest_dir = str(Path(t_rel).parent)
        # Include if conftest is in the same dir or a parent directory
        if test_dir == conftest_dir or test_dir.startswith(conftest_dir + os.sep) or conftest_dir == ".":
            if t_abs not in seen:
                seen.add(t_abs)
                try:
                    src = Path(t_abs).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                compressed = compress_source(src, max_lines=40)
                if compressed.strip():
                    parts.append(f"# ── Fixtures: {t_rel} ──\n{compressed}")

    context = "\n\n".join(parts)
    ctx_lines = context.splitlines()
    if len(ctx_lines) > max_lines:
        context = "\n".join(ctx_lines[:max_lines]) + "\n# ... (context truncated)"
    return context


# ═══════════════════════════════════════════════════════════════════════
# Part 6 — QnA pair generation
# ═══════════════════════════════════════════════════════════════════════

def _find_test_functions(source: str) -> List[ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "test" in node.name.lower()
    ]


def _choose_cut_points(lines: List[str], test_start: int,
                        test_end: int) -> List[Tuple[int, str]]:
    """Return candidate ``(line_index, kind)`` cut points inside a test."""
    candidates: List[Tuple[int, str]] = []
    for i in range(test_start + 1, test_end):
        s = lines[i].strip()
        if s.startswith("assert ") or "self.assert" in s or "assert_" in s:
            candidates.append((i, "assert"))
        elif "pytest.raises" in s or "self.assertRaises" in s:
            candidates.append((i, "raises"))
        elif s.startswith("except"):
            candidates.append((i, "except"))

    if not candidates:
        mid = (test_start + test_end) // 2
        if test_start + 1 < mid < test_end:
            candidates.append((mid, "midpoint"))
    return candidates


def _extract_target(lines: List[str], cut_i: int,
                    max_lines: int = MAX_TARGET_LINES) -> Tuple[str, int]:
    """Extract the target code block starting at *cut_i*."""
    if cut_i >= len(lines):
        return "", 0

    target: List[str] = []
    first = lines[cut_i]
    base_indent = len(first) - len(first.lstrip())

    for j in range(cut_i, min(len(lines), cut_i + max_lines)):
        line = lines[j]
        if line.strip() == "":
            if target:
                break
            continue
        indent = len(line) - len(line.lstrip())
        if target and indent < base_indent:
            break
        target.append(line)
        if re.match(r"^\s*(return|raise)\b", line):
            break

    return "\n".join(target), len(target)


def generate_qna_pairs(
    test_files: Dict[str, str],
    regular_files: Dict[str, str],
    module_index: Dict[str, str],
    num_pairs: int = NUM_PAIRS,
    seed: int = SEED,
) -> List[Dict]:
    """Create QnA pairs from test files with cross-file context."""
    random.seed(seed)

    all_candidates: List[Dict] = []

    for test_rel, test_abs in test_files.items():
        if not test_rel.endswith(".py"):
            continue
        if Path(test_rel).name == "conftest.py":
            continue  # conftest is context, not a QnA source

        try:
            source = Path(test_abs).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = source.splitlines()
        if len(lines) < 10:
            continue

        test_nodes = _find_test_functions(source)
        if not test_nodes:
            continue

        # Build context once per test file (expensive)
        context = build_context_for_test(
            source, test_rel, regular_files, module_index, test_files,
        )

        for node in test_nodes:
            test_start = node.lineno - 1
            test_end = getattr(node, "end_lineno", node.lineno)
            if test_end - test_start < MIN_TEST_LINES:
                continue

            for cut_i, cut_kind in _choose_cut_points(lines, test_start, test_end):
                target, n_tgt = _extract_target(lines, cut_i)
                if n_tgt == 0 or target.strip() in ("", "pass"):
                    continue

                prefix_lines = lines[:cut_i]
                if len(prefix_lines) > MAX_PREFIX_LINES:
                    prefix_lines = prefix_lines[-MAX_PREFIX_LINES:]
                prefix = "\n".join(prefix_lines)

                all_candidates.append({
                    "test_file": test_rel,
                    "function": node.name,
                    "cut_kind": cut_kind,
                    "cut_line": cut_i + 1,
                    "context": context,
                    "prefix": prefix,
                    "target": target,
                })

    # ── sample ──
    if len(all_candidates) > num_pairs:
        random.shuffle(all_candidates)
        selected = all_candidates[:num_pairs]
    else:
        selected = all_candidates
        if len(selected) < num_pairs:
            print(f"  [warn] only {len(selected)} candidates found "
                  f"(requested {num_pairs})")

    # ── format for instruction-tuning ──
    pairs: List[Dict] = []
    for cand in selected:
        input_parts: List[str] = []
        if cand["context"].strip():
            input_parts.append(cand["context"])
        input_parts.append(f"\n# ── Test file: {cand['test_file']} ──")
        input_parts.append(cand["prefix"])
        input_text = "\n".join(input_parts)

        pairs.append({
            "instruction": (
                "Complete the next block of test code based on the "
                "repository context and test file prefix below."
            ),
            "input": input_text,
            "output": cand["target"],
            "metadata": {
                "test_file": cand["test_file"],
                "function": cand["function"],
                "cut_kind": cand["cut_kind"],
                "cut_line": cand["cut_line"],
            },
        })

    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Part 7 — LoRA training
# ═══════════════════════════════════════════════════════════════════════

def train_lora(train_data: List[Dict], val_data: List[Dict],
               output_dir: str, repo_name: str) -> str:
    """Train a repo-specific LoRA and return the adapter path."""

    # Deferred imports so the pure-Python parts above stay fast
    import torch
    import wandb
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from typing import List as TList

    print(f"\n{'=' * 60}")
    print(f"  Training repo-specific LoRA for: {repo_name}")
    print(f"  Train examples : {len(train_data)}")
    print(f"  Val examples   : {len(val_data)}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'=' * 60}\n")

    # ── quantisation ────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── tokeniser ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── model ───────────────────────────────────────────────────────
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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── datasets ────────────────────────────────────────────────────
    # Strip metadata before feeding into HF Dataset
    def _strip_meta(examples):
        return [{k: v for k, v in ex.items() if k != "metadata"}
                for ex in examples]

    train_dataset = Dataset.from_list(_strip_meta(train_data))
    val_dataset = Dataset.from_list(_strip_meta(val_data))

    # ── formatting (same template as baselines) ─────────────────────
    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            text += f"### Input:\n{example['input']}\n"
        text += f"### Response:\n{example['output']}"
        return text

    # ── data collator (label-masking, copied from baselines) ────────
    class InstructionDataCollator(DataCollatorForLanguageModeling):
        """Mask labels for everything before ``### Response:`` so the loss
        is only computed on the target completion."""

        def __init__(self, tokenizer, response_marker="### Response:", **kwargs):
            super().__init__(tokenizer=tokenizer, **kwargs)
            self.response_marker = response_marker
            self.response_marker_tokens = tokenizer.encode(
                response_marker, add_special_tokens=False,
            )

        def __call__(self, examples):
            if isinstance(examples[0], dict):
                texts = []
                for ex in examples:
                    t = f"### Instruction:\n{ex['instruction']}\n"
                    if ex.get("input"):
                        t += f"### Input:\n{ex['input']}\n"
                    t += f"### Response:\n{ex['output']}"
                    texts.append(t)
                examples = texts

            if isinstance(examples[0], str):
                batch = self.tokenizer(
                    examples, padding=True, truncation=True,
                    max_length=2048, return_tensors="pt",
                )
                labels = batch["input_ids"].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels
            else:
                batch = super().__call__(examples)

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            marker = self.response_marker_tokens

            for i in range(len(labels)):
                ids = input_ids[i].tolist()
                pos = self._find_marker(ids, marker)
                if pos > 0:
                    labels[i, :pos] = -100
            return batch

        @staticmethod
        def _find_marker(ids: list, marker: list) -> int:
            for i in range(len(ids) - len(marker) + 1):
                if ids[i : i + len(marker)] == marker:
                    return i + len(marker)
            return 0

    data_collator = InstructionDataCollator(
        tokenizer=tokenizer, response_marker="### Response:", mlm=False,
    )

    # ── wandb ───────────────────────────────────────────────────────
    safe_name = repo_name.replace("/", "_")
    wandb.init(
        project="repo-specific-lora",
        name=f"lora-{safe_name}",
        config={
            "repo": repo_name,
            "model_name": MODEL_NAME,
            "lora_r": 16,
            "lora_alpha": 32,
            "train_examples": len(train_data),
            "val_examples": len(val_data),
        },
    )

    # ── training args (tuned for small dataset) ─────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,               # more epochs for tiny dataset
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="wandb",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── trainer ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=training_args,
    )

    # ── train ───────────────────────────────────────────────────────
    print("Evaluating before training …")
    pre_eval = trainer.evaluate()
    print(f"  initial eval_loss = {pre_eval['eval_loss']:.4f}")
    wandb.log({"init_eval_loss": pre_eval["eval_loss"]})

    print("\nTraining …")
    trainer.train()

    print("\nEvaluating after training …")
    post_eval = trainer.evaluate()
    print(f"  final eval_loss = {post_eval['eval_loss']:.4f}")
    wandb.log({"final_eval_loss": post_eval["eval_loss"]})

    # ── save adapter ────────────────────────────────────────────────
    adapter_path = os.path.join(output_dir, "adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    wandb.finish()

    print(f"\n  LoRA adapter saved → {adapter_path}")
    return adapter_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate QnA pairs from a repo and train a repo-specific LoRA",
    )
    parser.add_argument("--repo", type=str, default=None,
                        help="Repository name (owner/repo)")
    parser.add_argument("--repos-root", type=str, default=PULLED_REPOS_ROOT,
                        help="Root dir of pulled repos")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT,
                        help="Output root for adapter & data")
    parser.add_argument("--num-pairs", type=int, default=NUM_PAIRS,
                        help="Number of QnA pairs to generate")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--list-repos", action="store_true",
                        help="List available repos and exit")
    parser.add_argument("--qna-only", action="store_true",
                        help="Generate QnA pairs only (skip LoRA training)")

    args = parser.parse_args()

    # ── list repos ──────────────────────────────────────────────────
    if args.list_repos:
        repos = list_available_repos(args.repos_root)
        if not repos:
            print(f"No repos found under {args.repos_root}")
        else:
            print(f"Available repos ({len(repos)}):")
            for r in repos:
                print(f"  {r}")
        return

    if args.repo is None:
        parser.error("--repo is required (use --list-repos to see choices)")

    test_dir, reg_dir = get_repo_dirs(args.repos_root, args.repo)
    if not os.path.isdir(test_dir) or not os.path.isdir(reg_dir):
        sys.exit(f"ERROR: repo '{args.repo}' not found under {args.repos_root}\n"
                 f"  Expected:\n    {test_dir}\n    {reg_dir}")

    # ── index files ─────────────────────────────────────────────────
    print(f"Repo            : {args.repo}")
    test_files = collect_python_files(test_dir)
    regular_files = collect_python_files(reg_dir)
    module_index = build_module_index(regular_files)
    print(f"Test .py files  : {len(test_files)}")
    print(f"Regular .py files: {len(regular_files)}")
    print(f"Module index     : {len(module_index)} modules")

    # ── generate QnA pairs ──────────────────────────────────────────
    print(f"\nGenerating {args.num_pairs} QnA pairs …")
    pairs = generate_qna_pairs(
        test_files, regular_files, module_index,
        num_pairs=args.num_pairs, seed=args.seed,
    )
    print(f"  Generated {len(pairs)} pairs")

    if not pairs:
        sys.exit("ERROR: no QnA pairs could be generated for this repo")

    # ── split 80 / 20 ──────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * TRAIN_RATIO))
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]
    # Ensure at least 1 val example
    if not val_data:
        val_data = [train_data.pop()]
    print(f"  Train : {len(train_data)}")
    print(f"  Val   : {len(val_data)}")

    # ── save QnA to disk ────────────────────────────────────────────
    safe_name = args.repo.replace("/", "_")
    repo_output_dir = os.path.join(args.output, safe_name)
    os.makedirs(repo_output_dir, exist_ok=True)

    train_path = os.path.join(repo_output_dir, "train.jsonl")
    val_path = os.path.join(repo_output_dir, "val.jsonl")

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  Saved train → {train_path}")
    print(f"  Saved val   → {val_path}")

    if args.qna_only:
        print("\n--qna-only: skipping LoRA training.")
        return

    # ── train LoRA ──────────────────────────────────────────────────
    adapter_path = train_lora(train_data, val_data, repo_output_dir, args.repo)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete for: {args.repo}")
    print(f"  QnA train : {train_path}")
    print(f"  QnA val   : {val_path}")
    print(f"  Adapter   : {adapter_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
