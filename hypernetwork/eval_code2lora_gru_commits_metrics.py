#!/usr/bin/env python3
"""
Commit-parquet evaluation for Code2LoRA-GRU checkpoints with task metrics.

Unlike :mod:`eval_code2lora_gru_commits.py` (loss-only), this script performs
generation and reports:

* Exact Match (EM)
* Edit Similarity
* CodeBLEU (codebleu if installed, else tokenized BLEU fallback)

It supports:
* Cross-repo (CR) and in-repo (IR) evaluation suites
* Per-repo final metrics (at the last kept commit)
* Per-commit timeline metrics (either every kept commit or a set of commit percentiles)

This is intended to mirror the scoring used by other checkpoint evaluations
under :mod:`evaluation.metrics`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.baseline_config import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS
from evaluation.metrics import (
    aggregate_metrics_with_ci,
    compute_metrics,
    format_ci,
)

from code2lora_gru import Code2LoRAGRU
from parquet_commit_dataset import (
    LAZY_QNA_SPEC_KEY,
    load_commit_sequences_from_parquet,
    materialize_lazy_qna_for_repo,
    resolve_parquet_sources,
)
from train_code2lora_gru import (
    apply_lora_hooks,
    discover_target_modules,
    get_bos_id,
    remove_lora_hooks,
    set_seed,
)
from train_code2lora_gru_commits import (
    DEFAULT_EMBED_MODEL,
    DiffEmbedder,
    get_assertions_up_to,
)


def _build_model_from_ckpt(
    ckpt_path: Path,
    base_model_name: str,
    file_embed_dim: int,
    target_modules: List[str],
    device: torch.device,
) -> Tuple[Code2LoRAGRU, Any, Dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("model_config", {})
    lcfg = cfg.get("lora_generator", {}) or {}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    for p in base_model.parameters():
        p.requires_grad = False
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    base_model.eval()

    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, target_modules,
    )

    ckpt_module_dims = lcfg.get("module_dims")
    if ckpt_module_dims:
        module_dims = {k: tuple(v) for k, v in ckpt_module_dims.items()}
    ckpt_num_layers = int(lcfg.get("num_layers", num_layers))

    rank = int(lcfg.get("rank", 16))
    scaling = float(lcfg.get("lora_scaling", 2.0))
    alpha = float(rank) * scaling

    model = Code2LoRAGRU(
        file_embed_dim=int(cfg.get("file_embed_dim", file_embed_dim)),
        gru_hidden_dim=int(cfg.get("gru_hidden_dim", 1024)),
        gru_num_layers=int(cfg.get("gru_num_layers", 1)),
        num_target_layers=ckpt_num_layers,
        module_dims=module_dims,
        lora_hidden_dim=int(lcfg.get("hidden_dim", 512)),
        lora_rank=rank,
        lora_alpha=alpha,
        lora_num_bases=int(lcfg.get("num_bases", 16)),
        lora_trunk_depth=int(lcfg.get("trunk_depth", 2) or 2),
        init_type=str(cfg.get("init_type", "zeros")),
        gru_dropout=float(cfg.get("gru_dropout", 0.0) or 0.0),
        bptt_window=int(cfg.get("bptt_window", 32) or 32),
    ).to(device=device, dtype=torch.float32)

    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.eval()
    return model, base_model, target_modules_dict


def _select_commit_positions(
    commit_indices: List[int],
    *,
    mode: str,
    percentiles: List[int],
) -> List[int]:
    if not commit_indices:
        return []
    if mode == "all":
        return list(commit_indices)
    if mode != "percentiles":
        raise ValueError(f"unknown timeline mode: {mode}")
    n = len(commit_indices)
    out: List[int] = []
    for p in percentiles:
        p2 = max(0, min(100, int(p)))
        if n == 1:
            idx = 0
        else:
            idx = int(round((p2 / 100.0) * (n - 1)))
        out.append(commit_indices[idx])
    # preserve order + uniqueness
    seen = set()
    uniq = []
    for ci in out:
        if ci not in seen:
            uniq.append(ci)
            seen.add(ci)
    return uniq


@torch.no_grad()
def _generate_one(
    base_model,
    tok,
    device: torch.device,
    input_ids: List[int],
    max_new_tokens: int,
) -> str:
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = base_model.generate(
        inp,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_ids = out[0][len(input_ids) :].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True)


def _acc_init() -> Dict[str, Any]:
    return {
        "n": 0,
        "em_sum": 0.0,
        "edit_sum": 0.0,
        "bleu_sum": 0.0,
        # raw per-pair tuples (em01, edit, bleu) for bootstrap CIs.
        "samples": [],
    }


def _acc_add(acc: Dict[str, Any], m: Dict[str, Any]) -> None:
    em01 = 1.0 if m["exact_match"] else 0.0
    edit = float(m["edit_similarity"])
    bleu = float(m["code_bleu"])
    acc["n"] += 1
    acc["em_sum"] += em01
    acc["edit_sum"] += edit
    acc["bleu_sum"] += bleu
    acc["samples"].append((em01, edit, bleu))


def _acc_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["n"] += int(src["n"])
    dst["em_sum"] += float(src["em_sum"])
    dst["edit_sum"] += float(src["edit_sum"])
    dst["bleu_sum"] += float(src["bleu_sum"])
    if src.get("samples"):
        dst.setdefault("samples", []).extend(src["samples"])


def _acc_finalize(acc: Dict[str, Any]) -> Dict[str, Any]:
    n = max(int(acc["n"]), 1)
    return {
        "n": int(acc["n"]),
        "exact_match": float(acc["em_sum"]) / n,
        "edit_similarity": float(acc["edit_sum"]) / n,
        "code_bleu": float(acc["bleu_sum"]) / n,
    }


def _score_assertions(
    assertions: List[Tuple[str, str]],
    base_model,
    tok,
    device: torch.device,
    bos_id: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Score a list of (prefix, target) pairs with whatever LoRA hooks are
    currently attached to ``base_model`` and return an accumulator.

    The caller is responsible for installing/removing LoRA hooks; this function
    only does the inference + metric accumulation.
    """
    acc = _acc_init()
    for (prefix, target) in assertions:
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        input_ids = [bos_id] + prefix_ids
        if len(input_ids) > int(max_input_tokens):
            input_ids = input_ids[-int(max_input_tokens) :]
        pred = _generate_one(
            base_model, tok, device, input_ids, max_new_tokens=max_new_tokens,
        )
        m = compute_metrics(pred, target)
        _acc_add(acc, m)
    return acc


def _eval_repo_timeline(
    *,
    repo_item: Dict[str, Any],
    code2lora_gru: Code2LoRAGRU,
    base_model,
    diff_embedder: DiffEmbedder,
    target_modules_dict: Dict,
    tok,
    device: torch.device,
    keep_splits: Optional[List[str]],
    assertion_mode: str,
    max_assertions_per_commit: int,
    max_input_tokens: int,
    max_new_tokens: int,
    timeline_mode: str,
    timeline_percentiles: List[int],
    final_mode: str,
    max_assertions_final: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate one repo. Returns (repo_final_acc, per_commit, sanity).

    The ``repo_final_acc`` is the headline number for the suite. Two modes:

    * ``final_mode == "last_lora_all_assertions"`` (default, paper protocol):
      walk the full commit sequence to obtain the final hidden state ``h_T``,
      generate a single LoRA from ``h_T``, and score **all** held-out
      assertions in the repo (filtered by ``keep_splits``). This matches the
      static-encoder evaluation used everywhere else in the paper.

    * ``final_mode == "last_commit_new_assertions"`` (legacy): score only the
      assertions newly introduced at ``commit_indices[-1]`` (mode='new'). This
      reproduces the previous behavior and is only useful for the timeline
      figure.

    ``per_commit`` always contains the timeline rows requested by the caller
    (mode/percentiles), independent of ``final_mode``.

    ``sanity`` reports per-split assertion counts to make ``n=0`` failures
    obvious from the log.
    """
    # Materialize QnAs for this repo only if they were deferred.
    if repo_item.get(LAZY_QNA_SPEC_KEY) is not None:
        materialize_lazy_qna_for_repo(repo_item)

    splits_by_commit = repo_item.get("assertion_splits")
    assertions_by_commit = repo_item.get("assertions_by_commit") or {}

    sanity: Dict[str, Any] = {
        "n_commits": 0,
        "n_assertions_total": 0,
        "n_assertions_kept": 0,
        "n_assertions_per_split": defaultdict(int),
        "final_lora_assertions": 0,
    }

    # Commit sequence
    commit_indices = repo_item["commit_indices"]
    commit_diffs = repo_item["commit_diffs"]
    sanity["n_commits"] = len(commit_indices)
    if not commit_indices:
        return _acc_init(), [], sanity

    # Per-split assertion counts (sanity).
    for ci, pairs in assertions_by_commit.items():
        sanity["n_assertions_total"] += len(pairs)
        splits_here = (splits_by_commit or {}).get(ci, [])
        if len(splits_here) == len(pairs):
            for s in splits_here:
                sanity["n_assertions_per_split"][s] += 1
        else:
            sanity["n_assertions_per_split"]["__unknown__"] += len(pairs)

    bos_id = get_bos_id(tok)
    positions = _select_commit_positions(
        commit_indices,
        mode=timeline_mode,
        percentiles=timeline_percentiles,
    )
    pos_set = set(positions)
    last_ci = commit_indices[-1]

    # GRU hidden state over commits
    h = code2lora_gru.compute_h0(batch_size=1, device=device, dtype=torch.float32)

    per_commit: List[Dict[str, Any]] = []

    for k in range(len(commit_indices)):
        ci = commit_indices[k]
        diff_text = commit_diffs[k]
        diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        h = code2lora_gru.encode_repository_commit(diff_emb, h)

        # Skip cheaply unless this commit is on the timeline grid.
        if ci not in pos_set:
            h = h.detach()
            continue

        assertions = get_assertions_up_to(
            assertions_by_commit,
            ci,
            splits_by_commit=splits_by_commit,
            keep_splits=keep_splits,
            mode=assertion_mode,
        )
        if not assertions:
            h = h.detach()
            continue

        if max_assertions_per_commit > 0 and len(assertions) > max_assertions_per_commit:
            assertions = rng.sample(assertions, max_assertions_per_commit)

        lora_params = code2lora_gru.generate_lora_from_h(h)
        if lora_params is None:
            h = h.detach()
            continue

        handles = apply_lora_hooks(
            target_modules_dict,
            lora_params,
            code2lora_gru.lora_generator.lora_scaling,
        )
        acc = _score_assertions(
            assertions, base_model, tok, device, bos_id,
            max_input_tokens, max_new_tokens,
        )
        remove_lora_hooks(handles)

        per_commit.append({
            "commit_index": int(ci),
            **_acc_finalize(acc),
        })

        h = h.detach()

    # ------------------------------------------------------------------
    # Headline FINAL eval. Two modes:
    #   * last_lora_all_assertions: freeze LoRA at h_T, score every held-out
    #     assertion in the repo (paper protocol).
    #   * last_commit_new_assertions: legacy/buggy reproducer (only the new
    #     assertions at the very last commit). Kept for back-compat, NOT the
    #     headline number.
    # ------------------------------------------------------------------
    repo_final_acc = _acc_init()

    if final_mode == "last_lora_all_assertions":
        final_assertions = get_assertions_up_to(
            assertions_by_commit,
            last_ci,
            splits_by_commit=splits_by_commit,
            keep_splits=keep_splits,
            mode="cumulative",
        )
        sanity["final_lora_assertions"] = len(final_assertions)
        if final_assertions:
            if max_assertions_final > 0 and len(final_assertions) > max_assertions_final:
                final_assertions = rng.sample(final_assertions, max_assertions_final)
                sanity["final_lora_assertions_capped"] = max_assertions_final
            lora_params = code2lora_gru.generate_lora_from_h(h)
            if lora_params is not None:
                handles = apply_lora_hooks(
                    target_modules_dict,
                    lora_params,
                    code2lora_gru.lora_generator.lora_scaling,
                )
                repo_final_acc = _score_assertions(
                    final_assertions, base_model, tok, device, bos_id,
                    max_input_tokens, max_new_tokens,
                )
                remove_lora_hooks(handles)

    elif final_mode == "last_commit_new_assertions":
        last_assertions = get_assertions_up_to(
            assertions_by_commit,
            last_ci,
            splits_by_commit=splits_by_commit,
            keep_splits=keep_splits,
            mode="new",
        )
        sanity["final_lora_assertions"] = len(last_assertions)
        if last_assertions:
            if max_assertions_per_commit > 0 and len(last_assertions) > max_assertions_per_commit:
                last_assertions = rng.sample(last_assertions, max_assertions_per_commit)
            lora_params = code2lora_gru.generate_lora_from_h(h)
            if lora_params is not None:
                handles = apply_lora_hooks(
                    target_modules_dict,
                    lora_params,
                    code2lora_gru.lora_generator.lora_scaling,
                )
                repo_final_acc = _score_assertions(
                    last_assertions, base_model, tok, device, bos_id,
                    max_input_tokens, max_new_tokens,
                )
                remove_lora_hooks(handles)
    else:
        raise ValueError(f"unknown final_mode: {final_mode}")

    sanity["n_assertions_kept"] = int(repo_final_acc["n"])
    sanity["n_assertions_per_split"] = dict(sanity["n_assertions_per_split"])

    return repo_final_acc, per_commit, sanity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--parquet-dir", type=str, default=os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "commit_parquet_hf",
    ))
    ap.add_argument("--parquet-prefer", type=str, default="hf",
                    choices=["auto", "concat", "shards", "hf"])
    ap.add_argument("--commits-parquet", type=str, default=None)
    ap.add_argument("--qna-parquet", type=str, default=None)

    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)

    ap.add_argument("--max-assertions-per-commit", type=int, default=64,
                    help="Cap # assertions evaluated at each TIMELINE commit (0 = all).")
    ap.add_argument(
        "--max-assertions-final", type=int, default=0,
        help=(
            "Cap # held-out assertions scored at the FINAL headline pass. "
            "0 = all (recommended for paper-grade numbers); use a small value "
            "only for fast smoke tests."
        ),
    )
    ap.add_argument("--assertion-mode", type=str, default="new",
                    choices=["cumulative", "new"],
                    help="Assertion-mode used for the per-commit TIMELINE rows only.")
    ap.add_argument(
        "--final-mode", type=str, default="last_lora_all_assertions",
        choices=["last_lora_all_assertions", "last_commit_new_assertions"],
        help=(
            "How to compute the per-suite FINAL number. The default freezes "
            "the LoRA at h_T and scores every held-out assertion in the repo "
            "(paper protocol). 'last_commit_new_assertions' reproduces the "
            "previous (buggy) behavior and is kept for back-compat only."
        ),
    )
    ap.add_argument(
        "--allow-empty-suite", action="store_true",
        help="Do not abort when a suite ends up with FINAL n=0 (debug only).",
    )
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument(
        "--suites", nargs="+",
        default=["in_repo_val", "in_repo_test", "cross_repo_cr_val", "cross_repo_cr_test"],
    )

    ap.add_argument("--timeline-mode", type=str, default="percentiles",
                    choices=["percentiles", "all"])
    ap.add_argument("--timeline-percentiles", type=str, default="0,25,50,75,100",
                    help="Comma-separated list, used when timeline-mode=percentiles.")
    ap.add_argument(
        "--bootstrap", type=int, default=0,
        help="If >0, compute bootstrap 95%% CIs over (em, edit, codebleu) "
             "with this many resamples per suite.",
    )

    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--target-modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ])
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda:0")

    timeline_percentiles = [
        int(x.strip()) for x in str(args.timeline_percentiles).split(",") if x.strip()
    ]

    sources = resolve_parquet_sources(
        parquet_dir=args.parquet_dir,
        commits_path=args.commits_parquet,
        qna_path=args.qna_parquet,
        prefer=args.parquet_prefer,
    )

    # Diff embedder
    etok = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
    emod = AutoModel.from_pretrained(
        args.embed_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    for p in emod.parameters():
        p.requires_grad = False
    diff_embedder = DiffEmbedder(model=emod, tokenizer=etok, device=str(device))
    file_embed_dim = diff_embedder.embed_dim

    # Tokenizer + base LLM + Code2LoRA-GRU
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    code2lora_gru, base_model, target_modules_dict = _build_model_from_ckpt(
        Path(args.checkpoint), args.model_name, file_embed_dim, args.target_modules, device,
    )

    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "parquet_dir": str(args.parquet_dir),
        "suites": list(args.suites),
        "timeline_mode": args.timeline_mode,
        "timeline_percentiles": timeline_percentiles,
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "max_assertions_per_commit": int(args.max_assertions_per_commit),
        "assertion_mode": args.assertion_mode,
        "limit_repos": args.limit_repos,
        "seed": int(args.seed),
    }

    for suite in args.suites:
        if suite == "in_repo_val":
            cross = ["train"]
            keep = ["val"]
        elif suite == "in_repo_test":
            cross = ["train"]
            keep = ["test"]
        elif suite == "cross_repo_cr_val":
            cross = ["cr_val"]
            keep = None
        elif suite == "cross_repo_cr_test":
            cross = ["cr_test"]
            keep = None
        else:
            print(f"  [warn] unknown suite {suite}, skipping")
            continue

        data = load_commit_sequences_from_parquet(
            sources,
            cross_repo_splits=cross,
            in_repo_splits=None,
            limit_repos=args.limit_repos,
            defer_qna_materialization=True,
        )
        if not data:
            print(f"  [{suite}] no data; skipping.")
            results[suite] = None
            continue

        print(f"\n=== Suite: {suite} (repos={len(data)}, keep={keep}) ===", flush=True)

        suite_acc = _acc_init()
        timeline_acc: Dict[int, Dict[str, Any]] = defaultdict(_acc_init)
        per_repo: Dict[str, Any] = {}
        suite_sanity = {
            "n_repos": len(data),
            "n_assertions_total": 0,
            "n_assertions_per_split": defaultdict(int),
            "final_lora_assertions": 0,
            "repos_with_zero_final": 0,
        }

        for ri, repo_item in enumerate(data):
            repo_id = repo_item.get("repo_id", f"repo_{ri}")
            repo_final_acc, per_commit, sanity = _eval_repo_timeline(
                repo_item=repo_item,
                code2lora_gru=code2lora_gru,
                base_model=base_model,
                diff_embedder=diff_embedder,
                target_modules_dict=target_modules_dict,
                tok=tok,
                device=device,
                keep_splits=keep,
                assertion_mode=args.assertion_mode,
                max_assertions_per_commit=int(args.max_assertions_per_commit),
                max_input_tokens=int(args.max_input_tokens),
                max_new_tokens=int(args.max_new_tokens),
                timeline_mode=args.timeline_mode,
                timeline_percentiles=timeline_percentiles,
                final_mode=args.final_mode,
                max_assertions_final=int(args.max_assertions_final),
                rng=rng,
            )

            _acc_merge(suite_acc, repo_final_acc)
            suite_sanity["n_assertions_total"] += int(sanity["n_assertions_total"])
            for s, n in (sanity.get("n_assertions_per_split") or {}).items():
                suite_sanity["n_assertions_per_split"][s] += int(n)
            suite_sanity["final_lora_assertions"] += int(sanity["final_lora_assertions"])
            if int(repo_final_acc["n"]) == 0:
                suite_sanity["repos_with_zero_final"] += 1

            for row in per_commit:
                ci = int(row["commit_index"])
                # re-expand row into accumulator for proper weighting
                tmp = _acc_init()
                tmp["n"] = int(row["n"])
                tmp["em_sum"] = float(row["exact_match"]) * int(row["n"])
                tmp["edit_sum"] = float(row["edit_similarity"]) * int(row["n"])
                tmp["bleu_sum"] = float(row["code_bleu"]) * int(row["n"])
                _acc_merge(timeline_acc[ci], tmp)

            per_repo[repo_id] = {
                "final": _acc_finalize(repo_final_acc),
                "timeline": per_commit,
                "sanity": {
                    "n_commits": sanity["n_commits"],
                    "n_assertions_total": sanity["n_assertions_total"],
                    "n_assertions_per_split": sanity["n_assertions_per_split"],
                    "final_lora_assertions": sanity["final_lora_assertions"],
                },
            }

            # Free large per-repo QnA strings before moving on.
            repo_item.pop("assertions_by_commit", None)
            repo_item.pop("assertion_splits", None)
            repo_item.pop(LAZY_QNA_SPEC_KEY, None)

            if (ri + 1) % 5 == 0:
                print(f"  processed {ri+1}/{len(data)} repos", flush=True)

        # Sanity printout: per-split counts so n=0 is never silent.
        suite_sanity["n_assertions_per_split"] = dict(
            suite_sanity["n_assertions_per_split"]
        )
        print(
            "  SANITY: total_assertions={:,} per_split={} final_eval_pool={:,} "
            "repos_with_zero_final={}/{}".format(
                suite_sanity["n_assertions_total"],
                suite_sanity["n_assertions_per_split"],
                suite_sanity["final_lora_assertions"],
                suite_sanity["repos_with_zero_final"],
                suite_sanity["n_repos"],
            ),
            flush=True,
        )

        suite_out = {
            "final": _acc_finalize(suite_acc),
            "timeline": [
                {"commit_index": int(ci), **_acc_finalize(acc)}
                for ci, acc in sorted(timeline_acc.items(), key=lambda x: x[0])
            ],
            "per_repo": per_repo,
            "sanity": suite_sanity,
        }

        if args.bootstrap > 0 and suite_acc.get("samples"):
            suite_out["final_ci"] = aggregate_metrics_with_ci(
                suite_acc["samples"],
                n_resamples=int(args.bootstrap),
                seed=int(args.seed),
            )

        if int(suite_out["final"]["n"]) == 0 and not args.allow_empty_suite:
            raise RuntimeError(
                f"Suite '{suite}' produced FINAL n=0. This is almost always a "
                f"misconfiguration: keep_splits={keep}, final_mode={args.final_mode}. "
                f"Inspect the SANITY line printed above. Re-run with "
                f"--allow-empty-suite to override."
            )

        print(
            f"  FINAL: n={suite_out['final']['n']:,} "
            f"em={suite_out['final']['exact_match']:.4f} "
            f"edit={suite_out['final']['edit_similarity']:.4f} "
            f"codebleu={suite_out['final']['code_bleu']:.4f}",
            flush=True,
        )
        if "final_ci" in suite_out:
            ci = suite_out["final_ci"]
            print(
                "  CI95: em={} edit={} codebleu={}".format(
                    format_ci(ci["exact_match"], pct=True),
                    format_ci(ci["edit_similarity"]),
                    format_ci(ci["code_bleu"]),
                ),
                flush=True,
            )

        results[suite] = suite_out

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults JSON -> {out}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

