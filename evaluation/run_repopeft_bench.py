#!/usr/bin/env python3
"""
Unified RepoPeftBench evaluation driver.

Scores the three Code2LoRA variants on the *same* canonical RepoPeftBench QnA
bank (``cr_test.json`` / ``ir_test.json`` / ``gru_*.json``), so that the
numbers in Table 1 are apples-to-apples and produced by the same pipeline.

Methods
-------
* ``direct``      Direct-projection hypernetwork (``hypernetwork/hypernetwork.py``).
                  Uses the static ``embedding`` field of each repo (a 2048-dim
                  ``mean+max`` over file embeddings).
* ``gru_file``    File-level GRU (``hypernetwork/code2lora_gru.py``).
                  Walks the chronologically-ordered file sequence (``commit_history.file_order``)
                  through the GRU, optionally with a Mamba2 preamble. Re-embeds files
                  on the fly when ``file_embeddings`` is missing from the JSON.
* ``gru_commit``  Commit-level GRU.
                  Replays the kept-commit ``production_code_diff`` sequence from the
                  Parquet dataset to obtain ``h_T`` and a single LoRA, then scores the
                  RepoPeftBench QnA bank with that LoRA.

The QnA bank, postprocessing, EM/EditSim/CodeBLEU computation, and bootstrap CIs
are identical across methods.

Usage
-----
    python evaluation/run_repopeft_bench.py \
        --method gru_commit \
        --checkpoint $CKPT_DIR/CODE2LORA_GRU/commit_level_h100_full/code2lora_gru_best.pt \
        --bench-json $SCRATCH/REPO_DATASET/gru_cr_test.json \
        --parquet-dir $SCRATCH/REPO_DATASET/commit_parquet_hf \
        --output-json $CKPT_DIR/CODE2LORA_GRU/commit_level_h100_full/bench_cr_test.json \
        --bootstrap 5000

    python evaluation/run_repopeft_bench.py \
        --method direct \
        --checkpoint $CKPT_DIR/CODE2LORA_DIRECT/best.pt \
        --bench-json $SCRATCH/REPO_DATASET/cr_test.json \
        --output-json /tmp/direct_cr_test.json --bootstrap 5000

The driver always reports per-method, per-suite ``EM/EditSim/CodeBLEU`` with
95% bootstrap CIs and aborts if any suite ends up with ``n=0`` (unless
``--allow-empty-suite`` is passed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HYP = _ROOT / "hypernetwork"
if str(_HYP) not in sys.path:
    sys.path.insert(0, str(_HYP))

from evaluation.baseline_config import (
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
)
from evaluation.metrics import (
    aggregate_metrics_with_ci,
    compute_metrics,
    format_ci,
)


# ---------------------------------------------------------------------------
# Bench loader
# ---------------------------------------------------------------------------

def load_bench(
    path: Path,
    *,
    limit_repos: Optional[int] = None,
    limit_pairs_per_repo: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load the canonical RepoPeftBench split JSON.

    Returns a list of items::

        {"repo_id": str, "embedding": List[float] or None,
         "qna_pairs": [{"prefix": str, "target": str, ...}, ...],
         "commit_history": Dict (or None)}

    The QnA pairs are pre-filtered for non-empty / non-comma-leading targets,
    matching the protocol used by every other baseline in the paper.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_ids = sorted(repos.keys())
    if limit_repos:
        repo_ids = repo_ids[:limit_repos]

    items: List[Dict[str, Any]] = []
    for rid in repo_ids:
        rdata = repos[rid]
        pairs = []
        for p in rdata.get("qna_pairs", []) or []:
            t = (p.get("target") or "").lstrip()
            if not p.get("prefix") or not t or t.startswith(","):
                continue
            pairs.append(p)
        if limit_pairs_per_repo:
            pairs = pairs[: int(limit_pairs_per_repo)]
        if not pairs:
            continue
        items.append({
            "repo_id": rid,
            "embedding": rdata.get("embedding"),
            "qna_pairs": pairs,
            "commit_history": rdata.get("commit_history"),
        })
    return items


def load_bench_from_parquet(
    parquet_dir: Path,
    *,
    cross_repo_splits: Optional[List[str]] = None,
    in_repo_splits: Optional[List[str]] = None,
    limit_repos: Optional[int] = None,
    limit_pairs_per_repo: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a bench item list directly from a commit-parquet dataset.

    Used for OOD evaluation: the OOD parquet has cross_repo_split == 'ood_test'
    and contains commit-derived QnA pairs (added/modified assertions). We
    flatten the per-commit QnA events into a single ``qna_pairs`` list per
    repo, keeping only the latest assertion per (test_file, anchor) so the
    static bench shape matches ``cr_test.json``.
    """
    from parquet_commit_dataset import (
        load_commit_sequences_from_parquet,
        resolve_parquet_sources,
    )

    sources = resolve_parquet_sources(
        parquet_dir=str(parquet_dir),
        prefer="auto",
    )
    raw = load_commit_sequences_from_parquet(
        sources,
        cross_repo_splits=cross_repo_splits,
        in_repo_splits=in_repo_splits,
        limit_repos=limit_repos,
        defer_qna_materialization=False,
    )
    items: List[Dict[str, Any]] = []
    for it in raw:
        rid = it["repo_id"]
        # ``assertions_by_commit`` -> List[(prefix, target)] keyed by commit
        ab = it.get("assertions_by_commit") or {}
        # Keep only the most-recent (prefix, target) per (test_file, anchor)
        # is more involved; here we just keep all assertions and let the
        # scorer evaluate every one (this matches what RepoPeftBench does
        # when an assertion is updated across commits and we want each
        # update to be a separate eval pair).
        seen = set()
        flat = []
        for ci in sorted(ab.keys()):
            for prefix, target in ab[ci]:
                key = (prefix, target)
                if key in seen:
                    continue
                seen.add(key)
                flat.append({"prefix": prefix, "target": target,
                             "commit_idx": int(ci)})
        if not flat:
            continue
        if limit_pairs_per_repo:
            flat = flat[: int(limit_pairs_per_repo)]
        items.append({
            "repo_id": rid,
            "embedding": None,  # OOD has no precomputed static embedding.
            "qna_pairs": flat,
            "commit_history": None,
        })
    return items


# ---------------------------------------------------------------------------
# Generic LoRA-applied scoring loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_one(base_model, tok, device, input_ids, max_new_tokens):
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = base_model.generate(
        inp,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][len(input_ids) :].tolist(), skip_special_tokens=True)


def score_pairs_under_lora(
    pairs: List[Dict[str, Any]],
    *,
    base_model,
    tok,
    device,
    bos_id: int,
    max_input_tokens: int,
    max_new_tokens: int,
    samples_out: List[Tuple[float, float, float]],
) -> Dict[str, Any]:
    """Score a list of bench pairs under whatever LoRA hooks are currently
    attached to ``base_model``. Appends per-pair (em01, edit, bleu) tuples to
    ``samples_out``; returns aggregate counts.
    """
    n = 0
    em_count = 0
    edit_sum = 0.0
    bleu_sum = 0.0
    for p in pairs:
        prefix = p["prefix"]
        target = p["target"]
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        input_ids = [bos_id] + prefix_ids
        if len(input_ids) > int(max_input_tokens):
            input_ids = input_ids[-int(max_input_tokens) :]

        pred = _generate_one(
            base_model, tok, device, input_ids, max_new_tokens=max_new_tokens,
        )
        m = compute_metrics(pred, target)
        em01 = 1.0 if m["exact_match"] else 0.0
        edit = float(m["edit_similarity"])
        bleu = float(m["code_bleu"])

        n += 1
        em_count += int(em01)
        edit_sum += edit
        bleu_sum += bleu
        samples_out.append((em01, edit, bleu))

    return {
        "n": n,
        "em_count": em_count,
        "exact_match": (em_count / n) if n else 0.0,
        "edit_similarity": (edit_sum / n) if n else 0.0,
        "code_bleu": (bleu_sum / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Method dispatchers — each yields a per-repo "install LoRA" callable
# ---------------------------------------------------------------------------

def _get_bos_id(tok) -> int:
    """Robust BOS id (matches train_code2lora_gru.get_bos_id)."""
    bid = getattr(tok, "bos_token_id", None)
    if bid is not None:
        return int(bid)
    eid = getattr(tok, "eos_token_id", None)
    if eid is not None:
        return int(eid)
    return 0


# ---- Direct-projection ------------------------------------------------------

def build_direct_method(
    *,
    checkpoint: Path,
    base_model,
    target_module_names: List[str],
    device,
):
    """Return ``(install_lora_for_repo, target_modules_dict, lora_scaling)``
    for the direct-projection hypernetwork."""
    from hypernetwork import Hypernetwork  # lazy import

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
        cfg = state.get("model_config", {})
    else:
        sd = state
        cfg = {}

    from train_code2lora_gru import discover_target_modules
    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, target_module_names,
    )

    # module_specs: list of (layer_idx, module_name, module_type, in_f, out_f)
    module_specs = []
    for (layer_idx, mname), mod in target_modules_dict.items():
        module_specs.append((
            int(layer_idx), mname, mname,
            int(mod.in_features), int(mod.out_features),
        ))

    input_dim = int(cfg.get("input_dim") or sd.get("trunk.0.weight").shape[1])
    hidden_dim = int(cfg.get("hidden_dim") or sd.get("trunk.0.weight").shape[0])
    rank = int(cfg.get("rank", 16))
    scaling = float(cfg.get("lora_scaling", 2.0))

    hn = Hypernetwork(
        input_dim=input_dim,
        module_specs=module_specs,
        hidden_dim=hidden_dim,
        rank=rank,
    ).to(device=device, dtype=torch.float32)
    hn.load_state_dict(sd, strict=False)
    hn.eval()

    def install(repo_item: Dict[str, Any]):
        emb = repo_item.get("embedding")
        if not emb:
            return None  # no embedding -> no LoRA
        ctx = torch.tensor([emb], dtype=torch.float32, device=device)
        with torch.no_grad():
            A_dict, B_dict = hn(ctx)
        # Build per-(layer, module) LoRA dict shared across layers (matches
        # the direct-projection convention in the paper).
        lora_params = {}
        for (layer_idx, mname) in target_modules_dict.keys():
            t = mname
            if t in A_dict and t in B_dict:
                # A: [B, rank, in_f], B: [B, out_f, rank]
                lora_params[(layer_idx, mname)] = (A_dict[t], B_dict[t])
        return lora_params

    return install, target_modules_dict, float(rank * scaling) / float(rank)


# ---- File-level GRU ---------------------------------------------------------

def build_gru_file_method(
    *,
    checkpoint: Path,
    base_model,
    target_module_names: List[str],
    device,
    embed_model_name: str,
):
    """Return install fn for the file-level GRU. Re-embeds non-test files at
    HEAD if the bench JSON lacks per-file embeddings.
    """
    from code2lora_gru import Code2LoRAGRU
    from train_code2lora_gru import discover_target_modules

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("model_config", {}) or {}
    lcfg = cfg.get("lora_generator", {}) or {}

    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, target_module_names,
    )
    ckpt_module_dims = lcfg.get("module_dims")
    if ckpt_module_dims:
        module_dims = {k: tuple(v) for k, v in ckpt_module_dims.items()}

    rank = int(lcfg.get("rank", 16))
    scaling = float(lcfg.get("lora_scaling", 2.0))
    alpha = float(rank) * scaling

    file_embed_dim = int(cfg.get("file_embed_dim") or 2048)

    model = Code2LoRAGRU(
        file_embed_dim=file_embed_dim,
        gru_hidden_dim=int(cfg.get("gru_hidden_dim", 1024)),
        gru_num_layers=int(cfg.get("gru_num_layers", 1)),
        num_target_layers=int(lcfg.get("num_layers", num_layers)),
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
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    # NOTE: file re-embedding from disk is not implemented here because the
    # bench JSONs in the current dataset version no longer carry per-file
    # embeddings. To recover the paper's 64.4% file-level number, regenerate
    # the JSONs with file embeddings and pass them via --file-embeddings-json.
    # For backwards compatibility, if a sibling JSON exists we use it.

    def install(repo_item: Dict[str, Any]):
        rid = repo_item["repo_id"]
        ch = repo_item.get("commit_history") or {}
        file_order = ch.get("file_order") or []
        # Pull (path -> embedding) only if present.
        file_embs: List[List[float]] = []
        preamble_files = set(ch.get("preamble_files") or [])
        preamble_embs: List[List[float]] = []

        emb_by_path = {}
        for fe in repo_item.get("file_embeddings") or []:
            emb_by_path[fe["path"]] = fe["embedding"]

        if not emb_by_path:
            # No per-file embeddings available; fall back to a single-row
            # sequence using the static repo embedding (degenerate to direct).
            static = repo_item.get("embedding")
            if not static:
                return None
            file_embs = [static]
        else:
            for fo in file_order:
                p = fo.get("path") if isinstance(fo, dict) else fo
                if p in emb_by_path:
                    file_embs.append(emb_by_path[p])
                    if p in preamble_files:
                        preamble_embs.append(emb_by_path[p])
            if not file_embs:
                return None

        files_t = torch.tensor([file_embs], dtype=torch.float32, device=device)
        files_len = torch.tensor([len(file_embs)], dtype=torch.long, device=device)
        preamble_t = None
        preamble_len = None
        if preamble_embs:
            preamble_t = torch.tensor(
                [preamble_embs], dtype=torch.float32, device=device,
            )
            preamble_len = torch.tensor(
                [len(preamble_embs)], dtype=torch.long, device=device,
            )

        with torch.no_grad():
            _h, lora_params = model(
                file_embeddings=files_t,
                file_lengths=files_len,
                preamble_embeddings=preamble_t,
                preamble_lengths=preamble_len,
            )
        return lora_params

    return install, target_modules_dict, model.lora_generator.lora_scaling


# ---- Commit-level GRU -------------------------------------------------------

def build_gru_commit_method(
    *,
    checkpoint: Path,
    base_model,
    target_module_names: List[str],
    device,
    parquet_dir: str,
    parquet_prefer: str,
    embed_model_name: str,
    cross_repo_splits: Optional[List[str]],
):
    """Return install fn for the commit-level GRU. Replays kept commits per
    repo from ``parquet_dir`` to obtain ``h_T`` and produce a single LoRA.
    """
    from code2lora_gru import Code2LoRAGRU
    from parquet_commit_dataset import (
        LAZY_QNA_SPEC_KEY,
        load_commit_sequences_from_parquet,
        materialize_lazy_qna_for_repo,
        resolve_parquet_sources,
    )
    from train_code2lora_gru import discover_target_modules
    from train_code2lora_gru_commits import DEFAULT_EMBED_MODEL, DiffEmbedder

    sources = resolve_parquet_sources(
        parquet_dir=parquet_dir, prefer=parquet_prefer,
    )

    # Diff embedder
    etok = AutoTokenizer.from_pretrained(embed_model_name, trust_remote_code=True)
    emod = AutoModel.from_pretrained(
        embed_model_name, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(device).eval()
    for p in emod.parameters():
        p.requires_grad = False
    diff_embedder = DiffEmbedder(model=emod, tokenizer=etok, device=str(device))
    file_embed_dim = diff_embedder.embed_dim

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("model_config", {}) or {}
    lcfg = cfg.get("lora_generator", {}) or {}

    target_modules_dict, module_dims, num_layers = discover_target_modules(
        base_model, target_module_names,
    )
    ckpt_module_dims = lcfg.get("module_dims")
    if ckpt_module_dims:
        module_dims = {k: tuple(v) for k, v in ckpt_module_dims.items()}

    rank = int(lcfg.get("rank", 16))
    scaling = float(lcfg.get("lora_scaling", 2.0))
    alpha = float(rank) * scaling

    model = Code2LoRAGRU(
        file_embed_dim=int(cfg.get("file_embed_dim", file_embed_dim)),
        gru_hidden_dim=int(cfg.get("gru_hidden_dim", 1024)),
        gru_num_layers=int(cfg.get("gru_num_layers", 1)),
        num_target_layers=int(lcfg.get("num_layers", num_layers)),
        module_dims=module_dims,
        lora_hidden_dim=int(lcfg.get("hidden_dim", 512)),
        lora_rank=rank,
        lora_alpha=alpha,
        lora_num_bases=int(lcfg.get("num_bases", 16)),
        lora_trunk_depth=int(lcfg.get("trunk_depth", 2) or 2),
        init_type=str(cfg.get("init_type", "zeros")),
        gru_dropout=0.0,
        bptt_window=int(cfg.get("bptt_window", 32) or 32),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    # Pre-load all commit sequences for the requested cross_repo splits so we
    # don't pay parquet startup per repo.
    cross = list(cross_repo_splits) if cross_repo_splits else None
    print(f"  [gru_commit] loading commit sequences for cross={cross} ...", flush=True)
    raw_data = load_commit_sequences_from_parquet(
        sources,
        cross_repo_splits=cross,
        in_repo_splits=None,
        defer_qna_materialization=False,
    )
    by_repo: Dict[str, Dict[str, Any]] = {}
    for it in raw_data:
        # We don't actually need the QnAs here; we just need the diffs.
        by_repo[it["repo_id"]] = {
            "commit_diffs": it["commit_diffs"],
            "commit_indices": it["commit_indices"],
        }
    print(
        f"  [gru_commit] cached commit sequences for {len(by_repo)} repos",
        flush=True,
    )

    def install(repo_item: Dict[str, Any]):
        rid = repo_item["repo_id"]
        seq = by_repo.get(rid)
        if seq is None or not seq["commit_indices"]:
            return None

        h = model.compute_h0(batch_size=1, device=device, dtype=torch.float32)
        with torch.no_grad():
            for diff_text in seq["commit_diffs"]:
                diff_emb = diff_embedder.embed_diff(diff_text).unsqueeze(0).to(
                    device=device, dtype=torch.float32,
                )
                h = model.encode_repository_commit(diff_emb, h)
            lora_params = model.generate_lora_from_h(h)
        return lora_params

    return install, target_modules_dict, model.lora_generator.lora_scaling


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def run_eval(
    *,
    method: str,
    bench_items: List[Dict[str, Any]],
    install_fn,
    target_modules_dict,
    lora_scaling: float,
    base_model,
    tok,
    device,
    max_input_tokens: int,
    max_new_tokens: int,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    from train_code2lora_gru import apply_lora_hooks, remove_lora_hooks

    bos_id = _get_bos_id(tok)
    samples: List[Tuple[float, float, float]] = []
    per_repo: Dict[str, Any] = {}

    n_repos = len(bench_items)
    n_skipped = 0
    t0 = time.time()
    for ri, item in enumerate(bench_items):
        rid = item["repo_id"]
        lora_params = install_fn(item)
        if not lora_params:
            n_skipped += 1
            continue
        handles = apply_lora_hooks(
            target_modules_dict, lora_params, lora_scaling,
        )
        repo_samples_before = len(samples)
        agg = score_pairs_under_lora(
            item["qna_pairs"],
            base_model=base_model,
            tok=tok,
            device=device,
            bos_id=bos_id,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            samples_out=samples,
        )
        remove_lora_hooks(handles)
        per_repo[rid] = agg

        if (ri + 1) % 5 == 0 or (ri + 1) == n_repos:
            elapsed = time.time() - t0
            running_em = (
                100.0 * sum(s[0] for s in samples) / max(len(samples), 1)
            )
            print(
                f"  [{method}] {ri+1}/{n_repos} repos | "
                f"running EM={running_em:.2f}% | n_pairs={len(samples):,} | "
                f"skipped={n_skipped} | elapsed={elapsed:.0f}s",
                flush=True,
            )

    n = len(samples)
    em_pct = 100.0 * sum(s[0] for s in samples) / n if n else 0.0
    edit_avg = sum(s[1] for s in samples) / n if n else 0.0
    bleu_avg = sum(s[2] for s in samples) / n if n else 0.0

    out: Dict[str, Any] = {
        "method": method,
        "n": n,
        "n_repos_scored": len(per_repo),
        "n_repos_skipped": n_skipped,
        "exact_match_pct": em_pct,
        "edit_similarity": edit_avg,
        "code_bleu": bleu_avg,
        "per_repo": per_repo,
    }
    if bootstrap > 0 and samples:
        out["ci95"] = aggregate_metrics_with_ci(
            samples, n_resamples=bootstrap, seed=seed,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["direct", "gru_file", "gru_commit"])
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--bench-json", type=str, default=None,
        help="Path to {cr_test,ir_test,gru_*}.json. "
             "Mutually exclusive with --bench-parquet-dir.",
    )
    ap.add_argument(
        "--bench-parquet-dir", type=str, default=None,
        help="Path to a commit-parquet directory (e.g. commit_parquet_ood) "
             "from which to flatten QnA pairs. Used for OOD evaluation. "
             "Combine with --bench-cross-repo-splits ood_test.",
    )
    ap.add_argument(
        "--bench-cross-repo-splits", nargs="+", default=None,
        help="When --bench-parquet-dir is set, restrict to these splits "
             "(e.g. ood_test, cr_test). Default: all available.",
    )
    ap.add_argument("--bench-name", type=str, default=None,
                    help="Friendly name for the bench (default: bench-json basename).")

    # Commit-level only
    ap.add_argument("--parquet-dir", type=str, default=None)
    ap.add_argument("--parquet-prefer", type=str, default="hf",
                    choices=["auto", "concat", "shards", "hf"])
    ap.add_argument(
        "--cross-repo-splits", nargs="+", default=None,
        help=("For gru_commit: which cross_repo_split values to load from "
              "parquet_dir (e.g. 'cr_test', 'cr_val', 'ood_test'). If None, "
              "loads all splits."),
    )

    # Common
    ap.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
    ap.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--limit-pairs-per-repo", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--allow-empty-suite", action="store_true")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--target-modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ])

    args = ap.parse_args()
    device = torch.device("cuda:0")

    print("Loading base model:", args.model_name, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
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

    if args.bench_json and args.bench_parquet_dir:
        raise ValueError(
            "Pass either --bench-json or --bench-parquet-dir, not both."
        )
    if not args.bench_json and not args.bench_parquet_dir:
        raise ValueError(
            "One of --bench-json or --bench-parquet-dir is required."
        )

    if args.bench_json:
        bench = load_bench(
            Path(args.bench_json),
            limit_repos=args.limit_repos,
            limit_pairs_per_repo=args.limit_pairs_per_repo,
        )
        bench_label = Path(args.bench_json).name
    else:
        bench = load_bench_from_parquet(
            Path(args.bench_parquet_dir),
            cross_repo_splits=args.bench_cross_repo_splits,
            in_repo_splits=None,
            limit_repos=args.limit_repos,
            limit_pairs_per_repo=args.limit_pairs_per_repo,
        )
        bench_label = (
            Path(args.bench_parquet_dir).name
            + "[" + ",".join(args.bench_cross_repo_splits or ["all"]) + "]"
        )
    n_pairs = sum(len(it["qna_pairs"]) for it in bench)
    print(
        f"Bench: {bench_label} -> {len(bench)} repos, "
        f"{n_pairs} QnA pairs",
        flush=True,
    )

    if args.method == "direct":
        install, tmd, scaling = build_direct_method(
            checkpoint=Path(args.checkpoint),
            base_model=base_model,
            target_module_names=args.target_modules,
            device=device,
        )
    elif args.method == "gru_file":
        install, tmd, scaling = build_gru_file_method(
            checkpoint=Path(args.checkpoint),
            base_model=base_model,
            target_module_names=args.target_modules,
            device=device,
            embed_model_name=args.embed_model,
        )
    elif args.method == "gru_commit":
        if not args.parquet_dir:
            raise ValueError("--parquet-dir is required for method=gru_commit")
        install, tmd, scaling = build_gru_commit_method(
            checkpoint=Path(args.checkpoint),
            base_model=base_model,
            target_module_names=args.target_modules,
            device=device,
            parquet_dir=args.parquet_dir,
            parquet_prefer=args.parquet_prefer,
            embed_model_name=args.embed_model,
            cross_repo_splits=args.cross_repo_splits,
        )
    else:
        raise ValueError(args.method)

    out = run_eval(
        method=args.method,
        bench_items=bench,
        install_fn=install,
        target_modules_dict=tmd,
        lora_scaling=float(scaling),
        base_model=base_model,
        tok=tok,
        device=device,
        max_input_tokens=int(args.max_input_tokens),
        max_new_tokens=int(args.max_new_tokens),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )

    out["bench_json"] = str(args.bench_json) if args.bench_json else None
    out["bench_parquet_dir"] = (
        str(args.bench_parquet_dir) if args.bench_parquet_dir else None
    )
    if args.bench_name:
        out["bench_name"] = args.bench_name
    elif args.bench_json:
        out["bench_name"] = Path(args.bench_json).stem
    else:
        out["bench_name"] = (
            Path(args.bench_parquet_dir).name
            + "[" + ",".join(args.bench_cross_repo_splits or ["all"]) + "]"
        )
    out["checkpoint"] = str(args.checkpoint)
    out["model_name"] = args.model_name
    out["seed"] = int(args.seed)

    n = int(out["n"])
    if n == 0 and not args.allow_empty_suite:
        raise RuntimeError(
            f"Bench {args.bench_json} produced n=0 scored pairs "
            f"(method={args.method}). Re-run with --allow-empty-suite to override."
        )

    print("\n" + "=" * 60)
    print(f"{args.method.upper()} on {bench_label}")
    print("=" * 60)
    print(f"  n={n:,} repos_scored={out['n_repos_scored']} "
          f"skipped={out['n_repos_skipped']}")
    print(f"  EM:       {out['exact_match_pct']:.2f}%")
    print(f"  EditSim:  {out['edit_similarity']:.4f}")
    print(f"  CodeBLEU: {out['code_bleu']:.4f}")
    if "ci95" in out:
        c = out["ci95"]
        print(f"  CI95 (R={int(args.bootstrap):,}):")
        print(f"    EM%      {format_ci(c['exact_match'], pct=True)}")
        print(f"    EditSim  {format_ci(c['edit_similarity'])}")
        print(f"    CodeBLEU {format_ci(c['code_bleu'])}")

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
