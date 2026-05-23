#!/usr/bin/env python3
"""Generate Doc-to-LoRA teacher logprobs against the **v2 commit-derived
RepoPeft dataset**.

Drops in next to ``generate_teacher_logprobs.py`` (which targets the v1
flat-JSON splits + per-repo ``ORACLE_CONTEXT_CACHE_V4``). The v2 sibling
reads:

* QnAs from ``code2lora_snapshots_hf/qna/train.parquet`` (44 K rows,
  400 anchor-commit groups -- exactly one anchor commit per repo).
* DRC contexts from either of two layouts, controlled by
  ``--drc-cache-mode``:

  * ``per_commit`` (default): ``<cache_dir>/<repo>__<sha>.json`` --
    the v2 per-commit cache built by
    ``evaluation/build_drc_cache_per_commit.py``. Requires the QnA
    parquet to expose ``lineno`` / ``col_offset`` /
    ``assertion_event_id``; ``train.parquet`` does not, but
    ``ir_*`` / ``cr_*`` do.
  * ``per_repo``: ``<cache_dir>/<repo_slug>.json`` -- the v1
    ``ORACLE_CONTEXT_CACHE_V4`` layout (one JSON per repo, ``contexts``
    keyed by ``<test_file>::<lineno>``). This is what the existing
    ``generate_teacher_logprobs.py`` consumed; we reuse it for v2
    train because ``train.parquet`` lacks the per-assertion location
    columns.

One parquet row per ``(repo_id, commit_sha)`` group -- exactly matching
the v1 layout's "one repo per row" semantics, just with the **per-commit
DRC document** as the row's ``ctx_ids``. This is the same document the
v2 evaluator will internalize at test time.

Sharding
--------
``--shard-i / --num-shards`` is round-robin over sorted ``repo_id`` (one
shard task = a slice of repos), so multiple GPU jobs can write disjoint
``ds_*.parquet`` files into the same output dir. Each shard's filenames
are namespaced ``ds_shard{i}of{n}_{seq:04d}.parquet`` to avoid
collisions with parallel array tasks.

Output schema (matches D2L's expected fields exactly, see the v1
generator and ``doc2lora/src/ctx_to_lora/data/processing.py``)::

    ctx_ids            : list[int]                 tokenized per-commit DRC doc
    input_ids          : list[list[int]]           per-QA: prefix + target tokens
    response_start_end : list[(int, int)]          per-QA: target span in input_ids
    logprobs_vals      : list[ndarray (n_tgt, 16)] per-QA top-16 logprob values
    logprobs_indices   : list[ndarray (n_tgt, 16)] per-QA top-16 vocab indices

Usage
-----

    python baselines/doc2lora/generate_teacher_logprobs_v2.py \\
        --qna-parquet $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/train.parquet \\
        --drc-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V4 \\
        --drc-cache-mode per_repo \\
        --output-dir doc2lora/data/raw_datasets/self_gen/Qwen/Qwen2.5-Coder-1.5B/repopeft/train_v2/train \\
        --model Qwen/Qwen2.5-Coder-1.5B \\
        --num-shards 4 --shard-i 0

The directory layout matters: ``ctx_to_lora.data.processing:get_ds_kwargs``
globs ``<SELF_GEN_DATA_DIR>/<base_model>/<base_ds>/<split>/*.parquet``, so
the output dir MUST end in ``/<split>`` (typically ``/train``). The v1
trainer's flat-JSON splits used the same convention.
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

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hypernetwork.code2lora_core import load_qna_rows  # noqa: E402


TOP_K = 16
DRC_SEPARATOR = "\n\n\n"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"


# ---------------------------------------------------------------------------
# DRC cache loader (per_commit = v2; per_repo = v1 ORACLE_CONTEXT_CACHE_V4)
# ---------------------------------------------------------------------------

def _drc_per_commit_path(cache_dir: Path, repo_id: str, sha: str) -> Path:
    safe = repo_id.replace("/", "__")
    return cache_dir / f"{safe}__{sha}.json"


def _drc_per_repo_path(cache_dir: Path, repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return cache_dir / f"{safe}.json"


def _load_drc_contexts(cache_dir: Path, repo_id: str, sha: str,
                       mode: str) -> Dict[str, Dict[str, Any]]:
    """Return a ``{key: {extracted_code: ..., ...}}`` map for this
    ``(repo, commit)``. Supports both v2 per-commit and v1 per-repo
    cache layouts (see module docstring)."""
    if mode == "per_commit":
        path = _drc_per_commit_path(cache_dir, repo_id, sha)
    elif mode == "per_repo":
        path = _drc_per_repo_path(cache_dir, repo_id)
    else:
        raise ValueError(f"unknown drc-cache-mode: {mode}")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [drc] failed to load {path}: {type(e).__name__}: {e}",
              flush=True)
        return {}
    return data.get("contexts") or {}


def _drc_keys_for_qna(q) -> List[str]:
    """Possible keys used by the evaluator for the same QnA. We try the
    canonical ``assertion_event_id`` first and fall back to the
    ``test_file::lineno::col_offset`` triple.
    """
    keys: List[str] = []
    if getattr(q, "assertion_event_id", ""):
        keys.append(q.assertion_event_id)
    keys.append(
        f"{q.test_file}::{int(q.lineno)}::{int(q.col_offset)}"
    )
    return keys


def _build_commit_doc(contexts: Dict[str, Dict[str, Any]],
                      tokenizer, max_ctx_tokens: int) -> Tuple[List[int], str]:
    """Concatenate deduplicated DRC ``extracted_code`` strings present at
    this commit into one document, tokenize, and left-truncate to
    ``max_ctx_tokens``. The original text is also returned for debug.

    Mirrors the v1 generator's per-repo doc construction, applied
    per-commit instead.
    """
    snippets: List[str] = []
    seen: set = set()
    for v in contexts.values():
        if not isinstance(v, dict):
            continue
        code = v.get("extracted_code") or ""
        if not code or code in seen:
            continue
        seen.add(code)
        snippets.append(code)
    snippets.sort()
    doc_text = "\n\n".join(snippets)
    if not doc_text:
        return [], ""
    ids = tokenizer.encode(doc_text, add_special_tokens=False)
    if len(ids) > max_ctx_tokens:
        ids = ids[:max_ctx_tokens]
    return ids, doc_text


# ---------------------------------------------------------------------------
# Teacher forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_teacher_logprobs(model, teacher_ids: torch.Tensor,
                              target_start: int, target_end: int,
                              top_k: int = TOP_K
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Run one teacher forward; return top-k logprob (vals, indices) at
    the target span. Shapes: ``(n_target, top_k)`` for both."""
    out = model(teacher_ids)
    logits = out.logits[0]
    pred_logits = logits[target_start - 1: target_end - 1]  # [n_target, V]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    top_vals, top_idx = torch.topk(log_probs, k=top_k, dim=-1)
    return (
        top_vals.cpu().to(torch.float16).numpy(),
        top_idx.cpu().to(torch.int32).numpy(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--qna-parquet",
                    default=f"{scratch}/REPO_DATASET/code2lora_snapshots_hf/"
                            f"qna/train.parquet")
    ap.add_argument("--drc-cache-dir",
                    default=f"{scratch}/ORACLE_CONTEXT_CACHE_V4")
    ap.add_argument("--drc-cache-mode", default="per_repo",
                    choices=["per_commit", "per_repo"],
                    help="per_commit: <cache_dir>/<repo>__<sha>.json "
                         "(v2 build_drc_cache_per_commit.py output, requires "
                         "lineno/col_offset/assertion_event_id columns in "
                         "the qna parquet). "
                         "per_repo:  <cache_dir>/<repo_slug>.json "
                         "(v1 ORACLE_CONTEXT_CACHE_V4 layout, schema-"
                         "agnostic - use this for train.parquet).")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write ds_*.parquet shards into. "
                         "Multiple parallel runs (different --shard-i) "
                         "should target the same dir.")
    ap.add_argument("--model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--max-ctx-tokens", type=int, default=4096,
                    help="Truncate per-commit DRC document to this many "
                         "tokens (matches the v2 DRC eval cap).")
    ap.add_argument("--max-teacher-tokens", type=int, default=8192)
    ap.add_argument("--max-input-tokens", type=int, default=2048,
                    help="Truncate (prefix + target) for the student-side "
                         "input_ids. Matches the v2 max_qas_len in the "
                         "D2L config.")
    ap.add_argument("--qnas-per-commit-limit", type=int, default=0,
                    help="Optional cap on QnAs per (repo, commit) when "
                         "generating logprobs. 0 = no cap (use all). "
                         "Setting this to 8 would match the eval cap.")
    ap.add_argument("--shard-size", type=int, default=50,
                    help="Number of (repo, commit) rows per ds_*.parquet.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[args] qna_parquet   = {args.qna_parquet}", flush=True)
    print(f"[args] drc_cache_dir = {args.drc_cache_dir}", flush=True)
    print(f"[args] output_dir    = {args.output_dir}", flush=True)
    print(f"[args] model         = {args.model}", flush=True)
    print(f"[args] shard         = {args.shard_i + 1}/{args.num_shards}",
          flush=True)

    qna_parquet = Path(args.qna_parquet).expanduser().resolve()
    drc_cache_dir = Path(args.drc_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not qna_parquet.exists():
        raise SystemExit(f"missing qna parquet: {qna_parquet}")
    if not drc_cache_dir.exists():
        raise SystemExit(f"missing DRC cache dir: {drc_cache_dir}")

    print(f"\n[load] teacher model {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    # ---- Enumerate (repo, commit) groups & filter to this shard ----
    print(f"\n[load] qna rows from {qna_parquet} ...", flush=True)
    rows = load_qna_rows(qna_parquet)
    print(f"[load]   {len(rows):,} qnas", flush=True)
    by_group: Dict[Tuple[str, str], List] = defaultdict(list)
    for qr in rows:
        by_group[(qr.repo_id, qr.commit_sha)].append(qr)
    group_keys = sorted(by_group.keys())
    print(f"[load]   {len(group_keys):,} (repo, commit) groups", flush=True)

    if args.num_shards > 1:
        all_repos = sorted({r for (r, _) in group_keys})
        kept_repos = {r for i, r in enumerate(all_repos)
                      if i % args.num_shards == args.shard_i}
        group_keys = [k for k in group_keys if k[0] in kept_repos]
        print(f"[shard] {args.shard_i + 1}/{args.num_shards}: "
              f"{len(kept_repos)}/{len(all_repos)} repos -> "
              f"{len(group_keys)} groups", flush=True)

    # ---- Stream groups; write shards ----
    sep_ids = tokenizer.encode(DRC_SEPARATOR, add_special_tokens=False)
    out_rows: List[Dict[str, Any]] = []
    n_skipped_no_drc = 0
    n_skipped_overflow = 0
    n_qa_done = 0
    n_groups_done = 0
    t0 = time.time()
    shard_id_in_run = 0

    def _flush_shard() -> None:
        nonlocal out_rows, shard_id_in_run
        if not out_rows:
            return
        if args.num_shards > 1:
            fname = (f"ds_shard{args.shard_i}of{args.num_shards}"
                     f"_{shard_id_in_run:04d}.parquet")
        else:
            fname = f"ds_{shard_id_in_run:04d}.parquet"
        path = output_dir / fname
        tmp = path.with_suffix(path.suffix + ".tmp")
        Dataset.from_list(out_rows).to_parquet(str(tmp))
        os.replace(tmp, path)
        print(f"  [flush] wrote {len(out_rows):>4d} rows -> {path.name}",
              flush=True)
        out_rows = []
        shard_id_in_run += 1

    for (repo_id, sha) in tqdm(group_keys, desc="groups", smoothing=0.0):
        contexts = _load_drc_contexts(drc_cache_dir, repo_id, sha,
                                      args.drc_cache_mode)
        if not contexts:
            n_skipped_no_drc += len(by_group[(repo_id, sha)])
            continue

        ctx_ids, _doc_text = _build_commit_doc(
            contexts, tokenizer, args.max_ctx_tokens,
        )
        if not ctx_ids:
            n_skipped_no_drc += len(by_group[(repo_id, sha)])
            continue

        qnas = by_group[(repo_id, sha)]
        if args.qnas_per_commit_limit and len(qnas) > args.qnas_per_commit_limit:
            qnas = qnas[: args.qnas_per_commit_limit]

        qa_input_ids: List[List[int]] = []
        qa_response_se: List[List[int]] = []
        qa_logprobs_vals: List[np.ndarray] = []
        qa_logprobs_indices: List[np.ndarray] = []

        for q in qnas:
            prefix = q.prefix or ""
            target = q.target or ""
            if not prefix or not target:
                n_skipped_overflow += 1
                continue
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                n_skipped_overflow += 1
                continue

            student_ids = prefix_ids + target_ids
            if len(student_ids) > args.max_input_tokens:
                max_prefix = args.max_input_tokens - len(target_ids)
                if max_prefix <= 0:
                    n_skipped_overflow += 1
                    continue
                prefix_ids = prefix_ids[-max_prefix:]
                student_ids = prefix_ids + target_ids
            response_start = len(prefix_ids)
            response_end = len(student_ids)

            # Try to look up per-QnA context first (more focused doc for
            # this assertion); otherwise fall back to the commit-level doc.
            per_qna_ctx_text: Optional[str] = None
            for k in _drc_keys_for_qna(q):
                v = contexts.get(k)
                if isinstance(v, dict):
                    code = v.get("extracted_code") or ""
                    if code:
                        per_qna_ctx_text = code
                        break

            if per_qna_ctx_text:
                teacher_ctx_ids = tokenizer.encode(per_qna_ctx_text,
                                                  add_special_tokens=False)
                if len(teacher_ctx_ids) > args.max_ctx_tokens:
                    teacher_ctx_ids = teacher_ctx_ids[: args.max_ctx_tokens]
            else:
                teacher_ctx_ids = list(ctx_ids)

            teacher_full = teacher_ctx_ids + sep_ids + prefix_ids + target_ids
            if len(teacher_full) > args.max_teacher_tokens:
                budget = args.max_teacher_tokens - len(sep_ids) - len(prefix_ids) - len(target_ids)
                if budget <= 0:
                    n_skipped_overflow += 1
                    continue
                teacher_ctx_ids = teacher_ctx_ids[:budget]
                teacher_full = teacher_ctx_ids + sep_ids + prefix_ids + target_ids

            target_start_t = len(teacher_full) - len(target_ids)
            target_end_t = len(teacher_full)
            teacher_tensor = torch.tensor([teacher_full], dtype=torch.long, device=device)
            lp_vals, lp_idx = _extract_teacher_logprobs(
                model, teacher_tensor, target_start_t, target_end_t, top_k=TOP_K,
            )

            qa_input_ids.append(student_ids)
            qa_response_se.append([response_start, response_end])
            qa_logprobs_vals.append(lp_vals)
            qa_logprobs_indices.append(lp_idx)
            n_qa_done += 1

        if not qa_input_ids:
            continue

        out_rows.append({
            "ctx_ids": ctx_ids,
            "input_ids": qa_input_ids,
            "response_start_end": qa_response_se,
            "logprobs_vals": qa_logprobs_vals,
            "logprobs_indices": qa_logprobs_indices,
        })
        n_groups_done += 1
        if len(out_rows) >= args.shard_size:
            _flush_shard()

        if n_groups_done % 10 == 0:
            elapsed = (time.time() - t0) / 60
            rate = n_qa_done / max(elapsed, 1e-6)
            print(f"  [progress] {n_groups_done}/{len(group_keys)} groups, "
                  f"{n_qa_done:,} qas, "
                  f"elapsed={elapsed:.1f}m, rate={rate:.1f} qa/min",
                  flush=True)

    _flush_shard()
    elapsed = (time.time() - t0) / 60
    print(f"\nDONE shard {args.shard_i + 1}/{args.num_shards}:")
    print(f"  groups       : {n_groups_done}/{len(group_keys)}")
    print(f"  qas          : {n_qa_done:,}")
    print(f"  skipped(noDRC): {n_skipped_no_drc:,}")
    print(f"  skipped(oflow): {n_skipped_overflow:,}")
    print(f"  elapsed      : {elapsed:.1f} min")


if __name__ == "__main__":
    main()
