#!/usr/bin/env python3
"""Per-commit task-metric evaluation for **v2 Code2LoRA-GRU** checkpoints.

Loads a checkpoint produced by ``hypernetwork/train_code2lora_gru_v2.py``
(``gru_head.*.pt``), reconstructs the :class:`CommitGRU` + :class:`Code2LoRAHead`,
wraps the base LLM with :class:`LoRA` modules, and walks every (repo, commit)
in a suite -- stepping the GRU one commit at a time and scoring every QnA
attached to that commit with the LoRA freshly generated from ``head(ctx_t)``.

This is **the** evaluation script for the v2 GRU. Output schema matches
``evaluation/run_baselines_v2.py`` so plots overlay cleanly:

    {
        "summary": {"suite": ..., "exact_match": ..., "exact_match_ci": [lo, hi], ...},
        "per_commit": [
            {"repo_id": ..., "commit_sha": ..., "commit_index": ...,
             "n_qnas": ..., "exact_match": ..., "edit_similarity": ...,
             "code_bleu": ...},
            ...
        ],
        "raw_samples": {"exact_match": [...], "edit_similarity": [...], "code_bleu": [...]}
    }

Key features:

* **Sharded** by repo (deterministic round-robin) via ``--shard-i / --num-shards``.
* **Per-(repo, commit) incremental atomic write** (``tmp.write_text + os.replace``)
  -- a timeout never loses more than the one commit currently in flight.
* **Resumable**: if the output JSON already exists and is not ``finalized``,
  finished (repo, commit) groups are skipped on restart.
* Uses **precomputed diff + repo-state embeddings** straight out of the v2
  commits parquet -- no encoder forward pass during eval.

Usage::

    python evaluation/run_code2lora_gru_v2_eval.py \\
        --checkpoint $CKPT_DIR/CODE2LORA_GRU/.../gru_head.best.pt \\
        --commits-dir $SCRATCH/REPO_DATASET/commit_parquet_hf_v2 \\
        --qnas-dir    $SCRATCH/REPO_DATASET/code2lora_snapshots_hf \\
        --suite cr_val \\
        --output-dir $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/<run>/ \\
        --shard-i 0 --num-shards 4
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
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HYP = _ROOT / "hypernetwork"
if str(_HYP) not in sys.path:
    sys.path.insert(0, str(_HYP))

from code2lora_core import (  # noqa: E402
    Code2LoRAHead,
    CommitGRU,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_commit_rows_for_gru,
    load_qna_rows,
    replace_with_lora,
)
from evaluation.metrics import (  # noqa: E402
    aggregate_metrics_with_ci,
    compute_metrics,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_v2_gru_ckpt(
    ckpt_path: Path,
    base_model_name: str,
    target_modules: List[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, CommitGRU, Code2LoRAHead, List[Any], Any]:
    """Rebuild (base_model, gru, head, specs, tokenizer) from a v2 ckpt."""
    print(f"[load] ckpt={ckpt_path}", flush=True)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    head_cfg = state.get("head_config") or {}
    gru_cfg = state.get("gru_config") or {}

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    specs = get_module_specs(base_model, target_modules)
    type_dims = discover_module_types_and_dims(specs)
    # Rebuild the head with the SAME hyperparameters used at training time.
    rank = int(head_cfg.get("rank", 16))
    # Try a few argv sources to figure out alpha.
    alpha = float(state.get("args", {}).get("alpha", 32.0))
    head_hidden = int(head_cfg.get("hidden_dim", 1024))
    input_dim = int(head_cfg.get("input_dim",
                                 gru_cfg.get("hidden_dim", 2048)))
    head = Code2LoRAHead(
        input_dim=input_dim,
        type_dims=type_dims,
        hidden_dim=head_hidden,
        rank=rank,
    ).to(device)
    head.load_state_dict(state["head_state"])
    head.eval()

    # CommitGRU
    gru = CommitGRU(
        diff_input_dim=int(gru_cfg.get("diff_input_dim", 2048)),
        repo_state_dim=int(gru_cfg.get("repo_state_dim", 2048)),
        hidden_dim=int(gru_cfg.get("hidden_dim", 2048)),
    ).to(device)
    gru.load_state_dict(state["gru_state"])
    gru.eval()

    # Wrap base model with LoRA modules (matches training-time setup).
    replace_with_lora(base_model, specs, rank=rank, alpha=alpha)
    print(f"[load] base+LoRA ready, head rank={rank} alpha={alpha} "
          f"hidden={head_hidden}; gru hidden={gru.hidden_dim}; "
          f"types={sorted(type_dims)}", flush=True)
    return base_model, gru, head, specs, tokenizer


# ---------------------------------------------------------------------------
# Tokenization + generation
# ---------------------------------------------------------------------------

def _prepare_prefix_ids(tokenizer, prefix: str, max_input_tokens: int,
                        bos_id: Optional[int]) -> List[int]:
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if bos_id is not None:
        ids = [bos_id] + ids
    if len(ids) > max_input_tokens:
        ids = ids[-max_input_tokens:]
    return ids


def _get_bos_id(tok) -> Optional[int]:
    bid = getattr(tok, "bos_token_id", None)
    if bid is not None:
        return int(bid)
    eid = getattr(tok, "eos_token_id", None)
    if eid is not None:
        return int(eid)
    return None


@torch.no_grad()
def _generate_batch(model, tokenizer, device, input_ids_list: List[List[int]],
                    max_new_tokens: int) -> List[str]:
    L = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    bs = len(input_ids_list)
    input_ids = torch.full((bs, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, L), dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids_list):
        n = len(ids)
        input_ids[i, L - n:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, L - n:] = 1
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    decoded: List[str] = []
    for i in range(bs):
        gen = out[i, L:].tolist()
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True))
    return decoded


# ---------------------------------------------------------------------------
# Atomic incremental write
# ---------------------------------------------------------------------------

def _summarize(samples: List[Tuple[float, float, float]],
               bootstrap: int = 0) -> Dict[str, Any]:
    if not samples:
        return {"n_qnas": 0, "exact_match": 0.0,
                "edit_similarity": 0.0, "code_bleu": 0.0}
    n = len(samples)
    if bootstrap <= 0:
        return {
            "n_qnas": n,
            "exact_match": sum(s[0] for s in samples) / n,
            "edit_similarity": sum(s[1] for s in samples) / n,
            "code_bleu": sum(s[2] for s in samples) / n,
        }
    metric_dicts = [
        {"exact_match": bool(em), "edit_similarity": ed, "code_bleu": cb}
        for (em, ed, cb) in samples
    ]
    agg = aggregate_metrics_with_ci(metric_dicts, n_resamples=int(bootstrap))
    out: Dict[str, Any] = {"n_qnas": n}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            out[k] = float(v["mean"])
            out[f"{k}_ci"] = [float(v.get("low", 0.0)),
                              float(v.get("high", 0.0))]
            out[f"{k}_n"] = int(v.get("n", 0))
        else:
            out[k] = v
    return out


def _write_suite_json(
    out_path: Path, suite_name: str,
    per_commit_records: List[Dict[str, Any]],
    all_samples: List[Tuple[float, float, float]],
    *,
    bootstrap: int,
    finalized: bool,
    shard_i: int,
    num_shards: int,
    n_total_groups: int,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary is None:
        summary = _summarize(all_samples, bootstrap=bootstrap)
        summary["suite"] = suite_name
        summary["n_qnas"] = len(all_samples)
        summary["n_scored_commits"] = len(per_commit_records)
        summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})
    payload: Dict[str, Any] = {
        "finalized": finalized,
        "shard_i": int(shard_i),
        "num_shards": int(num_shards),
        "n_total_groups": int(n_total_groups),
        "summary": summary,
        "per_commit": per_commit_records,
    }
    if finalized:
        payload["raw_samples"] = {
            "exact_match": [int(s[0]) for s in all_samples],
            "edit_similarity": [float(s[1]) for s in all_samples],
            "code_bleu": [float(s[2]) for s in all_samples],
        }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite(
    *,
    suite_name: str,
    commits_parquet: Path,
    qna_parquet: Path,
    base_model: torch.nn.Module,
    gru: CommitGRU,
    head: Code2LoRAHead,
    specs: List[Any],
    tokenizer,
    device: torch.device,
    out_path: Path,
    max_input_tokens: int,
    max_new_tokens: int,
    batch_size: int,
    qnas_per_commit_limit: int,
    bootstrap: int,
    shard_i: int,
    num_shards: int,
    in_repo_splits_to_score: Optional[List[str]],
) -> Dict[str, Any]:
    """Walk every repo in the suite, step the GRU per commit, score QnAs."""
    print(f"\n[suite {suite_name}] loading commits parquet {commits_parquet}",
          flush=True)
    rows_by_repo = load_commit_rows_for_gru(commits_parquet)
    all_repos = sorted(rows_by_repo.keys())
    print(f"[suite {suite_name}] {len(all_repos)} repos, "
          f"{sum(len(v) for v in rows_by_repo.values())} commits total",
          flush=True)

    if num_shards > 1:
        kept = [r for i, r in enumerate(all_repos) if i % num_shards == shard_i]
        rows_by_repo = {r: rows_by_repo[r] for r in kept}
        print(f"[suite {suite_name}] shard {shard_i+1}/{num_shards}: "
              f"{len(rows_by_repo)} repos", flush=True)

    print(f"[suite {suite_name}] loading qnas {qna_parquet}", flush=True)
    qna_rows = load_qna_rows(qna_parquet)
    qnas_by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for qr in qna_rows:
        qnas_by_key[(qr.repo_id, qr.commit_sha)].append({
            "prefix": qr.prefix, "target": qr.target,
        })
    print(f"[suite {suite_name}] {len(qna_rows)} qnas, "
          f"{len(qnas_by_key)} (repo, commit) keys", flush=True)

    # Predict how many scored (repo, commit) groups exist for our shard so the
    # progress / ETA prints are meaningful.
    n_total_groups = 0
    for repo, rows in rows_by_repo.items():
        for row in rows:
            if (in_repo_splits_to_score is not None and
                    row.in_repo_split not in in_repo_splits_to_score):
                continue
            if (repo, row.commit_sha) in qnas_by_key:
                n_total_groups += 1
    print(f"[suite {suite_name}] expected scored groups in this shard: "
          f"{n_total_groups}", flush=True)

    # ---- Resume support ----
    per_commit_records: List[Dict[str, Any]] = []
    all_samples: List[Tuple[float, float, float]] = []
    done_keys: set = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if not prev.get("finalized"):
                per_commit_records = list(prev.get("per_commit", []))
                done_keys = {(r["repo_id"], r["commit_sha"])
                             for r in per_commit_records}
                for r in per_commit_records:
                    n = int(r["n_qnas"])
                    em = float(r["exact_match"])
                    ed = float(r["edit_similarity"])
                    cb = float(r["code_bleu"])
                    for _ in range(n):
                        all_samples.append((em, ed, cb))
                print(f"[suite {suite_name}] resuming: "
                      f"{len(done_keys)} groups already scored", flush=True)
        except Exception as e:
            print(f"[suite {suite_name}] could not parse existing JSON: {e}",
                  flush=True)

    bos_id = _get_bos_id(tokenizer)
    base_model.eval(); gru.eval(); head.eval()

    t0 = time.time()
    n_done = len(done_keys)
    n_run = 0
    for repo_id in sorted(rows_by_repo.keys()):
        rows = rows_by_repo[repo_id]
        if not rows:
            continue
        # h_0 from initial repo-state embedding @ commit 0.
        repo_emb_0 = torch.from_numpy(rows[0].repo_state_embedding).to(device).unsqueeze(0)
        h = gru.init_hidden(repo_emb_0)

        for row in rows:
            diff_emb = torch.from_numpy(row.diff_embedding).to(device).unsqueeze(0)
            h = gru.step(diff_emb, h)
            score_this = True
            if (in_repo_splits_to_score is not None and
                    row.in_repo_split not in in_repo_splits_to_score):
                score_this = False
            if not score_this:
                continue
            pairs = qnas_by_key.get((row.repo_id, row.commit_sha))
            if not pairs:
                continue
            if (row.repo_id, row.commit_sha) in done_keys:
                continue
            if qnas_per_commit_limit and len(pairs) > qnas_per_commit_limit:
                pairs = pairs[:qnas_per_commit_limit]

            ctx = gru.output_norm(h[-1])
            head_out = head(ctx)
            inject_lora_weights(base_model, specs, head_out, batch_index=0)

            commit_samples: List[Tuple[float, float, float]] = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                inputs = [
                    _prepare_prefix_ids(tokenizer, p["prefix"],
                                        max_input_tokens, bos_id)
                    for p in batch_pairs
                ]
                preds = _generate_batch(
                    base_model, tokenizer, device, inputs,
                    max_new_tokens=max_new_tokens,
                )
                for p, pred in zip(batch_pairs, preds):
                    m = compute_metrics(pred, p["target"])
                    em = 1.0 if m["exact_match"] else 0.0
                    ed = float(m["edit_similarity"])
                    cb = float(m["code_bleu"])
                    commit_samples.append((em, ed, cb))
                    all_samples.append((em, ed, cb))
            if commit_samples:
                n_c = len(commit_samples)
                per_commit_records.append({
                    "repo_id": row.repo_id,
                    "commit_sha": row.commit_sha,
                    "commit_index": int(row.commit_index),
                    "in_repo_split": row.in_repo_split,
                    "n_qnas": n_c,
                    "exact_match": sum(s[0] for s in commit_samples) / n_c,
                    "edit_similarity": sum(s[1] for s in commit_samples) / n_c,
                    "code_bleu": sum(s[2] for s in commit_samples) / n_c,
                })
            n_done += 1
            n_run += 1
            _write_suite_json(
                out_path, suite_name, per_commit_records, all_samples,
                bootstrap=0, finalized=False, shard_i=shard_i,
                num_shards=num_shards, n_total_groups=n_total_groups,
            )
            if n_run % 25 == 0 or n_done == n_total_groups:
                elapsed = (time.time() - t0) / 60
                done_em = (sum(s[0] for s in all_samples) /
                           max(len(all_samples), 1))
                rate = n_run / max(elapsed, 1e-6)
                eta = (n_total_groups - n_done) / max(rate, 1e-6)
                print(f"  [suite {suite_name} sh{shard_i+1}/{num_shards}] "
                      f"{n_done}/{n_total_groups} groups "
                      f"({len(all_samples):,} qnas) "
                      f"running_EM={done_em:.4f} "
                      f"elapsed={elapsed:.1f}m ETA={eta:.1f}m", flush=True)

    # Final summary with bootstrap CI on union of samples in THIS shard.
    summary = _summarize(all_samples, bootstrap=bootstrap)
    summary["suite"] = suite_name
    summary["n_qnas"] = len(all_samples)
    summary["n_scored_commits"] = len(per_commit_records)
    summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})
    _write_suite_json(
        out_path, suite_name, per_commit_records, all_samples,
        bootstrap=bootstrap, finalized=True, summary=summary,
        shard_i=shard_i, num_shards=num_shards,
        n_total_groups=n_total_groups,
    )
    print(f"[suite {suite_name}] shard {shard_i+1}/{num_shards} DONE  "
          f"EM={summary['exact_match']:.4f}  "
          f"EditSim={summary['edit_similarity']:.4f}  "
          f"BLEU={summary['code_bleu']:.4f}  "
          f"({len(all_samples):,} qnas, {len(per_commit_records):,} commits)",
          flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to gru_head.{best,latest,epN}.pt produced "
                         "by train_code2lora_gru_v2.py.")
    ap.add_argument("--commits-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2",
                    help="Dir containing commits/{ir_val,ir_test,cr_val,cr_test}.parquet.")
    ap.add_argument("--qnas-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf",
                    help="Dir containing qna/{ir_val,ir_test,cr_val,cr_test}.parquet.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--suite", required=True,
                    choices=["ir_val", "ir_test", "cr_val", "cr_test",
                             "ood_test"])
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--qnas-per-commit-limit", type=int, default=8,
                    help="Cap QnAs per commit (matches v2 trainer eval cap).")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = (f"_shard{args.shard_i}of{args.num_shards}"
                    if args.num_shards > 1 else "")
    out_path = out_dir / f"gru_v2_{args.suite}{shard_suffix}.json"

    base_model, gru, head, specs, tok = _load_v2_gru_ckpt(
        Path(args.checkpoint), args.base_model, args.target_modules,
        device=device,
    )

    if args.suite.startswith("cr_") or args.suite == "ood_test":
        in_repo_splits = None
    elif args.suite == "ir_val":
        in_repo_splits = ["val"]
    else:
        in_repo_splits = ["test"]

    commits_parquet = Path(args.commits_dir) / "commits" / f"{args.suite}.parquet"
    qna_parquet = Path(args.qnas_dir) / "qna" / f"{args.suite}.parquet"
    # In-repo suites (ir_val / ir_test) share the 400-repo timeline that's
    # already materialized in commits/train.parquet (in_repo_split tags val/test
    # rows inside it). Only cr_* have their own dedicated commits parquet.
    if not commits_parquet.exists() and args.suite.startswith("ir_"):
        fallback = Path(args.commits_dir) / "commits" / "train.parquet"
        if fallback.exists():
            print(f"[info] {commits_parquet.name} not found; using train.parquet "
                  f"(filtered to in_repo_split={in_repo_splits}) instead",
                  flush=True)
            commits_parquet = fallback
    if not commits_parquet.exists():
        raise SystemExit(f"missing commits parquet: {commits_parquet}")
    if not qna_parquet.exists():
        raise SystemExit(f"missing qna parquet: {qna_parquet}")

    evaluate_suite(
        suite_name=args.suite,
        commits_parquet=commits_parquet,
        qna_parquet=qna_parquet,
        base_model=base_model, gru=gru, head=head, specs=specs,
        tokenizer=tok, device=device,
        out_path=out_path,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        qnas_per_commit_limit=args.qnas_per_commit_limit,
        bootstrap=args.bootstrap,
        shard_i=args.shard_i,
        num_shards=args.num_shards,
        in_repo_splits_to_score=in_repo_splits,
    )


if __name__ == "__main__":
    main()
