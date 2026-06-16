#!/usr/bin/env python3
"""Per-commit task-metric evaluation for **Code2LoRA-GRU anchor-only** checkpoints.

This is the eval-time analogue of ``hypernetwork/train_code2lora_gru_anchor.py``:
load a ``gru_head.*.pt`` checkpoint trained with the anchor-only ablation,
walk the chronological commit list of every (repo, commit) in a suite, and at
every step feed the **anchor** ``repo_state_embedding`` from the
``code2lora_snapshots_hf`` dataset (never the per-commit ``diff_embedding``).
Otherwise the per-commit generation / metric pipeline is identical to
``evaluation/run_code2lora_gru_v2_eval.py``, so output JSONs overlay cleanly.

Anchor selection per repo (mirrors the trainer):
  * ``ir_val`` / ``ir_test`` -> the train anchor row from
    ``snapshots/commits/train.parquet``.
  * ``cr_val`` / ``cr_test`` -> the latest kept commit per repo from
    ``snapshots/commits/{cr_val,cr_test}.parquet``.

Usage::

    python evaluation/run_code2lora_gru_anchor_eval.py \\
        --checkpoint $CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_anchor_3ep/gru_head.best.pt \\
        --commits-dir   $SCRATCH/REPO_DATASET/commit_parquet_hf_v2 \\
        --snapshots-dir $SCRATCH/REPO_DATASET/code2lora_snapshots_hf \\
        --suite cr_test \\
        --output-dir $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/anchor_<run>/ \\
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

import numpy as np
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
    SnapshotRow,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_commit_rows_for_gru,
    load_qna_rows,
    load_snapshot_rows,
    replace_with_lora,
)
from evaluation.metrics import (  # noqa: E402
    aggregate_metrics_with_ci,
    compute_metrics,
)
from evaluation.run_code2lora_gru_v2_eval import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_TARGET_MODULES,
    _generate_batch,
    _get_bos_id,
    _load_v2_gru_ckpt,
    _prepare_prefix_ids,
    _summarize,
    _write_suite_json,
)


# ---------------------------------------------------------------------------
# Anchor table (mirrors hypernetwork/train_code2lora_gru_anchor.py)
# ---------------------------------------------------------------------------

def _select_anchor_per_repo(snap_rows: List[SnapshotRow], *, latest: bool
                            ) -> Dict[str, Tuple[str, np.ndarray]]:
    out: Dict[str, Tuple[str, np.ndarray, int]] = {}
    for r in snap_rows:
        prev = out.get(r.repo_id)
        if prev is None:
            out[r.repo_id] = (r.commit_sha, r.repo_state_embedding, r.commit_index)
            continue
        if latest and r.commit_index > prev[2]:
            out[r.repo_id] = (r.commit_sha, r.repo_state_embedding, r.commit_index)
    return {k: (v[0], v[1]) for k, v in out.items()}


def build_anchor_table(snapshots_dir: Path, split: str,
                       train_anchors: Optional[Dict[str, Tuple[str, np.ndarray]]] = None,
                       ) -> Dict[str, Tuple[str, np.ndarray]]:
    if split in ("ir_val", "ir_test"):
        if train_anchors is None:
            train_rows = load_snapshot_rows(snapshots_dir / "commits" / "train.parquet")
            train_anchors = _select_anchor_per_repo(train_rows, latest=False)
        return train_anchors
    parquet = snapshots_dir / "commits" / f"{split}.parquet"
    rows = load_snapshot_rows(parquet)
    return _select_anchor_per_repo(rows, latest=split.startswith("cr_"))


# ---------------------------------------------------------------------------
# Main eval loop (anchor-only variant)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite_anchor(
    *,
    suite_name: str,
    commits_parquet: Path,
    qna_parquet: Path,
    anchor_by_repo: Dict[str, Tuple[str, np.ndarray]],
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
    predictions_out: Optional[Path] = None,
    restrict_keys: Optional[set] = None,
    method_label: str = "code2lora_gru_anchor",
) -> Dict[str, Any]:
    """Anchor-only per-commit eval. Same protocol as the v2 GRU runner, but
    every GRU step receives the per-repo anchor embedding instead of
    per-commit diff embeddings."""
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
            "test_file": getattr(qr, "test_file", ""),
            "lineno": int(getattr(qr, "lineno", 0) or 0),
        })
    preds_fh = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        preds_fh = open(predictions_out, "a")
    print(f"[suite {suite_name}] {len(qna_rows)} qnas, "
          f"{len(qnas_by_key)} (repo, commit) keys", flush=True)

    n_total_groups = 0
    n_missing_anchor = 0
    for repo, rows in rows_by_repo.items():
        if repo not in anchor_by_repo:
            n_missing_anchor += 1
            continue
        for row in rows:
            if (in_repo_splits_to_score is not None and
                    row.in_repo_split not in in_repo_splits_to_score):
                continue
            if (repo, row.commit_sha) in qnas_by_key:
                n_total_groups += 1
    print(f"[suite {suite_name}] expected scored groups in this shard: "
          f"{n_total_groups}  (repos missing anchor: {n_missing_anchor})",
          flush=True)

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
        anchor = anchor_by_repo.get(repo_id)
        if anchor is None:
            continue
        _anchor_sha, anchor_emb_np = anchor
        anchor_emb = torch.from_numpy(anchor_emb_np).to(device).unsqueeze(0)
        h = gru.init_hidden(anchor_emb)

        for row in rows:
            h = gru.step(anchor_emb, h)
            score_this = True
            if (in_repo_splits_to_score is not None and
                    row.in_repo_split not in in_repo_splits_to_score):
                score_this = False
            if (restrict_keys is not None and
                    (row.repo_id, row.commit_sha) not in restrict_keys):
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
            qna_pos_offset = 0
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
                for j, (p, pred) in enumerate(zip(batch_pairs, preds)):
                    m = compute_metrics(pred, p["target"])
                    em = 1.0 if m["exact_match"] else 0.0
                    ed = float(m["edit_similarity"])
                    cb = float(m["code_bleu"])
                    commit_samples.append((em, ed, cb))
                    all_samples.append((em, ed, cb))
                    if preds_fh is not None:
                        preds_fh.write(json.dumps({
                            "method": method_label,
                            "repo_id": row.repo_id,
                            "commit_sha": row.commit_sha,
                            "commit_index": int(row.commit_index),
                            "qna_pos": qna_pos_offset + j,
                            "test_file": p.get("test_file", ""),
                            "lineno": p.get("lineno", 0),
                            "prefix": p["prefix"],
                            "augmented_prompt": p["prefix"],
                            "target": p["target"],
                            "prediction": pred,
                            "exact_match": em,
                            "edit_similarity": ed,
                            "code_bleu": cb,
                        }, ensure_ascii=False) + "\n")
                        preds_fh.flush()
                qna_pos_offset += len(batch_pairs)
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

    summary = _summarize(all_samples, bootstrap=bootstrap)
    summary["suite"] = suite_name
    summary["n_qnas"] = len(all_samples)
    summary["n_scored_commits"] = len(per_commit_records)
    summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})
    summary["method"] = method_label
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
                         "by train_code2lora_gru_anchor.py (also accepts a "
                         "v2 GRU checkpoint -- it just won't read the diff "
                         "embeddings).")
    ap.add_argument("--commits-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2",
                    help="Source of chronological per-commit lists.")
    ap.add_argument("--snapshots-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf",
                    help="Source of anchor repo_state_embeddings and QnAs.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--suite", required=True,
                    choices=["ir_val", "ir_test", "cr_val", "cr_test"])
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--qnas-per-commit-limit", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--predictions-out", type=str, default=None)
    ap.add_argument("--restrict-keys", type=str, default=None)
    ap.add_argument("--method-label", default="code2lora_gru_anchor")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = (f"_shard{args.shard_i}of{args.num_shards}"
                    if args.num_shards > 1 else "")
    out_path = out_dir / f"gru_anchor_{args.suite}{shard_suffix}.json"

    base_model, gru, head, specs, tok = _load_v2_gru_ckpt(
        Path(args.checkpoint), args.base_model, args.target_modules,
        device=device,
    )

    if args.suite.startswith("cr_"):
        in_repo_splits = None
    elif args.suite == "ir_val":
        in_repo_splits = ["val"]
    else:
        in_repo_splits = ["test"]

    snapshots_dir = Path(args.snapshots_dir)
    print(f"[anchor] building anchor table for suite={args.suite}", flush=True)
    anchor_by_repo = build_anchor_table(snapshots_dir, args.suite)
    print(f"[anchor] {len(anchor_by_repo)} repos with an anchor", flush=True)

    commits_parquet = Path(args.commits_dir) / "commits" / f"{args.suite}.parquet"
    qna_parquet = snapshots_dir / "qna" / f"{args.suite}.parquet"
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

    restrict_keys = None
    if args.restrict_keys:
        rk: set = set()
        for line in Path(args.restrict_keys).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rk.add((rec["repo_id"], rec["commit_sha"]))
        restrict_keys = rk
        print(f"[restrict] loaded {len(restrict_keys)} (repo, commit) keys",
              flush=True)

    evaluate_suite_anchor(
        suite_name=args.suite,
        commits_parquet=commits_parquet,
        qna_parquet=qna_parquet,
        anchor_by_repo=anchor_by_repo,
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
        predictions_out=(Path(args.predictions_out)
                         if args.predictions_out else None),
        restrict_keys=restrict_keys,
        method_label=args.method_label,
    )


if __name__ == "__main__":
    main()
