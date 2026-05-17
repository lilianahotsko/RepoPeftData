#!/usr/bin/env python3
"""Unified v2 baseline evaluator -- scores **pretrained**, **FFT**, and
**SLoRA** on exactly the same (repo, commit, qna) triples as the v2
Code2LoRA trainers (``hypernetwork/train_code2lora_{static,gru}_v2.py``).

The evaluator pulls QnAs from a v2 parquet suite -- by default the
*Dataset A* (GRU-aligned) parquets at
``$SCRATCH/REPO_DATASET/commit_parquet_hf_v2/qna/<suite>.parquet`` -- and
walks (repo_id, commit_sha) groups, scoring every QnA at every commit.
For each suite it dumps a per-commit JSON suitable for decay-curve plotting:

    {
        "summary": {"suite": ..., "n_qnas": ..., "exact_match": ..., ...,
                    "exact_match_ci": [lo, hi], ...},
        "per_commit": [
            {"repo_id": ..., "commit_sha": ..., "commit_index": ...,
             "n_qnas": ..., "exact_match": ..., "edit_similarity": ...,
             "code_bleu": ...},
            ...
        ]
    }

Both v2 Code2LoRA trainers emit the same shape so plots overlay cleanly.

Methods
-------
* ``pretrained``  No adaptation -- raw Qwen2.5-Coder-1.5B.
* ``fft``         Full fine-tune; ``--ckpt`` must be a HF model directory
                  (output of ``baselines/finetuned/train_fft_v2.py``).
* ``slora``       Single LoRA shared across repos; ``--ckpt`` must contain
                  a PEFT adapter (output of
                  ``baselines/single_lora/train_slora_v2.py``).

Truncation policy matches the v2 Code2LoRA trainers: **left-truncate, left-pad**
so the assertion-adjacent code is preserved.

Usage::

    sbatch scripts/slurm/eval_baselines_v2.sh   # default = pretrained, all 4 suites
    METHOD=fft CKPT=$CKPT_DIR/FFT_V2/checkpoint-final sbatch ...
    METHOD=slora CKPT=$CKPT_DIR/SLORA_V2/adapter sbatch ...
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

from code2lora_core import load_qna_rows  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate_metrics_with_ci,
    compute_metrics,
    format_ci,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_QNA_DIR = "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/qna"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_method(method: str, base_model_name: str, ckpt: Optional[Path],
                          device: torch.device,
                          dtype: torch.dtype = torch.bfloat16):
    """Return (tokenizer, model) ready for inference, all params frozen."""
    if method == "fft":
        if ckpt is None:
            raise SystemExit("--ckpt is required for method=fft")
        print(f"[load] FFT -> AutoModelForCausalLM.from_pretrained({ckpt})",
              flush=True)
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    elif method == "slora":
        if ckpt is None:
            raise SystemExit("--ckpt is required for method=slora")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit("peft must be installed for method=slora") from e
        print(f"[load] SLoRA: base={base_model_name}, adapter={ckpt}",
              flush=True)
        tok = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        if hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
    else:  # pretrained
        print(f"[load] Pretrained {base_model_name}", flush=True)
        tok = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


# ---------------------------------------------------------------------------
# Tokenization (left-truncate, left-pad) -- mirrors v2 Code2LoRA trainers
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


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_batch(model, tokenizer, device, input_ids_list: List[List[int]],
                    max_new_tokens: int) -> List[str]:
    """Left-pad a batch, generate, decode only the new tokens per sample."""
    L = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    bs = len(input_ids_list)
    input_ids = torch.full((bs, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, L), dtype=torch.long, device=device)
    lens = []
    for i, ids in enumerate(input_ids_list):
        n = len(ids)
        input_ids[i, L - n:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, L - n:] = 1
        lens.append(n)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    # Strip the prefix off each row; whatever's left is the prediction.
    decoded: List[str] = []
    for i in range(bs):
        gen = out[i, L:].tolist()
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True))
    return decoded


def score_suite(
    *,
    qna_path: Path,
    suite_name: str,
    model,
    tokenizer,
    device: torch.device,
    max_input_tokens: int = 4096,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    repo_limit: int = 0,
    qnas_per_commit_limit: int = 0,
    out_path: Path,
    bootstrap: int = 5000,
) -> Dict[str, Any]:
    """Walk one v2 QnA parquet, group by (repo, commit), generate, score."""
    print(f"\n[suite {suite_name}] loading {qna_path} ...", flush=True)
    rows = load_qna_rows(qna_path)
    # Group by (repo_id, commit_sha) preserving commit_index.
    groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"commit_index": -1, "pairs": []}
    )
    for qr in rows:
        key = (qr.repo_id, qr.commit_sha)
        g = groups[key]
        g["commit_index"] = int(qr.commit_index)
        g["pairs"].append({"prefix": qr.prefix, "target": qr.target})
    print(f"[suite {suite_name}] {len(rows):,} qnas across "
          f"{len(groups):,} (repo, commit) groups", flush=True)

    if repo_limit:
        # Keep only the first `repo_limit` repos (sorted by repo_id).
        repos_kept = sorted({r for (r, _) in groups.keys()})[:repo_limit]
        groups = {k: v for k, v in groups.items() if k[0] in repos_kept}
        print(f"[suite {suite_name}] limited to {repo_limit} repos -> "
              f"{len(groups)} (repo, commit) groups", flush=True)

    bos_id = _get_bos_id(tokenizer)

    # Sort groups by (repo, commit_index) for deterministic per-commit output.
    group_keys = sorted(groups.keys(),
                        key=lambda k: (k[0], groups[k]["commit_index"]))

    all_samples: List[Tuple[float, float, float]] = []
    per_commit_records: List[Dict[str, Any]] = []
    t0 = time.time()
    n_done = 0
    for (repo_id, commit_sha) in group_keys:
        g = groups[(repo_id, commit_sha)]
        pairs = g["pairs"]
        if qnas_per_commit_limit and len(pairs) > qnas_per_commit_limit:
            pairs = pairs[:qnas_per_commit_limit]
        commit_samples: List[Tuple[float, float, float]] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = [
                _prepare_prefix_ids(tokenizer, p["prefix"],
                                    max_input_tokens, bos_id)
                for p in batch_pairs
            ]
            preds = _generate_batch(
                model, tokenizer, device, inputs,
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
            em_m = sum(s[0] for s in commit_samples) / n_c
            ed_m = sum(s[1] for s in commit_samples) / n_c
            cb_m = sum(s[2] for s in commit_samples) / n_c
            per_commit_records.append({
                "repo_id": repo_id,
                "commit_sha": commit_sha,
                "commit_index": g["commit_index"],
                "n_qnas": n_c,
                "exact_match": em_m,
                "edit_similarity": ed_m,
                "code_bleu": cb_m,
            })
        n_done += 1
        if n_done % 50 == 0 or n_done == len(group_keys):
            elapsed = (time.time() - t0) / 60
            done_em = sum(s[0] for s in all_samples) / max(len(all_samples), 1)
            print(f"  [suite {suite_name}] {n_done}/{len(group_keys)} groups "
                  f"({len(all_samples):,} qnas) "
                  f"running_EM={done_em:.4f} elapsed={elapsed:.1f}m",
                  flush=True)
            # Incremental write so the file is usable even if the job times out.
            _write_suite_json(out_path, suite_name, per_commit_records,
                              all_samples, bootstrap=bootstrap, finalized=False)

    summary = _summarize(all_samples, bootstrap=bootstrap)
    summary["suite"] = suite_name
    summary["n_qnas"] = len(all_samples)
    summary["n_scored_commits"] = len(per_commit_records)
    summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})

    _write_suite_json(out_path, suite_name, per_commit_records, all_samples,
                      bootstrap=bootstrap, finalized=True, summary=summary)
    print(f"[suite {suite_name}] DONE  EM={summary['exact_match']:.4f}  "
          f"EditSim={summary['edit_similarity']:.4f}  "
          f"BLEU={summary['code_bleu']:.4f}  ({len(all_samples):,} qnas, "
          f"{len(per_commit_records):,} commits)", flush=True)
    return summary


def _summarize(samples: List[Tuple[float, float, float]],
               bootstrap: int = 5000) -> Dict[str, Any]:
    if not samples:
        return {"n_qnas": 0, "exact_match": 0.0, "edit_similarity": 0.0,
                "code_bleu": 0.0}
    metric_dicts = [
        {"exact_match": bool(em), "edit_similarity": ed, "code_bleu": cb}
        for (em, ed, cb) in samples
    ]
    agg = aggregate_metrics_with_ci(metric_dicts, n_resamples=int(bootstrap))
    out: Dict[str, Any] = {}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            # ``v`` is the per-metric dict returned by bootstrap_ci:
            # {mean, low, high, n, n_resamples, ci (=conf level, not bounds)}.
            out[k] = float(v["mean"])
            out[f"{k}_ci"] = [float(v.get("low", 0.0)),
                              float(v.get("high", 0.0))]
            out[f"{k}_n"] = int(v.get("n", 0))
        else:
            out[k] = v
    return out


def _write_suite_json(out_path: Path, suite_name: str,
                      per_commit_records: List[Dict[str, Any]],
                      all_samples: List[Tuple[float, float, float]],
                      *,
                      bootstrap: int,
                      finalized: bool,
                      summary: Optional[Dict[str, Any]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary is None:
        summary = _summarize(all_samples, bootstrap=bootstrap)
        summary["suite"] = suite_name
        summary["n_qnas"] = len(all_samples)
        summary["n_scored_commits"] = len(per_commit_records)
        summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})
    payload = {
        "finalized": finalized,
        "summary": summary,
        "per_commit": per_commit_records,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["pretrained", "fft", "slora"])
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to FFT checkpoint dir (method=fft) or "
                         "SLoRA adapter dir (method=slora). Unused for pretrained.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="HF id of the base model -- used as-is for pretrained "
                         "and SLoRA; for FFT we load directly from --ckpt.")
    ap.add_argument("--qna-dir", default=DEFAULT_QNA_DIR,
                    help="Directory with v2 qna parquets (qna/<suite>.parquet).")
    ap.add_argument("--suites", nargs="+",
                    default=["ir_val", "ir_test", "cr_val", "cr_test"],
                    choices=["ir_val", "ir_test", "cr_val", "cr_test"])
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--max-input-tokens", type=int, default=4096,
                    help="Left-truncate prefix to keep last N tokens; matches "
                         "v2 Code2LoRA trainers' --max-seq-len (minus target budget).")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repo-limit", type=int, default=0,
                    help="Limit number of repos per suite (debug; 0 = all).")
    ap.add_argument("--qnas-per-commit-limit", type=int, default=8,
                    help="Cap QnAs per (repo, commit). Matches the GRU "
                         "trainer's --max-qna-per-commit eval cap so all "
                         "models score the same number of triples per "
                         "commit. 0 = no cap.")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model_for_method(
        args.method, args.base_model, ckpt, device=device,
    )

    suite_summaries: Dict[str, Any] = {}
    for suite in args.suites:
        qna_path = Path(args.qna_dir) / f"{suite}.parquet"
        if not qna_path.exists():
            print(f"[suite {suite}] MISSING parquet {qna_path}, skipping",
                  flush=True)
            continue
        out_path = out_dir / f"baseline_{args.method}_{suite}.json"
        s = score_suite(
            qna_path=qna_path, suite_name=suite,
            model=model, tokenizer=tok, device=device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            repo_limit=args.repo_limit,
            qnas_per_commit_limit=args.qnas_per_commit_limit,
            out_path=out_path,
            bootstrap=args.bootstrap,
        )
        suite_summaries[suite] = s

    # Final cross-suite summary file.
    summary_path = out_dir / f"baseline_{args.method}_summary.json"
    summary_path.write_text(json.dumps({
        "method": args.method,
        "base_model": args.base_model,
        "ckpt": str(ckpt) if ckpt else None,
        "qna_dir": str(args.qna_dir),
        "args": vars(args),
        "suites": suite_summaries,
    }, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
