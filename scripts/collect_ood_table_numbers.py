#!/usr/bin/env python3
"""Aggregate per-shard OOD eval JSONs into a single Table-3-style row per
Table 2 method (Pretrained, RAG-256, DRC-256, sLoRA, C2L-static, C2L-GRU).

Reads the *_shardX_ofN.json artifacts dumped by run_baselines_v2.py and
run_code2lora_static_v2_eval.py for the ood_test suite, plus the existing
C2L-GRU OOD bench JSON, computes qna-weighted EM / EditSim / CodeBLEU and
95% commit-clustered bootstrap CIs, and prints a LaTeX-ready row block.

Use after the OOD eval jobs (14749499..14749852, 14749828) finish:
    python scripts/collect_ood_table_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


CKPT_DIR = Path("/scratch/lhotsko/TRAINING_CHECKPOINTS")

# Method -> directory containing *_shardX_ofN.json files
METHODS = {
    "pretrained": {
        "dir": CKPT_DIR / "BASELINES_V2" / "pretrained_h100_v2_ood",
        "glob": "baseline_pretrained_ood_test_shard*.json",
        "label": "Pretrained (full prefix)",
    },
    "rag256": {
        "dir": CKPT_DIR / "BASELINES_V2" / "rag_h100_v2_prefix256_ood",
        "glob": "baseline_rag_ood_test_shard*.json",
        "label": "RAG ($k$=3, 256 prefix + chunks)",
    },
    "drc256": {
        "dir": CKPT_DIR / "BASELINES_V2" / "drc_h100_v2_prefix256_ood",
        "glob": "baseline_drc_ood_test_shard*.json",
        "label": "DRC (256 prefix + oracle)",
    },
    "slora_anchor": {
        "dir": CKPT_DIR / "BASELINES_V2" / "slora_h100_v2_anchor_ood",
        "glob": "baseline_slora_ood_test_shard*.json",
        "label": "Single LoRA (sLoRA)",
    },
    "doc2lora": {
        "dir": CKPT_DIR / "BASELINES_V2" / "doc2lora_h100_v2_ood",
        "glob": "baseline_doc2lora_ood_test_shard*.json",
        "label": "Doc2LoRA (D2L)",
    },
    "c2l_static": {
        "dir": CKPT_DIR / "CODE2LORA_STATIC_EVAL_V2" / "h100_v2_ood_run5",
        "glob": "static_v2_ood_test_shard*.json",
        "label": "Code2LoRA (direct projection)",
    },
    # NEW (May-2026 rerun, job 14768286): v2 GRU evaluated on OOD with the
    # SAME cap=8 / streaming protocol used by every other Table-3 baseline.
    # Replaces the older uncapped run that produced 78.9% on ~180k qnas.
    "c2l_gru": {
        "dir": CKPT_DIR / "CODE2LORA_GRU_EVAL_V2" / "h100_v2_gru_3ep_ood",
        "glob": "gru_v2_ood_test_shard*.json",
        "label": "Code2LoRA-GRU (streaming)",
    },
}


def _load_per_commit(file_paths: List[Path]):
    """Concat per_commit lists from shard JSONs."""
    rows = []
    for fp in sorted(file_paths):
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            print(f"  skip {fp}: {e}", file=sys.stderr)
            continue
        for r in d.get("per_commit", []):
            rows.append({
                "repo_id": r.get("repo_id"),
                "commit_sha": r.get("commit_sha"),
                "commit_index": int(r.get("commit_index", -1)),
                "n_qnas": int(r.get("n_qnas", 0)),
                "exact_match": float(r.get("exact_match", 0.0)),
                "edit_similarity": float(r.get("edit_similarity", 0.0)),
                "code_bleu": float(r.get("code_bleu", 0.0)),
            })
    return rows


def _weighted_mean(rows, key):
    n = sum(r["n_qnas"] for r in rows)
    if n == 0:
        return 0.0
    return sum(r["n_qnas"] * r[key] for r in rows) / n


def _bootstrap_repo_clustered(rows, key, n_iters=5000, seed=0):
    """95% bootstrap CI clustered by repo_id."""
    by_repo: Dict[str, List[Dict]] = {}
    for r in rows:
        by_repo.setdefault(r["repo_id"], []).append(r)
    repos = list(by_repo.keys())
    if not repos:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_iters):
        idx = rng.integers(0, len(repos), size=len(repos))
        boot_rows = []
        for i in idx:
            boot_rows.extend(by_repo[repos[i]])
        samples.append(_weighted_mean(boot_rows, key))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_method(method_id: str, info: dict) -> Dict:
    d = info["dir"]
    files = list(d.glob(info["glob"]))
    if not files:
        print(f"  [{method_id}] no shard files yet under {d}/{info['glob']}",
              file=sys.stderr)
        return None
    rows = _load_per_commit(files)
    n_qnas = sum(r["n_qnas"] for r in rows)
    n_repos = len({r["repo_id"] for r in rows})
    n_commits = len({(r["repo_id"], r["commit_sha"]) for r in rows})
    if n_qnas == 0:
        print(f"  [{method_id}] zero qnas across {len(files)} shards",
              file=sys.stderr)
        return None
    em = _weighted_mean(rows, "exact_match")
    es = _weighted_mean(rows, "edit_similarity")
    cb = _weighted_mean(rows, "code_bleu")
    em_lo, em_hi = _bootstrap_repo_clustered(rows, "exact_match")
    return {
        "method_id": method_id,
        "label": info["label"],
        "n_shards": len(files),
        "n_qnas": n_qnas,
        "n_repos": n_repos,
        "n_commits": n_commits,
        "exact_match": em,
        "edit_similarity": es,
        "code_bleu": cb,
        "em_ci": [em_lo, em_hi] if em_lo is not None else None,
    }


def main() -> None:
    print("\n=== OOD Table-3 row aggregation ===\n")
    all_results = []
    for mid, info in METHODS.items():
        res = summarize_method(mid, info)
        if res:
            all_results.append(res)

    print(f"\n{'Method':40s}  {'EM%':>6s}  {'EditSim':>8s}  {'CodeBLEU':>9s}  "
          f"{'95% CI':>18s}  {'n_qnas':>8s}  {'n_repos':>8s}")
    print("-" * 110)
    for r in all_results:
        em = r["exact_match"]
        if em < 1.0:
            em = em * 100  # convert fraction -> percent
        ci = r["em_ci"]
        ci_str = f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}]" if ci and ci[0] < 1.0 else (
            f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "n/a"
        )
        print(f"{r['label']:40s}  {em:6.2f}  {r['edit_similarity']:8.3f}  "
              f"{r['code_bleu']:9.3f}  {ci_str:>18s}  "
              f"{r['n_qnas']:>8d}  {r['n_repos']:>8d}")
    print()

    # LaTeX-friendly block
    print("\n=== LaTeX-ready rows (paste into tab:ood_results) ===\n")
    for r in all_results:
        em = r["exact_match"]
        if em < 1.0:
            em = em * 100
        es = r["edit_similarity"]
        cb = r["code_bleu"]
        if es > 1.0:
            es = es / 100  # in case it's already percent-scaled
        if cb > 1.0:
            cb = cb / 100
        print(f"  {r['label']} & {em:.1f} & {es:.3f} & {cb:.3f}  \\\\")
    print()


if __name__ == "__main__":
    main()
