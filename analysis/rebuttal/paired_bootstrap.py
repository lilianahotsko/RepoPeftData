#!/usr/bin/env python3
"""Paired bootstrap significance test (scipy) for the headline comparisons.

Implements the exact recipe requested: scipy.stats.bootstrap with paired=True,
statistic = mean(technique - baseline), one-sided ("greater"), percentile method,
n_resamples=100_000. Reports observed improvement, one-sided p-value, and the
95% lower confidence bound for Code2LoRA-Static vs each baseline on the static
track (per-instance EM, aligned by (repo, expected)).

Small stored-prediction reanalysis; safe on the login node.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

METHODS = {
    "Pretrained": "pretrained",
    "RAG (k=3)": "rag_top3",
    "DRC": "oracle_context_v2",
    "FFT": "fft_no_oracle",
    "sLoRA": "single_lora_r16_tuned",
    "Code2LoRA-Static": "hypernet_no_oracle",
}
REFERENCE = "Pretrained"
TECHNIQUE = "Code2LoRA-Static"
BASELINES = ["FFT", "DRC", "sLoRA", "RAG (k=3)", "Pretrained"]


def _as_bool(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def load_method(baselines_dir: Path, stem: str, split: str):
    path = baselines_dir / f"{stem}_{split}.json"
    if not path.exists():
        return None
    entries = json.load(open(path))["entries"]
    return [
        {"repo": e["repo"], "expected": e["expected"],
         "em": 1.0 if _as_bool(e["exact_match"]) else 0.0}
        for e in entries
    ]


def aligned_em(baselines_dir: Path, split: str):
    data = {lbl: load_method(baselines_dir, stem, split) for lbl, stem in METHODS.items()}
    data = {k: v for k, v in data.items() if v is not None}
    ref = data[REFERENCE]
    ref_key = [(r["repo"], r["expected"]) for r in ref]
    em = {}
    for lbl, rows in data.items():
        if len(rows) == len(ref) and [(r["repo"], r["expected"]) for r in rows] == ref_key:
            em[lbl] = np.array([r["em"] for r in rows], dtype=float)
        else:
            print(f"  [skip] {lbl}: misaligned ({len(rows)} vs {len(ref)})", file=sys.stderr)
    return em, len(ref)


def paired_bootstrap(technique: np.ndarray, baseline: np.ndarray):
    res = bootstrap(
        (technique, baseline),
        lambda t, b, axis: np.mean(t - b, axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=100_000,
        batch=2_000,  # bound peak memory (batch x n floats) for the login node
        confidence_level=0.95,
        alternative="greater",
        method="percentile",
        rng=np.random.default_rng(42),
    )
    diffs = res.bootstrap_distribution
    p_value = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
    return {
        "observed_pp": float(np.mean(technique - baseline) * 100),
        "p_value": float(p_value),
        "low_pp": float(res.confidence_interval.low * 100),
        "significant": bool(p_value < 0.05),
    }


def main():
    baselines_dir = Path(os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES"))
    lines = ["# Paired bootstrap significance (static track)\n",
             "scipy.stats.bootstrap, paired, one-sided (greater), percentile, "
             "n_resamples=100,000, seed=42. Statistic = mean(Code2LoRA-Static − baseline) EM.\n"]
    results = {}
    for split in ["cr_test", "ir_test"]:
        em, n = aligned_em(baselines_dir, split)
        tech = em[TECHNIQUE]
        lines.append(f"\n## {split} (n={n})\n")
        lines.append("| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) | significant |")
        lines.append("|---|---|---|---|---|")
        results[split] = {}
        for b in BASELINES:
            if b not in em:
                continue
            r = paired_bootstrap(tech, em[b])
            results[split][b] = r
            pstr = "<1e-5" if r["p_value"] < 1e-5 else f"{r['p_value']:.2e}"
            lines.append(f"| {TECHNIQUE} vs {b} | {r['observed_pp']:+.1f} | {pstr} | "
                         f"{r['low_pp']:+.1f} | {'yes' if r['significant'] else 'no'} |")
    md = "\n".join(lines) + "\n"
    out = Path(__file__).resolve().parent / "outputs" / "paired_bootstrap.md"
    out.write_text(md)
    (out.with_suffix(".json")).write_text(json.dumps(results, indent=2))
    print(md)


if __name__ == "__main__":
    main()
