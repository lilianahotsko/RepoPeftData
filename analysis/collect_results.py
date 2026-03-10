#!/usr/bin/env python3
"""
Quickly collect and display all results from $SCRATCH/BASELINES.
Prints a formatted table suitable for pasting into EXPERIMENT_LOG.md.

Usage:
    python analysis/collect_results.py
    python analysis/collect_results.py --results-dir /path/to/results
"""

import argparse
import json
import os
from pathlib import Path


METHOD_ORDER = [
    ("pretrained_cr_test", "pretrained_ir_test", "Pretrained"),
    ("rag_top3_cr_test", "rag_top3_ir_test", "RAG (k=3)"),
    ("icl_3shot_cr_test", "icl_3shot_ir_test", "ICL (3-shot)"),
    ("oracle_context_cr_test", "oracle_context_ir_test", "Oracle Context"),
    ("fft_no_oracle_cr_test", "fft_no_oracle_ir_test", "FFT"),
    ("single_lora_no_oracle_cr_test", "single_lora_no_oracle_ir_test", "Single LoRA r=64"),
    (None, "per_repo_lora_no_oracle_ir_test", "Per-repo LoRA"),
    ("hypernet_no_oracle_cr_test", "hypernet_no_oracle_ir_test", "Code2LoRA (Direct)"),
    ("hypernet_paw_no_oracle_cr_test", "hypernet_paw_no_oracle_ir_test", "Code2LoRA (PAW)"),
    ("fft_oracle_cr_test", "fft_oracle_ir_test", "FFT + Oracle"),
    ("single_lora_oracle_cr_test", "single_lora_oracle_ir_test", "sLoRA + Oracle"),
    ("hypernet_oracle_cr_test", "hypernet_oracle_ir_test", "Code2LoRA + Oracle"),
    ("hypernet_paw_oracle_cr_test", "hypernet_paw_oracle_ir_test", "Code2LoRA PAW + Oracle"),
]

SCALE_ORDER = [
    ("hypernet_scale_50_cr_test", "50 repos"),
    ("hypernet_scale_100_cr_test", "100 repos"),
    ("hypernet_scale_200_cr_test", "200 repos"),
    ("hypernet_no_oracle_cr_test", "409 repos (full)"),
]


def load_result(results_dir: Path, name: str) -> dict | None:
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fmt(val, pct=False):
    if val is None:
        return "—"
    if pct:
        return f"{val:.2f}%"
    return f"{val:.4f}"


def main():
    default_dir = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "BASELINES",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=default_dir)
    args = ap.parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()

    print(f"Results directory: {results_dir}\n")

    header = f"{'Method':<25} | {'CR EM':>8} | {'CR ES':>7} | {'CR CB':>7} | {'IR EM':>8} | {'IR ES':>7} | {'IR CB':>7} | {'N(CR)':>6} | {'N(IR)':>6}"
    print(header)
    print("-" * len(header))

    for cr_key, ir_key, label in METHOD_ORDER:
        cr = load_result(results_dir, cr_key) if cr_key else None
        ir = load_result(results_dir, ir_key) if ir_key else None

        cr_em = cr.get("exact_match_pct") if cr else None
        cr_es = cr.get("edit_similarity") if cr else None
        cr_cb = cr.get("code_bleu") if cr else None
        cr_n = cr.get("n") if cr else None

        ir_em = ir.get("exact_match_pct") if ir else None
        ir_es = ir.get("edit_similarity") if ir else None
        ir_cb = ir.get("code_bleu") if ir else None
        ir_n = ir.get("n") if ir else None

        print(f"{label:<25} | {fmt(cr_em, True):>8} | {fmt(cr_es):>7} | {fmt(cr_cb):>7} | {fmt(ir_em, True):>8} | {fmt(ir_es):>7} | {fmt(ir_cb):>7} | {fmt(cr_n):>6} | {fmt(ir_n):>6}")

    print()
    print("=== Scaling Experiments (CR Test) ===")
    s_header = f"{'Training Data':<20} | {'CR EM':>8} | {'CR ES':>7} | {'CR CB':>7}"
    print(s_header)
    print("-" * len(s_header))
    for key, label in SCALE_ORDER:
        r = load_result(results_dir, key)
        em = r.get("exact_match_pct") if r else None
        es = r.get("edit_similarity") if r else None
        cb = r.get("code_bleu") if r else None
        print(f"{label:<20} | {fmt(em, True):>8} | {fmt(es):>7} | {fmt(cb):>7}")


if __name__ == "__main__":
    main()
