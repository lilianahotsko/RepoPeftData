#!/usr/bin/env python3
"""Memorization / leakage analysis (no GPU)  [covers yeRc-W4].

Tests whether Code2LoRA's gains reflect memorization of training targets rather
than genuine adaptation. We build the multiset of training targets and, for each
test split, measure:

  1. Novel-target rate: fraction of test targets whose (normalized) string never
     appears as a training target.
  2. EM on the novel-target subset vs. the seen-target subset, per method. High
     EM on strings never seen in training is generalization, not memorization.
  3. Among each method's CORRECT predictions, the fraction whose target string was
     seen in training. If Code2LoRA's correct answers are no more "seen" than the
     never-trained pretrained baseline's, memorization is not the driver.

CR test uses repositories held out entirely at training (the cleanest test);
IR test uses held-out instances within training repositories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.rebuttal.reanalysis import METHODS, load_method  # noqa: E402
from evaluation.metrics import normalize_for_match  # noqa: E402


def load_train_targets(parquet_path: Path) -> set[str]:
    seen = set()
    pf = pq.ParquetFile(str(parquet_path))
    for i in range(pf.num_row_groups):
        col = pf.read_row_group(i, columns=["target"]).column("target").to_pylist()
        for t in col:
            if t is not None:
                seen.add(normalize_for_match(str(t)))
    return seen


def analyze(baselines_dir: Path, train_targets: set[str], split: str):
    labels = ["Pretrained", "FFT", "sLoRA", "Code2LoRA-Static"]
    data = {}
    for lbl in labels:
        rows = load_method(baselines_dir, METHODS[lbl], split)
        if rows is not None:
            data[lbl] = rows
    if "Pretrained" not in data:
        return None
    ref = data["Pretrained"]
    for lbl, rows in list(data.items()):
        if len(rows) != len(ref) or [(r["repo"], r["expected"]) for r in rows] != [(r["repo"], r["expected"]) for r in ref]:
            print(f"  [skip] {lbl}: misaligned", file=sys.stderr)
            data.pop(lbl)

    seen_mask = np.array([normalize_for_match(r["expected"]) in train_targets for r in ref])
    novel_mask = ~seen_mask
    em = {lbl: np.array([r["em"] for r in rows]) for lbl, rows in data.items()}

    out = {
        "split": split,
        "n": len(ref),
        "novel_target_rate_pct": float(novel_mask.mean() * 100),
        "n_novel": int(novel_mask.sum()),
        "per_method": {},
    }
    for lbl in data:
        e = em[lbl]
        correct = e == 1.0
        n_corr = int(correct.sum())
        out["per_method"][lbl] = {
            "em_on_novel_targets_pct": float(e[novel_mask].mean() * 100) if novel_mask.any() else None,
            "em_on_seen_targets_pct": float(e[seen_mask].mean() * 100) if seen_mask.any() else None,
            # of the instances this method gets right, what fraction had a target seen in training
            "frac_correct_that_are_seen_pct": float(seen_mask[correct].mean() * 100) if n_corr else None,
        }
    return out


def to_md(results: dict) -> str:
    lines = ["# Memorization / leakage analysis (yeRc-W4)\n",
             "Training-target set built from the static-track train split.\n"]
    for split, res in results.items():
        if res is None:
            continue
        lines.append(f"\n## {split}  (n={res['n']})\n")
        lines.append(f"Novel-target rate (target string never seen in training): "
                     f"**{res['novel_target_rate_pct']:.1f}%** ({res['n_novel']}/{res['n']})\n")
        lines.append("| Method | EM on novel targets % | EM on seen targets % | of correct preds, % seen in train |")
        lines.append("|---|---|---|---|")
        for lbl, v in res["per_method"].items():
            def f(x):
                return f"{x:.1f}" if x is not None else "-"
            lines.append(f"| {lbl} | {f(v['em_on_novel_targets_pct'])} | {f(v['em_on_seen_targets_pct'])} | "
                         f"{f(v['frac_correct_that_are_seen_pct'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-dir", default=os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES"))
    ap.add_argument("--train-parquet",
                    default=os.environ.get("TRAIN_PARQUET",
                                           "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/qna/train.parquet"))
    ap.add_argument("--splits", nargs="+", default=["cr_test", "ir_test"])
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    print(f"[leakage] loading train targets from {args.train_parquet}", file=sys.stderr)
    train_targets = load_train_targets(Path(args.train_parquet))
    print(f"[leakage] {len(train_targets)} unique normalized train targets", file=sys.stderr)

    out = {"n_unique_train_targets": len(train_targets)}
    for sp in args.splits:
        print(f"[leakage] {sp} ...", file=sys.stderr)
        out[sp] = analyze(Path(args.baselines_dir), train_targets, sp)
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "leakage_analysis.json").write_text(json.dumps(out, indent=2))
    (od / "leakage_analysis.md").write_text(to_md({k: v for k, v in out.items() if k != "n_unique_train_targets"}))
    print(f"[leakage] wrote {od/'leakage_analysis.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
