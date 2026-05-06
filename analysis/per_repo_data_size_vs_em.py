#!/usr/bin/env python3
"""
Plot per-repo training-set size vs. per-repo EM for the pLoRA baseline.

Backs the Table~1 entry "Per-repo LoRA + DRC: N/A (data-sparse, overfits)":
the existing pLoRA-no-DRC results already show a clear dependence on the
number of training examples per repo. Adding DRC at 8K context multiplies
effective per-step capacity by ~4x, so any repo that already overfits at N
training examples without DRC overfits even faster with DRC. The figure is
the visual evidence behind the N/A.

Run::

    python analysis/per_repo_data_size_vs_em.py \
        --plora-results $SCRATCH/BASELINES/per_repo_lora_no_oracle_ir_test.json \
        --train-json    $SCRATCH/REPO_DATASET/train.json \
        --output        RepoPeft_Paper/figures/plora_data_sparsity.pdf

The script also prints the summary statistics referenced in the paper:
median pLoRA EM at N <= 50 vs. N > 50, and the linear-fit slope of EM on
log10(N).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_per_repo_em(path: Path) -> Dict[str, Tuple[float, int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Tuple[float, int]] = {}
    for rid, info in (d.get("per_repo") or {}).items():
        em_pct = float(info.get("exact_match_pct", 0.0))
        n_test = int(info.get("n", 0))
        out[rid] = (em_pct, n_test)
    return out


def load_train_sizes(path: Path) -> Dict[str, int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        rid: len((rd.get("qna_pairs") or []))
        for rid, rd in (d.get("repositories") or {}).items()
    }


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if var <= 0:
        return 0.0, my
    slope = cov / var
    intercept = my - slope * mx
    return slope, intercept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plora-results",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "."))
        / "BASELINES"
        / "per_repo_lora_no_oracle_ir_test.json",
    )
    ap.add_argument(
        "--train-json",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "."))
        / "REPO_DATASET"
        / "train.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("RepoPeft_Paper")
        / "figures"
        / "plora_data_sparsity.pdf",
    )
    ap.add_argument(
        "--cutoff",
        type=int,
        default=50,
        help="Training-size cutoff used in the paper text (default 50).",
    )
    args = ap.parse_args()

    if not args.plora_results.exists():
        print(f"ERROR: pLoRA results not found at {args.plora_results}", file=sys.stderr)
        return 1
    if not args.train_json.exists():
        print(f"ERROR: train.json not found at {args.train_json}", file=sys.stderr)
        return 1

    per_repo_em = load_per_repo_em(args.plora_results)
    train_sizes = load_train_sizes(args.train_json)

    rows: List[Tuple[str, int, float]] = []
    for rid, (em_pct, _n_test) in per_repo_em.items():
        ntrain = train_sizes.get(rid)
        if ntrain is None or ntrain <= 0:
            continue
        rows.append((rid, ntrain, em_pct))

    if not rows:
        print("ERROR: no overlap between pLoRA results and train sizes", file=sys.stderr)
        return 1

    # Summary statistics for paper text.
    small = [(r, em) for r, n, em in rows if n <= args.cutoff for r, em in [(r, em)]]
    large = [(r, em) for r, n, em in rows if n > args.cutoff for r, em in [(r, em)]]

    def median(xs):
        xs = sorted(xs)
        if not xs:
            return float("nan")
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    em_small = [em for _r, em in small]
    em_large = [em for _r, em in large]
    print(f"Repos:            {len(rows)}")
    print(f"  N_train <= {args.cutoff}: {len(small)} repos, "
          f"median EM = {median(em_small):.2f}%")
    print(f"  N_train >  {args.cutoff}: {len(large)} repos, "
          f"median EM = {median(em_large):.2f}%")

    log_x = [math.log10(n) for _r, n, _em in rows]
    y = [em for _r, _n, em in rows]
    slope, intercept = linfit(log_x, y)
    print(f"  Linear fit EM = {slope:.2f} * log10(N) + {intercept:.2f}")

    # Plot --------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return 0

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    sizes_arr = [n for _r, n, _em in rows]
    em_arr = [em for _r, _n, em in rows]
    ax.scatter(sizes_arr, em_arr, s=14, alpha=0.55, edgecolors="none")

    # Regression line in log-x space
    xs = sorted(sizes_arr)
    if xs:
        x_lo, x_hi = max(1, xs[0]), xs[-1]
        xs_line = [x_lo, x_hi]
        ys_line = [
            slope * math.log10(x) + intercept for x in xs_line
        ]
        ax.plot(xs_line, ys_line, "--", color="firebrick", lw=1.2,
                label=f"linear fit, slope={slope:.1f}/decade")

    ax.axvline(args.cutoff, color="gray", ls=":", lw=1,
               label=f"N_train = {args.cutoff}")

    ax.set_xscale("log")
    ax.set_xlabel("Per-repo training examples (log)")
    ax.set_ylabel("Per-repo EM (\\%)")
    ax.set_title("Per-repo LoRA: EM vs.\\ training-set size")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote figure -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
