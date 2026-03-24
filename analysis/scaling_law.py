#!/usr/bin/env python3
"""
Scaling law analysis and figure generation for Code2LoRA.
Fits power-law and log-linear models to EM vs. #training repos.

Usage:
    python analysis/scaling_law.py
    python analysis/scaling_law.py --include-predictions --output scaling_law.pdf
"""

import argparse
import json
import math
import os
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using manual fits only")

BASELINES_DIR = Path(os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))) / "BASELINES"

# Known data points: (n_repos, EM%)
# EM values from slurm logs since they're not always in the JSON
KNOWN_EM = {
    50: 60.88,
    100: 61.27,
    150: 61.51,
    200: 62.24,
    500: 61.18,
    623: 63.55,
}

SCALE_FILES = {
    10: "hypernet_scale_10_cr_test.json",
    25: "hypernet_scale_25_cr_test.json",
    50: "hypernet_scale_50_cr_test.json",
    100: "hypernet_scale_100_cr_test.json",
    150: "hypernet_scale_150_cr_test.json",
    200: "hypernet_scale_200_cr_test.json",
    300: "hypernet_scale_300_cr_test.json",
    409: "hypernet_no_oracle_cr_test.json",
    500: "hypernet_scale_500_cr_test.json",
    623: "hypernet_scale_623_cr_test.json",
}


def load_scaling_data():
    """Load all available scaling results."""
    data = []
    for n_repos, fname in sorted(SCALE_FILES.items()):
        fpath = BASELINES_DIR / fname
        if not fpath.exists():
            if n_repos in KNOWN_EM:
                data.append((n_repos, KNOWN_EM[n_repos], None, None))
                print(f"  N={n_repos:>5}: EM={KNOWN_EM[n_repos]:.2f}% (from log)")
            continue
        with open(fpath) as f:
            result = json.load(f)
        r = result.get("results", result)
        em = r.get("exact_match_pct") or r.get("exact_match")
        es = r.get("edit_similarity")
        cb = r.get("code_bleu")

        if em is not None and em <= 1.0:
            em *= 100

        if em is None and n_repos in KNOWN_EM:
            em = KNOWN_EM[n_repos]

        if em is not None:
            data.append((n_repos, em, es, cb))
            print(f"  N={n_repos:>5}: EM={em:.2f}%  ES={es:.4f}  CB={cb:.4f}" if es else f"  N={n_repos:>5}: EM={em:.2f}%")

    return data


def fit_log_linear(data):
    """Fit EM = a * ln(N) + b."""
    xs = [math.log(d[0]) for d in data]
    ys = [d[1] for d in data]
    n = len(data)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    a = num / den
    b = y_mean - a * x_mean
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def fit_power_law(data):
    """Fit error_rate = c * N^alpha (equivalently EM = 100 - c * N^alpha)."""
    xs = [math.log(d[0]) for d in data]
    ys = [math.log(100 - d[1]) for d in data]
    n = len(data)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    alpha = num / den
    log_c = y_mean - alpha * x_mean
    c = math.exp(log_c)
    ss_res = sum((y - (alpha * x + log_c)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return c, alpha, r2


def generate_figure(data, output_path, include_predictions=False):
    """Generate publication-quality scaling law figure."""
    if not data:
        print("No data to plot!")
        return

    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    a, b, r2_log = fit_log_linear(data)
    c, alpha, r2_pow = fit_power_law(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: EM vs N (log scale)
    ax1.scatter(xs, ys, s=80, c="#2563eb", zorder=5, edgecolors="white", linewidth=1.5)

    x_fit = list(range(5, max(xs) * 3 if include_predictions else max(xs) + 50))
    y_log = [a * math.log(x) + b for x in x_fit]
    y_pow = [100 - c * x ** alpha for x in x_fit]

    ax1.plot(x_fit, y_log, '--', color="#dc2626", linewidth=1.5, alpha=0.8,
             label=f'Log-linear: $R^2={r2_log:.3f}$')
    ax1.plot(x_fit, y_pow, '-', color="#16a34a", linewidth=1.5, alpha=0.8,
             label=f'Power law: $R^2={r2_pow:.3f}$')

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Training Repositories", fontsize=12)
    ax1.set_ylabel("Exact Match (%)", fontsize=12)
    ax1.set_title("Code2LoRA Scaling Law", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if include_predictions:
        ax1.axvline(x=max(d[0] for d in data), color="gray", linestyle=":", alpha=0.5)
        ax1.annotate("current data", xy=(max(d[0] for d in data), min(ys)),
                     fontsize=9, color="gray", ha="right")

    # Right: Error rate vs N (log-log for power law)
    error_rates = [100 - y for y in ys]
    ax2.scatter(xs, error_rates, s=80, c="#2563eb", zorder=5, edgecolors="white", linewidth=1.5)

    er_pow = [c * x ** alpha for x in x_fit]
    ax2.plot(x_fit, er_pow, '-', color="#16a34a", linewidth=1.5, alpha=0.8,
             label=f'$E = {c:.1f} \\cdot N^{{{alpha:.3f}}}$')

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Training Repositories", fontsize=12)
    ax2.set_ylabel("Error Rate (%)", fontsize=12)
    ax2.set_title("Error Rate Scaling (log-log)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="scaling_law.pdf")
    ap.add_argument("--include-predictions", action="store_true")
    args = ap.parse_args()

    print("Loading scaling data...")
    data = load_scaling_data()

    if len(data) < 2:
        print(f"Only {len(data)} data points found. Need at least 2 for fitting.")
        return

    print(f"\n{len(data)} data points loaded.")

    a, b, r2_log = fit_log_linear(data)
    c, alpha, r2_pow = fit_power_law(data)

    print(f"\n=== Log-linear fit ===")
    print(f"  EM = {a:.3f} * ln(N) + {b:.3f}")
    print(f"  R² = {r2_log:.4f}")

    print(f"\n=== Power law fit ===")
    print(f"  Error% = {c:.2f} * N^({alpha:.4f})")
    print(f"  R² = {r2_pow:.4f}")

    print(f"\n=== Predictions ===")
    for n in [10, 25, 50, 100, 200, 300, 409, 500, 623, 750, 1000, 2000]:
        em_log = a * math.log(n) + b
        em_pow = 100 - c * n ** alpha
        marker = " <-- data" if n in [d[0] for d in data] else ""
        print(f"  N={n:>5}: log={em_log:.2f}%  power={em_pow:.2f}%{marker}")

    try:
        generate_figure(data, args.output, args.include_predictions)
    except Exception as e:
        print(f"Could not generate figure: {e}")


if __name__ == "__main__":
    main()
