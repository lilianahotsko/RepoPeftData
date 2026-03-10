#!/usr/bin/env python3
"""
Generate publication-quality seaborn figures for the EMNLP paper.

Figures:
1. Main results grouped bar chart (CR + IR)
2. Scaling law: EM vs. #training repos with fit
3. Per-repo EM distribution (violin/box)
4. Data sparsity: Per-repo LoRA EM vs. #training pairs

Usage:
    python analysis/generate_figures.py
    python analysis/generate_figures.py --results-dir $SCRATCH/BASELINES --output-dir analysis/figures
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_result(results_dir: Path, name: str):
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_metrics(d):
    """Extract EM%, EditSim, CodeBLEU from either result format."""
    if d is None:
        return None, None, None
    r = d.get("results", d)
    em = r.get("exact_match_pct") or r.get("exact_match")
    if em is not None and em <= 1.0:
        em *= 100
    es = r.get("edit_similarity")
    cb = r.get("code_bleu")
    return em, es, cb


def per_repo_em(entries):
    by_repo = defaultdict(lambda: {"em": 0, "n": 0})
    for e in entries:
        repo = e.get("repo", "unknown")
        by_repo[repo]["n"] += 1
        by_repo[repo]["em"] += int(e.get("exact_match", False))
    return {r: 100.0 * v["em"] / v["n"] for r, v in by_repo.items() if v["n"] > 0}


def setup_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2, rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.edgecolor": ".3",
        "grid.color": ".85",
    })
    return plt, sns


def figure_main_results(results_dir, output_dir):
    """Figure 1: Grouped bar chart of main results (CR + IR)."""
    plt, sns = setup_style()
    import pandas as pd

    methods_cr = [
        ("Pretrained", "pretrained_cr_test", "#94c4df"),
        ("RAG (k=3)", "rag_top3_cr_test", "#94c4df"),
        ("ICL (3-shot)", "icl_3shot_cr_test", "#94c4df"),
        ("DRC", "oracle_context_cr_test", "#94c4df"),
        ("FFT", "fft_no_oracle_cr_test", "#fdae6b"),
        ("sLoRA", "single_lora_no_oracle_cr_test", "#fdae6b"),
        ("Code2LoRA", "hypernet_no_oracle_cr_test", "#b5cf6b"),
        ("Code2LoRA\n(PAW)", "hypernet_paw_no_oracle_cr_test", "#b5cf6b"),
    ]
    methods_ir = [
        ("Pretrained", "pretrained_ir_test"),
        ("RAG (k=3)", "rag_top3_ir_test"),
        ("ICL (3-shot)", "icl_3shot_ir_test"),
        ("DRC", "oracle_context_ir_test"),
        ("FFT", "fft_no_oracle_ir_test"),
        ("sLoRA", "single_lora_no_oracle_ir_test"),
        ("Code2LoRA", "hypernet_no_oracle_ir_test"),
        ("Code2LoRA\n(PAW)", "hypernet_paw_no_oracle_ir_test"),
    ]

    rows = []
    for label, key, color in methods_cr:
        r = load_result(results_dir, key)
        em, _, _ = get_metrics(r)
        if em is not None:
            rows.append({"Method": label, "Setting": "Cross-Repo", "EM (%)": em})

    for label, key in methods_ir:
        r = load_result(results_dir, key)
        em, _, _ = get_metrics(r)
        if em is not None:
            rows.append({"Method": label, "Setting": "In-Repo", "EM (%)": em})

    if len(rows) < 3:
        print("  Insufficient data for main results bar chart, skipping")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 5.5))

    palette = {"Cross-Repo": "#4c78a8", "In-Repo": "#f58518"}
    sns.barplot(data=df, x="Method", y="EM (%)", hue="Setting",
                palette=palette, edgecolor=".3", linewidth=0.8, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

    ax.set_ylabel("Exact Match (%)")
    ax.set_xlabel("")
    ax.set_title("Main Results: Cross-Repo vs. In-Repo Evaluation", fontweight="bold")
    ax.legend(title="", frameon=True, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    plt.tight_layout()
    out = output_dir / "main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


KNOWN_SCALING_EM = {
    50: 60.88,
    100: 61.27,
    200: 62.24,
}

SCALE_FILES = {
    10: "hypernet_scale_10_cr_test",
    25: "hypernet_scale_25_cr_test",
    50: "hypernet_scale_50_cr_test",
    100: "hypernet_scale_100_cr_test",
    150: "hypernet_scale_150_cr_test",
    200: "hypernet_scale_200_cr_test",
    300: "hypernet_scale_300_cr_test",
    409: "hypernet_no_oracle_cr_test",
    500: "hypernet_scale_500_cr_test",
    623: "hypernet_scale_623_cr_test",
}


def figure_scaling_law(results_dir, output_dir):
    """Figure 2: Scaling law — EM vs. #training repos with log-linear fit."""
    plt, sns = setup_style()

    points = []
    for n_repos, key in sorted(SCALE_FILES.items()):
        r = load_result(results_dir, key)
        em, _, _ = get_metrics(r)
        if em is not None:
            points.append((n_repos, em))
        elif n_repos in KNOWN_SCALING_EM:
            points.append((n_repos, KNOWN_SCALING_EM[n_repos]))

    if len(points) < 3:
        print(f"  Only {len(points)} scaling points, skipping")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Fit log-linear: EM = a * ln(N) + b
    n = len(points)
    log_xs = [math.log(x) for x in xs]
    x_m = sum(log_xs) / n
    y_m = sum(ys) / n
    a = sum((lx - x_m) * (y - y_m) for lx, y in zip(log_xs, ys)) / sum((lx - x_m) ** 2 for lx in log_xs)
    b = y_m - a * x_m
    ss_res = sum((y - (a * lx + b)) ** 2 for lx, y in zip(log_xs, ys))
    ss_tot = sum((y - y_m) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Fit power-law: error = c * N^alpha
    log_ers = [math.log(100 - y) for y in ys]
    er_m = sum(log_ers) / n
    alpha = sum((lx - x_m) * (le - er_m) for lx, le in zip(log_xs, log_ers)) / sum((lx - x_m) ** 2 for lx in log_xs)
    log_c = er_m - alpha * x_m
    c = math.exp(log_c)
    ss_res_p = sum((le - (alpha * lx + log_c)) ** 2 for lx, le in zip(log_xs, log_ers))
    ss_tot_p = sum((le - er_m) ** 2 for le in log_ers)
    r2_p = 1 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: EM vs N (log scale)
    x_fit = list(range(8, int(max(xs) * 1.5)))
    y_log = [a * math.log(x) + b for x in x_fit]
    y_pow = [100 - c * x ** alpha for x in x_fit]

    ax1.scatter(xs, ys, s=100, c="#2563eb", zorder=5, edgecolors="white", linewidth=2)
    ax1.plot(x_fit, y_log, '--', color="#dc2626", linewidth=2, alpha=0.7,
             label=f'Log-linear ($R^2={r2:.3f}$)')
    ax1.plot(x_fit, y_pow, '-', color="#16a34a", linewidth=2, alpha=0.7,
             label=f'Power law ($R^2={r2_p:.3f}$)')

    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, color=".3")

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Training Repositories")
    ax1.set_ylabel("Exact Match (%)")
    ax1.set_title("Code2LoRA Scaling with Repository Diversity", fontweight="bold")
    ax1.legend(fontsize=10, frameon=True)
    ax1.set_xlim(6, max(xs) * 2)

    # Right: Error rate (log-log)
    error_rates = [100 - y for y in ys]
    er_pow = [c * x ** alpha for x in x_fit]

    ax2.scatter(xs, error_rates, s=100, c="#2563eb", zorder=5, edgecolors="white", linewidth=2)
    ax2.plot(x_fit, er_pow, '-', color="#16a34a", linewidth=2, alpha=0.7,
             label=f'$E = {c:.1f} \\cdot N^{{{alpha:.3f}}}$')

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Training Repositories")
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("Error Rate Scaling (log-log)", fontweight="bold")
    ax2.legend(fontsize=10, frameon=True)

    plt.tight_layout()
    out = output_dir / "scaling_law.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  Fit: EM = {a:.3f} * ln(N) + {b:.3f}  (R²={r2:.4f})")
    print(f"  Fit: Error = {c:.1f} * N^({alpha:.4f})  (R²={r2_p:.4f})")


def figure_per_repo_violin(results_dir, output_dir):
    """Figure 3: Per-repo EM violin plot across methods."""
    plt, sns = setup_style()
    import pandas as pd

    methods = [
        ("Pretrained", "pretrained_ir_test"),
        ("FFT", "fft_no_oracle_ir_test"),
        ("sLoRA", "single_lora_no_oracle_ir_test"),
        ("Code2LoRA", "hypernet_no_oracle_ir_test"),
        ("Code2LoRA\n(PAW)", "hypernet_paw_no_oracle_ir_test"),
    ]

    rows = []
    for label, key in methods:
        r = load_result(results_dir, key)
        if r and "entries" in r:
            repo_ems = per_repo_em(r["entries"])
            for repo, em in repo_ems.items():
                rows.append({"Method": label, "Per-Repo EM (%)": em})

    if not rows:
        print("  No per-repo data available, skipping violin plot")
        return

    df = pd.DataFrame(rows)
    palette = sns.color_palette("Set2", n_colors=len(methods))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.violinplot(data=df, x="Method", y="Per-Repo EM (%)", hue="Method",
                   palette=palette, inner="quartile", cut=0, linewidth=1,
                   legend=False, ax=ax)

    medians = df.groupby("Method", sort=False)["Per-Repo EM (%)"].median()
    for i, (method, median) in enumerate(medians.items()):
        ax.annotate(f"median={median:.1f}", (i, median),
                    textcoords="offset points", xytext=(0, -25),
                    ha="center", fontsize=8, color=".3", style="italic")

    ax.set_xlabel("")
    ax.set_ylabel("Per-Repository Exact Match (%)")
    ax.set_title("Per-Repository Performance Distribution (IR Test)", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "per_repo_violin.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_data_sparsity(results_dir, splits_dir, output_dir):
    """Figure 4: Per-repo LoRA EM vs. training data size scatter."""
    plt, sns = setup_style()
    import pandas as pd

    prlora = load_result(results_dir, "per_repo_lora_no_oracle_ir_test")
    c2l = load_result(results_dir, "hypernet_no_oracle_ir_test")

    train_path = splits_dir / "train.json"
    if not train_path.exists():
        print(f"  train.json not found at {splits_dir}, skipping")
        return

    train_data = json.loads(train_path.read_text())
    repo_sizes = {r: len(v.get("qna_pairs", [])) for r, v in train_data.get("repositories", {}).items()}

    rows = []
    if prlora and "entries" in prlora:
        prlora_ems = per_repo_em(prlora["entries"])
        for repo, em in prlora_ems.items():
            n = repo_sizes.get(repo, 0)
            if n > 0:
                rows.append({"Method": "Per-Repo LoRA", "Training Pairs": n, "EM (%)": em})

    if c2l and "entries" in c2l:
        c2l_ems = per_repo_em(c2l["entries"])
        for repo, em in c2l_ems.items():
            n = repo_sizes.get(repo, 0)
            if n > 0:
                rows.append({"Method": "Code2LoRA", "Training Pairs": n, "EM (%)": em})

    if len(rows) < 10:
        print(f"  Only {len(rows)} data points, skipping scatter")
        return

    df = pd.DataFrame(rows)
    palette = {"Per-Repo LoRA": "#e6550d", "Code2LoRA": "#3182bd"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df, x="Training Pairs", y="EM (%)", hue="Method",
                    palette=palette, alpha=0.5, s=30, edgecolor=None, ax=ax)

    for method, color in palette.items():
        sub = df[df["Method"] == method]
        if len(sub) > 10:
            import numpy as np
            z = np.polyfit(np.log(sub["Training Pairs"]), sub["EM (%)"], 1)
            x_fit = np.linspace(sub["Training Pairs"].min(), sub["Training Pairs"].max(), 100)
            ax.plot(x_fit, z[0] * np.log(x_fit) + z[1], "--", color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel("Training Examples per Repository")
    ax.set_ylabel("Per-Repository Exact Match (%)")
    ax.set_title("Performance vs. Repository Training Data Size", fontweight="bold")
    ax.legend(frameon=True)

    plt.tight_layout()
    out = output_dir / "data_sparsity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def figure_scaling_editsim(results_dir, output_dir):
    """Figure 5: Scaling with EditSim and CodeBLEU alongside EM."""
    plt, sns = setup_style()
    import pandas as pd

    rows = []
    for n_repos, key in sorted(SCALE_FILES.items()):
        r = load_result(results_dir, key)
        _, es, cb = get_metrics(r)
        if es is not None:
            rows.append({"N Repos": n_repos, "Metric": "EditSim", "Score": es})
        if cb is not None:
            rows.append({"N Repos": n_repos, "Metric": "CodeBLEU", "Score": cb})

    if len(rows) < 4:
        print("  Insufficient EditSim/CodeBLEU scaling data, skipping")
        return

    df = pd.DataFrame(rows)
    palette = {"EditSim": "#4c78a8", "CodeBLEU": "#e45756"}

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=df, x="N Repos", y="Score", hue="Metric", style="Metric",
                 markers=True, dashes=False, palette=palette, markersize=10,
                 linewidth=2, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Repositories")
    ax.set_ylabel("Score")
    ax.set_title("EditSim & CodeBLEU Scaling", fontweight="bold")
    ax.legend(frameon=True)

    plt.tight_layout()
    out = output_dir / "scaling_editsim.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    default_results = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "BASELINES",
    )
    default_splits = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=default_results)
    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--output-dir", type=str, default="analysis/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_dir}")
    print(f"Splits:  {splits_dir}")
    print(f"Output:  {output_dir}\n")

    print("1. Main results (grouped bar)")
    figure_main_results(results_dir, output_dir)

    print("\n2. Scaling law (EM vs. repos)")
    figure_scaling_law(results_dir, output_dir)

    print("\n3. Per-repo EM distribution (violin)")
    figure_per_repo_violin(results_dir, output_dir)

    print("\n4. Data sparsity scatter")
    figure_data_sparsity(results_dir, splits_dir, output_dir)

    print("\n5. Scaling (EditSim + CodeBLEU)")
    figure_scaling_editsim(results_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
