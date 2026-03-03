#!/usr/bin/env python3
"""
Collect statistics and distributions about repos collected by mine_repos.py.
Produces both text and image statistics, saved to repo_stats/.
"""

import json
import statistics
from pathlib import Path
from collections import Counter
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Matplotlib for non-interactive use
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Input: repos from mine_repos.py output
INPUT_FILE = Path("/home/lhotsko/scratch/repos.json")

OUTPUT_DIR = REPO_ROOT / "repo_stats"


def load_repos():
    """Load repos from mine_repos output (JSON). Returns list of normalized dicts."""
    path = INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Repo data not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        repos = [_normalize(r) for r in data]
    else:
        repos = [_normalize(data)]
    print(f"[*] Loaded {len(repos)} repos from {path}")
    return repos


def _normalize(obj):
    """Normalize repo object to common schema."""
    size = obj.get("repo_size_kb") or obj.get("size_kb") or obj.get("size")
    if size is None:
        size = -1
    return {
        "full_name": obj.get("full_name", ""),
        "url": obj.get("url", ""),
        "stars": int(obj.get("stars", 0) or 0),
        "language": (obj.get("language") or "").strip() or "Unknown",
        "size_kb": int(size) if size is not None else -1,
        "matched_query": obj.get("matched_query", ""),
        "forks": int(obj.get("forks", 0) or 0),
        "open_issues": int(obj.get("open_issues_count", 0) or 0),
    }


def _percentile(data, p):
    """Compute percentile (0-100)."""
    if not data:
        return None
    s = sorted(data)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def compute_stats(repos):
    """Compute all statistics from repo list."""
    stars = [r["stars"] for r in repos if r["stars"] >= 0]
    sizes_kb = [r["size_kb"] for r in repos if r["size_kb"] > 0]
    sizes_mb = [s / 1024 for s in sizes_kb]
    languages = [r["language"] for r in repos]
    queries = [r["matched_query"] for r in repos if r.get("matched_query")]

    stats = {
        "n_repos": len(repos),
        "n_with_size": len(sizes_kb),
        "n_without_size": len(repos) - len(sizes_kb),
        "stars": {
            "min": min(stars) if stars else None,
            "max": max(stars) if stars else None,
            "mean": float(statistics.mean(stars)) if stars else None,
            "median": float(statistics.median(stars)) if stars else None,
            "p25": _percentile(stars, 25),
            "p75": _percentile(stars, 75),
        },
        "size_kb": {
            "min": min(sizes_kb) if sizes_kb else None,
            "max": max(sizes_kb) if sizes_kb else None,
            "mean": float(statistics.mean(sizes_kb)) if sizes_kb else None,
            "median": float(statistics.median(sizes_kb)) if sizes_kb else None,
        },
        "size_mb": {
            "min": min(sizes_mb) if sizes_mb else None,
            "max": max(sizes_mb) if sizes_mb else None,
            "mean": float(statistics.mean(sizes_mb)) if sizes_mb else None,
            "median": float(statistics.median(sizes_mb)) if sizes_mb else None,
        },
        "language_counts": dict(Counter(languages).most_common()),
        "matched_query_counts": dict(Counter(queries).most_common()) if queries else {},
    }
    return stats, stars, sizes_kb, sizes_mb, languages, queries


def write_text_report(stats, output_path):
    """Write text statistics report."""
    lines = [
        "=" * 60,
        "REPOSITORY STATISTICS REPORT",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"Total repositories:     {stats['n_repos']}",
        f"With valid size:        {stats['n_with_size']}",
        f"Without/missing size:   {stats['n_without_size']}",
        "",
        "STARS",
        "-" * 40,
    ]
    s = stats["stars"]
    if s["min"] is not None:
        lines.extend([
            f"Min:    {s['min']}",
            f"Max:    {s['max']}",
            f"Mean:   {s['mean']:.1f}",
            f"Median: {s['median']:.1f}",
            f"P25:    {s['p25']:.1f}",
            f"P75:    {s['p75']:.1f}",
        ])
    else:
        lines.append("(no star data)")
    lines.extend(["", "SIZE (KB)", "-" * 40])
    sk = stats["size_kb"]
    if sk["min"] is not None:
        lines.extend([
            f"Min:    {sk['min']:,} KB",
            f"Max:    {sk['max']:,} KB",
            f"Mean:   {sk['mean']:,.0f} KB",
            f"Median: {sk['median']:,.0f} KB",
        ])
    else:
        lines.append("(no size data)")
    lines.extend(["", "SIZE (MB)", "-" * 40])
    sm = stats["size_mb"]
    if sm["min"] is not None:
        lines.extend([
            f"Min:    {sm['min']:.2f} MB",
            f"Max:    {sm['max']:.2f} MB",
            f"Mean:   {sm['mean']:.2f} MB",
            f"Median: {sm['median']:.2f} MB",
        ])
    else:
        lines.append("(no size data)")
    lines.extend(["", "LANGUAGE DISTRIBUTION", "-" * 40])
    for lang, count in list(stats["language_counts"].items())[:20]:
        pct = 100 * count / stats["n_repos"]
        lines.append(f"  {lang:20} {count:5} ({pct:5.1f}%)")
    if stats["matched_query_counts"]:
        lines.extend(["", "MATCHED QUERY DISTRIBUTION", "-" * 40])
        for q, count in list(stats["matched_query_counts"].items())[:15]:
            short = (q[:50] + "…") if len(q) > 50 else q
            lines.append(f"  {short:52} {count:4}")
    lines.append("")
    lines.append("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"[*] Text report saved to {output_path}")


def _logspace(start, stop, num):
    """Log-spaced bins (fallback when numpy unavailable)."""
    import math
    log_start = math.log10(max(0.01, start))
    log_stop = math.log10(max(stop, start + 0.01))
    return [10 ** (log_start + (log_stop - log_start) * i / (num - 1)) for i in range(num)]


def plot_stars_distribution(stars, output_path):
    """Plot stars distribution histogram."""
    if not stars:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    lo, hi = max(1, min(stars)), max(stars) + 1
    bins = _logspace(lo, hi, 30)
    ax.hist(stars, bins=bins, color="#2E86AB", edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Stars", fontsize=11)
    ax.set_ylabel("Number of Repos", fontsize=11)
    ax.set_title(f"Stars Distribution (n={len(stars)})", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved {output_path}")


def plot_size_distribution(sizes_mb, output_path):
    """Plot repo size distribution (histogram + CDF)."""
    if not sizes_mb:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]
    bins = [b for b in bins if b <= max(sizes_mb) * 1.2]
    if not bins:
        mx = max(sizes_mb)
        bins = [mx * i / 19 for i in range(20)]
    counts, edges, patches = ax.hist(
        sizes_mb, bins=bins, color="#A23B72", edgecolor="white", linewidth=0.8
    )
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + 0.5,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlabel("Repo Size (MB)", fontsize=11)
    ax.set_ylabel("Number of Repos", fontsize=11)
    ax.set_title(f"Size Distribution (n={len(sizes_mb)})", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(axis="y", alpha=0.3)

    # CDF
    ax2 = axes[1]
    sorted_sizes = sorted(sizes_mb)
    n = len(sorted_sizes)
    cumulative = [100 * (i + 1) / n for i in range(n)]
    ax2.plot(sorted_sizes, cumulative, color="#A23B72", linewidth=2)
    ax2.fill_between(sorted_sizes, cumulative, alpha=0.15, color="#A23B72")
    for pct in [25, 50, 75, 90]:
        idx = min(int(len(sorted_sizes) * pct / 100), len(sorted_sizes) - 1)
        val = sorted_sizes[idx]
        ax2.axhline(y=pct, color="gray", linestyle="--", alpha=0.3)
        ax2.plot(val, pct, "o", color="#C44E52", markersize=6, zorder=5)
        ax2.annotate(
            f"P{pct}: {val:.0f} MB",
            xy=(val, pct),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=8,
            color="#C44E52",
        )
    ax2.set_xlabel("Repo Size (MB)", fontsize=11)
    ax2.set_ylabel("Cumulative %", fontsize=11)
    ax2.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved {output_path}")


def plot_language_distribution(language_counts, output_path):
    """Plot language distribution bar chart."""
    if not language_counts:
        return
    top_n = 15
    items = list(language_counts.items())[:top_n]
    langs = [x[0] for x in items]
    counts = [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(langs)), counts, color="#F18F01", edgecolor="white")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs, fontsize=10)
    ax.set_xlabel("Number of Repos", fontsize=11)
    ax.set_title(f"Language Distribution (top {top_n})", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for i, (bar, c) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved {output_path}")


def plot_matched_query_distribution(query_counts, output_path):
    """Plot matched query distribution bar chart."""
    if not query_counts:
        return
    top_n = 12
    items = list(query_counts.items())[:top_n]
    labels = []
    for q, _ in items:
        short = (q[:45] + "…") if len(q) > 45 else q
        labels.append(short.replace('"', "'"))
    counts = [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), counts, color="#3B1F2B", edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Repos", fontsize=11)
    ax.set_title(f"Matched Query Distribution (top {top_n})", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved {output_path}")


def plot_stars_vs_size(stars, sizes_kb, output_path):
    """Scatter plot: stars vs repo size."""
    if not stars or not sizes_kb or len(stars) != len(sizes_kb):
        return
    # Align by index - need repos with both
    repos_with_both = [
        (s, k / 1024)
        for s, k in zip(stars, sizes_kb)
        if s > 0 and k > 0
    ]
    if not repos_with_both:
        return
    stars_vals = [x[0] for x in repos_with_both]
    sizes_vals = [x[1] for x in repos_with_both]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stars_vals, sizes_vals, alpha=0.5, s=20, c="#2E86AB")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stars", fontsize=11)
    ax.set_ylabel("Repo Size (MB)", fontsize=11)
    ax.set_title("Stars vs Repo Size", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved {output_path}")


def main():
    print("[*] Loading repos...")
    repos = load_repos()

    print("[*] Computing statistics...")
    stats, stars, sizes_kb, sizes_mb, languages, queries = compute_stats(repos)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Text report
    write_text_report(stats, OUTPUT_DIR / "stats.txt")

    # Plots
    # plot_stars_distribution(stars, OUTPUT_DIR / "stars_distribution.png")
    # plot_size_distribution(sizes_mb, OUTPUT_DIR / "size_distribution.png")
    # plot_language_distribution(stats["language_counts"], OUTPUT_DIR / "language_distribution.png")
    # if stats["matched_query_counts"]:
    #     plot_matched_query_distribution(
    #         stats["matched_query_counts"],
    #         OUTPUT_DIR / "matched_query_distribution.png",
    #     )
    # # Stars vs size: align by repos that have both
    # aligned = [(r["stars"], r["size_kb"]) for r in repos if r["stars"] > 0 and r["size_kb"] > 0]
    # if aligned:
    #     ss, sk = zip(*aligned)
    #     plot_stars_vs_size(list(ss), list(sk), OUTPUT_DIR / "stars_vs_size.png")

    print(f"\n[✓] Done. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
