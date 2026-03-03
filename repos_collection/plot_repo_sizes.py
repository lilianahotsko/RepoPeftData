#!/usr/bin/env python3
"""Plot the distribution of repo sizes from pytest_repos_5k.jsonl."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSONL = Path(__file__).resolve().parent / "pytest_repos_5k.jsonl"
OUT_PNG = Path(__file__).resolve().parent / "repo_size_distribution.png"


def load_sizes():
    active, commented = [], []
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            obj = json.loads(line[2:])
            commented.append(obj.get("repo_size_kb", 0) / 1024)
        elif line.startswith("{"):
            obj = json.loads(line)
            s = obj.get("repo_size_kb", 0)
            if s > 0:
                active.append(s / 1024)
    return active, commented


def main():
    active_sizes, commented_sizes = load_sizes()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of active repos
    ax = axes[0]
    bins = [0, 1, 5, 10, 25, 50, 100, 200, 500]
    counts, edges, patches = ax.hist(
        active_sizes, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.8
    )
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + 2,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlabel("Repo Size (MB)", fontsize=11)
    ax.set_ylabel("Number of Repos", fontsize=11)
    ax.set_title(
        f"Active Repos Size Distribution (n={len(active_sizes)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xticks(bins)
    ax.set_xticklabels([str(b) for b in bins], fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: cumulative distribution
    ax2 = axes[1]
    sorted_sizes = np.sort(active_sizes)
    cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes) * 100
    ax2.plot(sorted_sizes, cumulative, color="#4C72B0", linewidth=2)
    ax2.fill_between(sorted_sizes, cumulative, alpha=0.15, color="#4C72B0")

    for pct in [25, 50, 75, 90]:
        idx = int(len(sorted_sizes) * pct / 100)
        val = sorted_sizes[min(idx, len(sorted_sizes) - 1)]
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

    summary = (
        f"Active: {len(active_sizes)} repos | "
        f"Commented out: {len(commented_sizes)} (>500MB)\n"
        f"Median: {np.median(active_sizes):.1f} MB | "
        f"Mean: {np.mean(active_sizes):.1f} MB | "
        f"Max: {max(active_sizes):.0f} MB"
    )
    fig.text(0.5, -0.02, summary, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
