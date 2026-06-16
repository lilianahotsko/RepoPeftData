#!/usr/bin/env python3
"""Plot Code2LoRA-GRU anchor-only ablation results.

Produces two figures under ``plots/``:

  * ``gru_anchor_training_curve.png`` -- cr_val eval loss vs. repos_done,
    overlaying the anchor-only run against the full-signal v2 GRU runs
    (whichever rerun directories carry a ``metrics.jsonl``).
  * ``gru_anchor_decay_cr_val.png`` -- per-commit cr_val loss vs.
    commit_index from the anchor-only run's *final* eval (epoch 2,
    repos_done=1200), binned for readability.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CKPT_ROOT = Path("/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU")
ANCHOR_DIR = CKPT_ROOT / "h100_v2_gru_anchor_3ep"
BASELINE_DIRS = [
    CKPT_ROOT / "h100_v2_gru_3ep_rerun1_seed3407",
    CKPT_ROOT / "h100_v2_gru_3ep_rerun2_seed3407",
    CKPT_ROOT / "h100_v2_gru_3ep_rerun3_seed1234",
    CKPT_ROOT / "h100_v2_gru_3ep_rerun4_seed5678",
]


def load_metrics(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def plot_training_curve(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    anchor = load_metrics(ANCHOR_DIR / "metrics.jsonl")
    if anchor:
        xs = [r["repos_done"] for r in anchor]
        ys = [r["eval_loss"] for r in anchor]
        ax.plot(xs, ys, marker="o", lw=2.0,
                label=f"anchor-only (best={min(ys):.4f})", color="#d62728")

    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(BASELINE_DIRS):
        m = load_metrics(d / "metrics.jsonl")
        if not m:
            continue
        xs = [r["repos_done"] for r in m]
        ys = [r["eval_loss"] for r in m]
        ax.plot(xs, ys, marker="s", lw=1.2, alpha=0.7,
                color=cmap(i % 10),
                label=f"v2 GRU ({d.name.split('_')[-1]}, best={min(ys):.4f})")

    ax.set_xlabel("repos trained (cumulative across epochs)")
    ax.set_ylabel("cr_val cross-entropy loss")
    ax.set_title("Code2LoRA-GRU: anchor-only vs. full per-commit signal\n"
                 "(cr_val eval suite, 10 repos, 1766 commits)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_decay_curve(out: Path) -> None:
    # Use the final eval (highest repos_done).
    files = sorted(
        ANCHOR_DIR.glob("eval_per_commit_cr_val_repos*.json"),
        key=lambda p: int(p.stem.split("repos")[-1]),
    )
    if not files:
        print(f"No per-commit eval JSONs under {ANCHOR_DIR}")
        return
    final = files[-1]
    print(f"Using {final.name}")
    payload = json.loads(final.read_text())
    rows = payload["per_commit"]

    # Group per-commit losses by commit_index, weighted by n_tokens.
    by_idx: Dict[int, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for r in rows:
        ci = int(r["commit_index"])
        loss = float(r["loss"])
        tok = int(r["n_tokens"])
        cur_l, cur_t = by_idx[ci]
        by_idx[ci] = (cur_l + loss * tok, cur_t + tok)

    xs = np.array(sorted(by_idx.keys()))
    ys = np.array([by_idx[i][0] / max(by_idx[i][1], 1) for i in xs])

    fig, (ax_raw, ax_bin) = plt.subplots(1, 2, figsize=(13, 5))

    ax_raw.scatter(xs, ys, s=10, alpha=0.35, color="#1f77b4")
    ax_raw.set_xscale("symlog", linthresh=1)
    ax_raw.set_xlabel("commit_index (symlog)")
    ax_raw.set_ylabel("per-commit cr_val loss")
    ax_raw.set_title(f"anchor-only, raw per-commit (N={len(xs)} commits)")
    ax_raw.grid(True, alpha=0.3)

    # Log-spaced bins
    max_ci = int(xs.max())
    edges = np.unique(np.concatenate([
        np.array([0, 1, 2, 5, 10]),
        np.geomspace(10, max(max_ci, 11), 12).astype(int),
    ]))
    centers = []
    means = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs < hi)
        if not m.any():
            continue
        # Token-weighted mean within bin.
        ls, ts = 0.0, 0
        for ci in xs[m]:
            ll, tt = by_idx[int(ci)]
            ls += ll
            ts += tt
        if ts == 0:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(ls / ts)
        counts.append(int(m.sum()))
    ax_bin.plot(centers, means, marker="o", lw=2.0, color="#d62728",
                label="token-weighted mean")
    for x, y, c in zip(centers, means, counts):
        ax_bin.annotate(f"n={c}", (x, y), textcoords="offset points",
                        xytext=(0, 6), fontsize=7, ha="center")
    ax_bin.set_xscale("symlog", linthresh=1)
    ax_bin.set_xlabel("commit_index (symlog, log-spaced bins)")
    ax_bin.set_ylabel("per-commit cr_val loss (token-weighted)")
    ax_bin.set_title("anchor-only, binned decay curve")
    ax_bin.grid(True, alpha=0.3)
    ax_bin.legend(loc="best")

    fig.suptitle("Code2LoRA-GRU anchor-only: per-commit cr_val loss "
                 "(final eval, repos_done=1200)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="plots",
                    help="Where to write the PNGs (default: plots/).")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_training_curve(out / "gru_anchor_training_curve.png")
    plot_decay_curve(out / "gru_anchor_decay_cr_val.png")


if __name__ == "__main__":
    main()
