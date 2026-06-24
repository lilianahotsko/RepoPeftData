#!/usr/bin/env python3
"""Hypernetwork-size scaling ablation: metric vs. hypernet parameter count.

Companion analysis for the size sweep launched by
``scripts/slurm/sweep_hypernet_size_{gru,static}.sh``. The base model
(Qwen2.5-Coder-1.5B) is held fixed across every point; the only thing that
changes is the hypernetwork width (a single joint multiplier ``m`` with
``head_hidden = 1024*m`` and, for the GRU variant, ``gru_hidden = 2048*m``).

For each (variant, size) this script:

  1. Loads the trained checkpoint and counts the *trainable hypernetwork*
     parameters directly from the saved state dict (``head_state`` (+
     ``gru_state`` for the GRU variant), or ``state_dict`` for the static
     variant). This is the x-axis -- exact, not estimated.
  2. Reads the merged sharded-eval summaries (produced by
     ``evaluation/merge_eval_shards.py --auto-detect``) for the requested
     suites and pulls EM / EditSim / CodeBLEU.

It then prints a table and writes a ``metric vs. hypernet params`` figure
(one line per variant, one panel per (suite, metric)) to ``analysis/figures/``.

Robust to partial results: any size/variant whose checkpoint or merged eval
JSON is missing is skipped with a warning, so this can be run while the sweep
is still in flight.

Usage::

    python analysis/plot_hypernet_size_scaling.py
    python analysis/plot_hypernet_size_scaling.py --suites ir_test cr_test \\
        --sizes 0.5 1 2 4 --variants gru static
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

SHARD_RE = re.compile(r"_shard\d+of\d+\.json$")
METRIC_KEYS = ("exact_match", "edit_similarity", "code_bleu")
METRIC_LABELS = {
    "exact_match": "Exact Match",
    "edit_similarity": "Edit Similarity",
    "code_bleu": "CodeBLEU",
}


# ---------------------------------------------------------------------------
# Checkpoint parameter counting
# ---------------------------------------------------------------------------

def _numel_state(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    total = 0
    for v in state.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return total


def count_hypernet_params(ckpt_path: Path) -> Optional[int]:
    """Sum of trainable hypernetwork params from a saved checkpoint.

    Handles both layouts:
      * GRU    : {"gru_state": ..., "head_state": ...}
      * static : {"state_dict": ...}  (head only)
    """
    if not ckpt_path.is_file():
        return None
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    n = 0
    n += _numel_state(state.get("head_state"))
    n += _numel_state(state.get("gru_state"))
    if n == 0:  # static layout
        n += _numel_state(state.get("state_dict"))
    return n or None


# ---------------------------------------------------------------------------
# Merged-eval-summary loading
# ---------------------------------------------------------------------------

def load_suite_summary(eval_dir: Path, suite: str) -> Optional[Dict[str, Any]]:
    """Find the merged (non-shard) JSON for ``suite`` in ``eval_dir`` and
    return its ``summary`` block, or None if absent."""
    if not eval_dir.is_dir():
        return None
    best: Optional[Dict[str, Any]] = None
    for p in sorted(eval_dir.glob("*.json")):
        if SHARD_RE.search(p.name):
            continue  # skip per-shard files; we want the merged one
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        summ = obj.get("summary")
        if not isinstance(summ, dict):
            continue
        if summ.get("suite") == suite or p.name.endswith(f"{suite}.json"):
            best = summ
    return best


# ---------------------------------------------------------------------------
# Sweep layout
# ---------------------------------------------------------------------------

def variant_paths(ckpt_root: Path, variant: str, m: float) -> Dict[str, Path]:
    h = int(round(1024 * m))
    g = int(round(2048 * m))
    if variant == "gru":
        run = f"h100_v2_gru_3ep_h{h}_g{g}"
        return {
            "ckpt": ckpt_root / "CODE2LORA_GRU" / run / "gru_head.best.pt",
            "eval_dir": ckpt_root / "CODE2LORA_GRU_EVAL_V2" / f"gru_h{h}_g{g}",
        }
    if variant == "static":
        run = f"h100_v2_static_3ep_h{h}"
        return {
            "ckpt": ckpt_root / "CODE2LORA_STATIC" / run / "head.latest.pt",
            "eval_dir": ckpt_root / "CODE2LORA_STATIC_EVAL_V2" / f"static_h{h}",
        }
    raise ValueError(f"unknown variant {variant!r}")


def collect(ckpt_root: Path, variants: List[str], sizes: List[float],
            suites: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for m in sizes:
            paths = variant_paths(ckpt_root, variant, m)
            n_params = count_hypernet_params(paths["ckpt"])
            row: Dict[str, Any] = {
                "variant": variant,
                "m": m,
                "head_hidden": int(round(1024 * m)),
                "gru_hidden": int(round(2048 * m)) if variant == "gru" else None,
                "n_params": n_params,
                "suites": {},
            }
            if n_params is None:
                print(f"  [skip] {variant} m={m}: checkpoint missing "
                      f"({paths['ckpt']})")
            for suite in suites:
                summ = load_suite_summary(paths["eval_dir"], suite)
                if summ is None:
                    print(f"  [warn] {variant} m={m} {suite}: no merged eval "
                          f"summary in {paths['eval_dir']}")
                    continue
                row["suites"][suite] = {
                    k: summ.get(k) for k in METRIC_KEYS
                }
                row["suites"][suite]["n_qnas"] = summ.get("n_qnas")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict[str, Any]], suites: List[str]) -> None:
    print("\n===== Hypernetwork-size scaling (base model fixed) =====")
    header = f"{'variant':8} {'m':>4} {'head':>5} {'gru':>5} {'params':>13}"
    for suite in suites:
        for k in METRIC_KEYS:
            header += f" {suite}/{k[:2].upper():>3}"
    print(header)
    for r in rows:
        params = f"{r['n_params']:,}" if r["n_params"] else "-"
        gru = r["gru_hidden"] if r["gru_hidden"] is not None else "-"
        line = (f"{r['variant']:8} {r['m']:>4} {r['head_hidden']:>5} "
                f"{str(gru):>5} {params:>13}")
        for suite in suites:
            sm = r["suites"].get(suite, {})
            for k in METRIC_KEYS:
                v = sm.get(k)
                line += f" {v:>6.3f}" if isinstance(v, (int, float)) else f" {'-':>6}"
        print(line)
    print()


def plot(rows: List[Dict[str, Any]], suites: List[str], out_path: Path) -> None:
    variants = sorted({r["variant"] for r in rows})
    n_rows = len(suites)
    n_cols = len(METRIC_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    colors = {"gru": "#1f77b4", "static": "#d62728"}
    markers = {"gru": "o", "static": "s"}

    for ri, suite in enumerate(suites):
        for ci, metric in enumerate(METRIC_KEYS):
            ax = axes[ri][ci]
            plotted = False
            for variant in variants:
                pts = [
                    (r["n_params"], r["suites"][suite][metric])
                    for r in rows
                    if r["variant"] == variant
                    and r["n_params"]
                    and suite in r["suites"]
                    and isinstance(r["suites"][suite].get(metric), (int, float))
                ]
                pts.sort()
                if not pts:
                    continue
                xs = [p[0] / 1e6 for p in pts]  # millions of params
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker=markers.get(variant, "o"), lw=2.0,
                        color=colors.get(variant), label=variant)
                plotted = True
            ax.set_xscale("log")
            ax.set_title(f"{suite} - {METRIC_LABELS[metric]}", fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel("Hypernet params (M, log)")
            if ci == 0:
                ax.set_ylabel(suite)
            ax.grid(True, alpha=0.3)
            if plotted and ri == 0 and ci == 0:
                ax.legend(fontsize=9)

    fig.suptitle("Code2LoRA hypernetwork-size scaling (base model fixed: "
                 "Qwen2.5-Coder-1.5B)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png = out_path.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[fig] wrote {out_path}\n[fig] wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get(
        "CKPT_DIR", "/scratch/lhotsko/TRAINING_CHECKPOINTS")
    ap.add_argument("--ckpt-root", default=default_root,
                    help="TRAINING_CHECKPOINTS root (default: $CKPT_DIR).")
    ap.add_argument("--variants", nargs="+", default=["gru", "static"],
                    choices=["gru", "static"])
    ap.add_argument("--sizes", nargs="+", type=float, default=[0.5, 1, 2, 4],
                    help="Joint width multipliers m (head=1024*m, gru=2048*m).")
    ap.add_argument("--suites", nargs="+", default=["ir_test", "cr_test"])
    ap.add_argument("--out", default="analysis/figures/hypernet_size_scaling.pdf")
    ap.add_argument("--dump-json", default=None,
                    help="Optional path to also dump the collected table as JSON.")
    args = ap.parse_args()

    rows = collect(Path(args.ckpt_root), args.variants, args.sizes, args.suites)
    print_table(rows, args.suites)
    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_json).write_text(json.dumps(rows, indent=2))
        print(f"[json] wrote {args.dump_json}")
    have_any = any(r["n_params"] and r["suites"] for r in rows)
    if have_any:
        plot(rows, args.suites, Path(args.out))
    else:
        print("[fig] no complete (params + eval) rows yet; skipping figure. "
              "Re-run after the sweep + merge finish.")


if __name__ == "__main__":
    main()
