#!/usr/bin/env python3
"""
Compute token-length statistics for all four quantities:
  1. Full repo  (all .py files in repo directory, tokenized)
  2. DRC        (oracle context v2, untruncated)
  3. Prefix     (structured prefix + target marker)
  4. Target     (assertion target expression)

Reports mean, median, std, p75, p95, p99, max across all QnA pairs.
Outputs LaTeX table rows ready to paste into the paper.

Usage:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/token_length_stats.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"
SPLITS_DIR     = Path(os.environ.get("SCRATCH", "/scratch/lhotsko")) / "REPO_DATASET"
DRC_CACHE_DIR  = Path(os.environ.get("SCRATCH", "/scratch/lhotsko")) / "ORACLE_CONTEXT_CACHE_V2"
REPOS_DIR      = SPLITS_DIR / "repositories"
SPLITS         = ["train", "cr_val", "cr_test", "ir_val", "ir_test"]

print(f"Loading tokenizer: {TOKENIZER_NAME}", flush=True)
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def tok_len(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def stats(arr):
    a = np.array(arr, dtype=float)
    return {
        "n":      int(len(a)),
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "std":    float(a.std()),
        "min":    float(a.min()),
        "p75":    float(np.percentile(a, 75)),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
        "max":    float(a.max()),
    }


def fmt(d, label):
    return (f"{label}: mean={d['mean']:.0f}  median={d['median']:.0f}  "
            f"std={d['std']:.0f}  p75={d['p75']:.0f}  p95={d['p95']:.0f}  "
            f"p99={d['p99']:.0f}  max={d['max']:.0f}  (n={d['n']:,})")


# Repo size: tokenize all .py source files (computed once per repo)
repo_size_cache: dict[str, int] = {}

def get_repo_size_tokens(repo_name: str) -> int:
    if repo_name in repo_size_cache:
        return repo_size_cache[repo_name]
    # Look in both flat and nested dirs
    slug = repo_name.replace("/", "/")
    repo_dir = REPOS_DIR / slug
    if not repo_dir.exists():
        # Try owner/name structure
        repo_size_cache[repo_name] = 0
        return 0
    # Sum all .py files (skip test files to get "source" size; or include all)
    total = 0
    for py_file in repo_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            total += tok_len(text)
        except Exception:
            pass
    repo_size_cache[repo_name] = total
    return total


def load_drc(repo_name: str) -> dict:
    slug = repo_name.replace("/", "__")
    p = DRC_CACHE_DIR / f"{slug}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("contexts", {})


# Accumulators
repo_toks_all = []      # one per QnA pair (repo size repeated)
drc_toks_all  = []      # only pairs that have DRC
prefix_toks_all = []
target_toks_all = []

# Per-split
split_stats = {}

for split in SPLITS:
    p = SPLITS_DIR / f"{split}.json"
    if not p.exists():
        print(f"  SKIP {split} (not found)")
        continue
    data = json.loads(p.read_text())
    repos = data.get("repositories", {})
    print(f"\n=== {split}: {len(repos)} repos ===", flush=True)

    s_repo = []; s_drc = []; s_prefix = []; s_target = []
    n_pairs = 0

    for repo_name, repo_data in repos.items():
        repo_sz = get_repo_size_tokens(repo_name)
        drc_ctx = load_drc(repo_name)

        for pair in repo_data.get("qna_pairs", []):
            prefix = pair.get("prefix", "")
            target = pair.get("target", "")
            meta   = pair.get("metadata", {})
            if not prefix or not target:
                continue

            prefix_input = prefix + "\n### Target:\n"
            pl = tok_len(prefix_input)
            tl = tok_len(target)

            s_prefix.append(pl)
            s_target.append(tl)
            s_repo.append(repo_sz)

            # DRC lookup
            src_file = meta.get("file", "")
            lineno   = meta.get("lineno", 0)
            key = f"{src_file}::{lineno}"
            entry = drc_ctx.get(key, {})
            drc_text = entry.get("extracted_code", "") if isinstance(entry, dict) else ""
            if drc_text and drc_text.strip():
                dl = tok_len(drc_text)
                s_drc.append(dl)
                drc_toks_all.append(dl)

            prefix_toks_all.append(pl)
            target_toks_all.append(tl)
            repo_toks_all.append(repo_sz)
            n_pairs += 1

        if n_pairs % 5000 == 0 and n_pairs > 0:
            print(f"  {n_pairs} pairs processed...", flush=True)

    split_stats[split] = {
        "repo": stats(s_repo) if s_repo else None,
        "drc":  stats(s_drc)  if s_drc  else None,
        "prefix": stats(s_prefix) if s_prefix else None,
        "target": stats(s_target) if s_target else None,
        "n_pairs": n_pairs,
        "drc_coverage": len(s_drc) / n_pairs if n_pairs else 0,
    }
    print(f"  {n_pairs} total pairs, DRC coverage: {split_stats[split]['drc_coverage']:.1%}")


# ── Aggregate stats ───────────────────────────────────────────────────────────
agg = {
    "repo":   stats(repo_toks_all),
    "drc":    stats(drc_toks_all),
    "prefix": stats(prefix_toks_all),
    "target": stats(target_toks_all),
    "n_pairs": len(prefix_toks_all),
    "drc_coverage": len(drc_toks_all) / len(prefix_toks_all) if prefix_toks_all else 0,
}

print("\n" + "="*80)
print("AGGREGATE STATS (all splits combined)")
print("="*80)
for k in ["repo", "drc", "prefix", "target"]:
    if agg[k]:
        print(fmt(agg[k], k.upper()))
print(f"DRC coverage: {agg['drc_coverage']:.1%} of {agg['n_pairs']:,} pairs")


# ── LaTeX table ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("LATEX TABLE")
print("="*80)

def lnum(x):
    """Format number for LaTeX: use comma thousands separator."""
    if x >= 10000:
        return f"{x:,.0f}"
    elif x >= 100:
        return f"{x:.0f}"
    else:
        return f"{x:.1f}"

rows = [
    ("Repo size",   "repo"),
    ("DRC context", "drc"),
    ("Prefix",      "prefix"),
    ("Target",      "target"),
]

print(r"""\begin{table}[t]
\centering
\small
\caption{Token length statistics for all four input components across 62{,}294 QnA pairs
(Qwen2.5-Coder-1.5B tokenizer). DRC statistics are computed over the
\DRCPCT\% of pairs that have oracle context available.}
\label{tab:token_stats}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Component} & \textbf{Mean} & \textbf{Median} & \textbf{Std} & \textbf{p75} & \textbf{p95} & \textbf{p99} & \textbf{Max} \\
\midrule""")

for label, key in rows:
    d = agg[key]
    if d is None:
        print(f"{label} & \\multicolumn{{7}}{{c}}{{n/a}} \\\\")
        continue
    note = r" \textsuperscript{\dag}" if key == "drc" else ""
    print(f"{label}{note} & {lnum(d['mean'])} & {lnum(d['median'])} & {lnum(d['std'])} & "
          f"{lnum(d['p75'])} & {lnum(d['p95'])} & {lnum(d['p99'])} & {lnum(d['max'])} \\\\")

drc_pct = f"{100*agg['drc_coverage']:.1f}"
print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item[$\dag$] DRC stats computed over """ + drc_pct + r"""\% of pairs with oracle context available.
\end{tablenotes}
\end{table}""")

# Save raw numbers
output_path = Path("analysis/output/token_length_stats.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps({"aggregate": agg, "splits": {
    k: v for k, v in split_stats.items()
}}, indent=2))
print(f"\nSaved to {output_path}")
