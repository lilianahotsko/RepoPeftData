#!/usr/bin/env python3
"""
Token-length statistics for two input formats:
  1. Prefix-only:  {prefix}\n### Target:\n{target}
  2. DRC+Prefix:   {drc_context}\n\n\n{prefix}\n### Target:\n{target}

Computed for all splits individually and combined.

Run:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/input_dist_stats.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"
SPLITS_DIR     = Path("/scratch/lhotsko/REPO_DATASET")
DRC_CACHE_DIR  = Path("/scratch/lhotsko/ORACLE_CONTEXT_CACHE_V2")
TARGET_MARKER  = "### Target:"
SPLITS         = ["train", "cr_val", "cr_test", "ir_val", "ir_test"]
MAX_INPUT_TOKS = 16384   # budget cap used at inference

print(f"Loading tokenizer: {TOKENIZER_NAME}")
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

def stats(arr):
    a = np.array(arr)
    return {
        "n":      len(a),
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "std":    float(a.std()),
        "min":    float(a.min()),
        "p75":    float(np.percentile(a, 75)),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
        "max":    float(a.max()),
    }

def fmt(d):
    return (f"n={d['n']:,}  mean={d['mean']:.0f}  median={d['median']:.0f}  "
            f"std={d['std']:.0f}  p95={d['p95']:.0f}  p99={d['p99']:.0f}  max={d['max']:.0f}")

def load_oracle_cache(repo: str) -> dict:
    slug = repo.replace("/", "__")
    p = DRC_CACHE_DIR / f"{slug}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("contexts", {})

def lookup_drc(repo: str, metadata: dict, ctx_cache: dict) -> str:
    src_file = metadata.get("file", "")   # relative path, e.g. "src/foo/bar.py"
    lineno   = metadata.get("lineno", 0)
    key = f"{src_file}::{lineno}"
    entry = ctx_cache.get(key, {})
    return entry.get("extracted_code", "") if isinstance(entry, dict) else ""

# Accumulators: per-split and global
all_prefix_lens   = defaultdict(list)
all_drc_lens      = defaultdict(list)   # DRC-only token lengths (subset with context)
all_full_drc_lens = defaultdict(list)   # DRC+prefix total (subset with context)
all_target_lens   = defaultdict(list)

for split in SPLITS:
    p = SPLITS_DIR / f"{split}.json"
    if not p.exists():
        print(f"  SKIP {split} (not found)")
        continue
    d = json.loads(p.read_text())
    repos = d.get("repositories", {})

    print(f"\n=== {split}: loading {len(repos)} repos ===")

    repo_cache = {}  # lazy DRC cache per repo
    n_pairs = 0

    for repo_name, repo_data in repos.items():
        pairs  = repo_data.get("qna_pairs", [])
        ctx_c  = None  # lazy load

        for pair in pairs:
            prefix = pair.get("prefix", "")
            target = pair.get("target", "")
            meta   = pair.get("metadata", {})
            if not prefix or not target:
                continue

            # ── Prefix-only format ──────────────────────────────────────────
            full_text = prefix + "\n" + TARGET_MARKER + "\n" + target
            ids = tok.encode(full_text, add_special_tokens=False)
            prefix_ids = tok.encode(prefix + "\n" + TARGET_MARKER + "\n", add_special_tokens=False)
            target_ids = tok.encode(target, add_special_tokens=False)

            all_prefix_lens[split].append(len(prefix_ids))
            all_target_lens[split].append(len(target_ids))
            all_prefix_lens["all"].append(len(prefix_ids))
            all_target_lens["all"].append(len(target_ids))

            # ── DRC+Prefix format ───────────────────────────────────────────
            if ctx_c is None:
                ctx_c = load_oracle_cache(repo_name)

            drc = lookup_drc(repo_name, meta, ctx_c)
            if drc and drc.strip():
                sep = "\n\n\n"
                # budget-capped as in real inference
                budget = MAX_INPUT_TOKS - len(prefix_ids) - len(tok.encode(sep, add_special_tokens=False)) - 2
                drc_truncated = drc if budget >= len(tok.encode(drc, add_special_tokens=False)) else \
                    tok.decode(tok.encode(drc, add_special_tokens=False)[:budget])
                drc_toks = len(tok.encode(drc_truncated, add_special_tokens=False))
                total_input = drc_toks + len(tok.encode(sep, add_special_tokens=False)) + len(prefix_ids)

                all_drc_lens[split].append(drc_toks)
                all_full_drc_lens[split].append(total_input)
                all_drc_lens["all"].append(drc_toks)
                all_full_drc_lens["all"].append(total_input)

            n_pairs += 1

    print(f"  Processed {n_pairs} pairs")

# ── Print results ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("INPUT DISTRIBUTION STATISTICS (token counts)")
print("="*80)

for split in SPLITS + ["all"]:
    if split not in all_prefix_lens:
        continue
    n_drc = len(all_drc_lens[split])
    pct_drc = 100 * n_drc / len(all_prefix_lens[split]) if all_prefix_lens[split] else 0

    print(f"\n{'─'*70}")
    print(f"SPLIT: {split}  (N={len(all_prefix_lens[split]):,} pairs, DRC coverage={pct_drc:.1f}%)")
    print(f"{'─'*70}")
    print(f"  Prefix+Marker (input tokens): {fmt(stats(all_prefix_lens[split]))}")
    print(f"  Target        (output tokens): {fmt(stats(all_target_lens[split]))}")
    if all_drc_lens[split]:
        print(f"  DRC context   (tokens, when present): {fmt(stats(all_drc_lens[split]))}")
        print(f"  DRC+Prefix    (total input tokens):   {fmt(stats(all_full_drc_lens[split]))}")
