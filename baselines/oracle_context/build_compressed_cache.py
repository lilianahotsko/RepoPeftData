#!/usr/bin/env python3
"""
Pre-build compressed oracle context cache (v3) from the v2 cache.

Reads v2 cache + split JSONs (for prefix access), compresses each entry
using relevance-aware class/function compression with a hard token budget,
and writes to ORACLE_CONTEXT_CACHE_V3/.

No repo access needed — operates entirely on cached extracted_code strings.

Usage:
    python baselines/oracle_context/build_compressed_cache.py
    python baselines/oracle_context/build_compressed_cache.py --max-tokens 6000
    python baselines/oracle_context/build_compressed_cache.py --v2-cache-dir $SCRATCH/ORACLE_CONTEXT_CACHE_V2
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.compress_context import compress_oracle_context


def load_prefixes_for_repo(splits_dir: Path, repo_name: str) -> dict[str, str]:
    """Load all prefixes for a repo across all splits. Returns {file::lineno: prefix}."""
    prefixes: dict[str, str] = {}
    for split_name in ["train", "cr_val", "cr_test", "ir_val", "ir_test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            continue
        data = json.loads(split_path.read_text(encoding="utf-8"))
        repo_data = data.get("repositories", {}).get(repo_name, {})
        for pair in repo_data.get("qna_pairs", []):
            meta = pair.get("metadata", {})
            key = f"{meta.get('file', '')}::{meta.get('lineno', 0)}"
            prefixes[key] = pair.get("prefix", "")
    return prefixes


def main():
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))

    ap = argparse.ArgumentParser(description="Build compressed oracle context cache (v3)")
    ap.add_argument("--v2-cache-dir", type=str,
                     default=os.path.join(scratch, "ORACLE_CONTEXT_CACHE_V2"))
    ap.add_argument("--output-dir", type=str,
                     default=os.path.join(scratch, "ORACLE_CONTEXT_CACHE_V3"))
    ap.add_argument("--splits-dir", type=str,
                     default=os.path.join(scratch, "REPO_DATASET"))
    ap.add_argument("--max-tokens", type=int, default=6000,
                     help="Hard token budget for compressed context")
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--limit-repos", type=int, default=None,
                     help="Process only first N repos (for testing)")
    args = ap.parse_args()

    v2_dir = Path(args.v2_cache_dir)
    out_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)

    if not v2_dir.exists():
        print(f"ERROR: v2 cache not found: {v2_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"v2 cache:   {v2_dir}")
    print(f"Output:     {out_dir}")
    print(f"Splits:     {splits_dir}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tokenizer:  {args.tokenizer}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    v2_files = sorted(v2_dir.glob("*.json"))
    if args.limit_repos:
        v2_files = v2_files[:args.limit_repos]

    print(f"\nProcessing {len(v2_files)} repos...\n")

    total_entries = 0
    total_compressed = 0
    total_skipped = 0
    total_orig_chars = 0
    total_comp_chars = 0
    over_budget = 0

    for v2_path in tqdm(v2_files, desc="Repos"):
        data = json.loads(v2_path.read_text(encoding="utf-8"))
        repo_name = data.get("repo", "")
        contexts = data.get("contexts", {})

        if not contexts:
            continue

        # Load prefixes for relevance scoring
        prefixes = load_prefixes_for_repo(splits_dir, repo_name)

        new_contexts = {}
        for key, ctx in contexts.items():
            total_entries += 1
            extracted = ctx.get("extracted_code", "")

            if not extracted or not extracted.strip():
                new_contexts[key] = ctx
                total_skipped += 1
                continue

            prefix = prefixes.get(key, "")
            orig_chars = len(extracted)
            total_orig_chars += orig_chars

            compressed = compress_oracle_context(
                extracted, prefix, tokenizer, max_tokens=args.max_tokens,
            )

            comp_chars = len(compressed)
            total_comp_chars += comp_chars

            # Verify token count
            comp_tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
            if comp_tokens > args.max_tokens:
                over_budget += 1

            total_compressed += 1

            new_ctx = dict(ctx)
            new_ctx["extracted_code"] = compressed
            new_ctx["extracted_code_full"] = extracted
            new_ctx["n_chars_compressed"] = comp_chars
            new_ctx["compression_ratio"] = comp_chars / orig_chars if orig_chars > 0 else 1.0
            new_contexts[key] = new_ctx

        out_data = {
            "repo": repo_name,
            "n_pairs": data.get("n_pairs", len(new_contexts)),
            "version": "v3_compressed",
            "max_tokens": args.max_tokens,
            "contexts": new_contexts,
        }
        out_path = out_dir / v2_path.name
        out_path.write_text(json.dumps(out_data, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print(f"Compressed cache built: {out_dir}")
    print(f"{'='*60}")
    print(f"  Total entries:     {total_entries:,}")
    print(f"  Compressed:        {total_compressed:,}")
    print(f"  Skipped (empty):   {total_skipped:,}")
    if total_orig_chars > 0:
        ratio = total_comp_chars / total_orig_chars * 100
        print(f"  Original chars:    {total_orig_chars:,}")
        print(f"  Compressed chars:  {total_comp_chars:,}")
        print(f"  Overall ratio:     {ratio:.1f}%")
    if over_budget > 0:
        print(f"  Over budget:       {over_budget} (should be 0)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
