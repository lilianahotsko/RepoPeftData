#!/usr/bin/env python3
"""
Build adaptive oracle context cache (v4) from the v2 cache.

Unlike v3 which uses a fixed token budget (6K) regardless of prefix length,
v4 uses an adaptive budget: each entry's oracle budget is computed as the
remaining space in the context window after accounting for prefix, target,
the target marker, and a safety margin.

    oracle_budget = max_seq_len - prefix_tokens - target_tokens - overhead - margin

This prevents oracle context from crowding out the actual task.  Short prefixes
get more oracle; long prefixes get less.

Usage:
    python baselines/oracle_context/build_adaptive_cache.py
    python baselines/oracle_context/build_adaptive_cache.py --max-seq-len 8192 --margin 64
    python baselines/oracle_context/build_adaptive_cache.py --min-oracle-tokens 128 --max-oracle-tokens 4096
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.compress_context import compress_oracle_context

TARGET_MARKER = "### Target:"


def load_pairs_for_repo(
    splits_dir: Path, repo_name: str,
) -> dict[str, dict[str, str]]:
    """Load prefix and target for every QnA pair across all splits.

    Returns {file::lineno: {"prefix": ..., "target": ...}}.
    """
    pairs: dict[str, dict[str, str]] = {}
    for split_name in ["train", "cr_val", "cr_test", "ir_val", "ir_test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            continue
        data = json.loads(split_path.read_text(encoding="utf-8"))
        repo_data = data.get("repositories", {}).get(repo_name, {})
        for pair in repo_data.get("qna_pairs", []):
            meta = pair.get("metadata", {})
            key = f"{meta.get('file', '')}::{meta.get('lineno', 0)}"
            pairs[key] = {
                "prefix": pair.get("prefix", ""),
                "target": pair.get("target", ""),
            }
    return pairs


def compute_adaptive_budget(
    prefix: str,
    target: str,
    tokenizer,
    max_seq_len: int,
    margin: int,
    min_oracle_tokens: int,
    max_oracle_tokens: int,
) -> int:
    """Compute the oracle token budget for a single example.

    budget = max_seq_len - prefix_tokens - target_tokens - overhead - margin
    clamped to [min_oracle_tokens, max_oracle_tokens].

    The overhead accounts for the target marker tokens and separator newlines
    added during formatting:  prefix + "\\n" + TARGET_MARKER + "\\n" + target
    """
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
    # overhead: "\n### Target:\n" ≈ 4-6 tokens + 3 newlines for oracle separator
    overhead = len(tokenizer.encode(
        "\n" + TARGET_MARKER + "\n", add_special_tokens=False,
    )) + 3  # 3 extra tokens for "\n\n\n" oracle-prefix separator

    remaining = max_seq_len - prefix_tokens - target_tokens - overhead - margin
    return max(min_oracle_tokens, min(remaining, max_oracle_tokens))


def main():
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))

    ap = argparse.ArgumentParser(
        description="Build adaptive oracle context cache (v4)",
    )
    ap.add_argument(
        "--v2-cache-dir", type=str,
        default=os.path.join(scratch, "ORACLE_CONTEXT_CACHE_V2"),
    )
    ap.add_argument(
        "--output-dir", type=str,
        default=os.path.join(scratch, "ORACLE_CONTEXT_CACHE_V4"),
    )
    ap.add_argument(
        "--splits-dir", type=str,
        default=os.path.join(scratch, "REPO_DATASET"),
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=8192,
        help="Model context window size",
    )
    ap.add_argument(
        "--margin", type=int, default=64,
        help="Safety margin in tokens (for BOS/EOS, padding, etc.)",
    )
    ap.add_argument(
        "--min-oracle-tokens", type=int, default=128,
        help="Minimum oracle budget — even long prefixes get at least this much",
    )
    ap.add_argument(
        "--max-oracle-tokens", type=int, default=4096,
        help="Maximum oracle budget — cap even for very short prefixes",
    )
    ap.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-1.5B",
    )
    ap.add_argument(
        "--limit-repos", type=int, default=None,
        help="Process only first N repos (for testing)",
    )
    args = ap.parse_args()

    v2_dir = Path(args.v2_cache_dir)
    out_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)

    if not v2_dir.exists():
        print(f"ERROR: v2 cache not found: {v2_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"v2 cache:         {v2_dir}")
    print(f"Output:           {out_dir}")
    print(f"Splits:           {splits_dir}")
    print(f"Max seq len:      {args.max_seq_len}")
    print(f"Margin:           {args.margin}")
    print(f"Oracle range:     [{args.min_oracle_tokens}, {args.max_oracle_tokens}]")
    print(f"Tokenizer:        {args.tokenizer}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    v2_files = sorted(v2_dir.glob("*.json"))
    if args.limit_repos:
        v2_files = v2_files[: args.limit_repos]

    print(f"\nProcessing {len(v2_files)} repos...\n")

    # Stats
    total_entries = 0
    total_compressed = 0
    total_skipped = 0
    total_orig_chars = 0
    total_comp_chars = 0
    budget_distribution: list[int] = []

    for v2_path in tqdm(v2_files, desc="Repos"):
        data = json.loads(v2_path.read_text(encoding="utf-8"))
        repo_name = data.get("repo", "")
        contexts = data.get("contexts", {})

        if not contexts:
            continue

        # Load prefixes + targets for adaptive budget
        pairs = load_pairs_for_repo(splits_dir, repo_name)

        new_contexts = {}
        for key, ctx in contexts.items():
            total_entries += 1
            extracted = ctx.get("extracted_code", "")

            if not extracted or not extracted.strip():
                new_contexts[key] = ctx
                total_skipped += 1
                continue

            pair = pairs.get(key, {})
            prefix = pair.get("prefix", "")
            target = pair.get("target", "")

            # Compute per-example adaptive budget
            budget = compute_adaptive_budget(
                prefix, target, tokenizer,
                max_seq_len=args.max_seq_len,
                margin=args.margin,
                min_oracle_tokens=args.min_oracle_tokens,
                max_oracle_tokens=args.max_oracle_tokens,
            )
            budget_distribution.append(budget)

            orig_chars = len(extracted)
            total_orig_chars += orig_chars

            compressed = compress_oracle_context(
                extracted, prefix, tokenizer, max_tokens=budget,
            )

            comp_chars = len(compressed)
            total_comp_chars += comp_chars
            total_compressed += 1

            new_ctx = dict(ctx)
            new_ctx["extracted_code"] = compressed
            new_ctx["extracted_code_full"] = extracted
            new_ctx["n_chars_compressed"] = comp_chars
            new_ctx["compression_ratio"] = (
                comp_chars / orig_chars if orig_chars > 0 else 1.0
            )
            new_ctx["adaptive_budget"] = budget
            new_contexts[key] = new_ctx

        out_data = {
            "repo": repo_name,
            "n_pairs": data.get("n_pairs", len(new_contexts)),
            "version": "v4_adaptive",
            "max_seq_len": args.max_seq_len,
            "min_oracle_tokens": args.min_oracle_tokens,
            "max_oracle_tokens": args.max_oracle_tokens,
            "margin": args.margin,
            "contexts": new_contexts,
        }
        out_path = out_dir / v2_path.name
        out_path.write_text(
            json.dumps(out_data, ensure_ascii=False), encoding="utf-8",
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Adaptive cache built: {out_dir}")
    print(f"{'=' * 60}")
    print(f"  Total entries:     {total_entries:,}")
    print(f"  Compressed:        {total_compressed:,}")
    print(f"  Skipped (empty):   {total_skipped:,}")
    if total_orig_chars > 0:
        ratio = total_comp_chars / total_orig_chars * 100
        print(f"  Original chars:    {total_orig_chars:,}")
        print(f"  Compressed chars:  {total_comp_chars:,}")
        print(f"  Overall ratio:     {ratio:.1f}%")
    if budget_distribution:
        bd = sorted(budget_distribution)
        n = len(bd)
        print(f"  Budget distribution:")
        print(f"    min={bd[0]}, p25={bd[n // 4]}, median={bd[n // 2]}, "
              f"p75={bd[3 * n // 4]}, p95={bd[int(n * 0.95)]}, max={bd[-1]}")
        at_floor = sum(1 for b in bd if b == args.min_oracle_tokens)
        at_cap = sum(1 for b in bd if b == args.max_oracle_tokens)
        print(f"    at min ({args.min_oracle_tokens}): {at_floor} "
              f"({at_floor / n * 100:.1f}%)")
        print(f"    at max ({args.max_oracle_tokens}): {at_cap} "
              f"({at_cap / n * 100:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
