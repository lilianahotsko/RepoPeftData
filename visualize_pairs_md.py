#!/usr/bin/env python3
"""
Visualize training pairs from a test_next_block JSONL file as Markdown.

Usage:
    python visualize_pairs_md.py <path_to_jsonl> [--framework pytest] [--max-samples 30] [--output report.md]
"""

import argparse
import json
import os
import statistics
from collections import Counter


def load_data(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_stats(records):
    stats = {}
    stats["total"] = len(records)

    repo_counts = Counter(r["repo"] for r in records)
    stats["num_repos"] = len(repo_counts)
    stats["top_repos"] = repo_counts.most_common(15)

    fw_counts = Counter(r.get("framework", "unknown") for r in records)
    stats["frameworks"] = fw_counts.most_common()

    tt_counts = Counter()
    for r in records:
        for t in r.get("test_type", []):
            tt_counts[t] += 1
    stats["test_types"] = tt_counts.most_common()

    ck_counts = Counter(r.get("metadata", {}).get("cut_kind", "unknown") for r in records)
    stats["cut_kinds"] = ck_counts.most_common()

    prefix_chars = [len(r.get("prefix", "")) for r in records]
    target_chars = [len(r.get("target", "")) for r in records]
    prefix_lines = [r.get("prefix", "").count("\n") + 1 for r in records]
    target_lines = [r.get("target", "").count("\n") + 1 for r in records]

    for name, vals in [("prefix_chars", prefix_chars), ("target_chars", target_chars),
                       ("prefix_lines", prefix_lines), ("target_lines", target_lines)]:
        stats[name] = {
            "min": min(vals), "max": max(vals),
            "mean": statistics.mean(vals), "median": statistics.median(vals),
        }

    trimmed = sum(1 for r in records if r.get("metadata", {}).get("prefix_trimmed", False))
    stats["prefix_trimmed_count"] = trimmed
    stats["prefix_trimmed_pct"] = 100.0 * trimmed / len(records) if records else 0

    return stats


def fmt_table(headers, rows):
    """Simple markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def fmt_stat(d):
    return fmt_table(
        ["Min", "Max", "Mean", "Median"],
        [[d["min"], d["max"], f'{d["mean"]:.1f}', f'{d["median"]:.1f}']]
    )


def last_n_lines(text, n=25):
    lines = text.split("\n")
    if len(lines) <= n:
        return text
    return f"# ... ({len(lines) - n} lines omitted) ...\n" + "\n".join(lines[-n:])


def generate_md(records, stats, max_display=30):
    parts = []
    parts.append(f"# Training Pairs — test_next_block (pytest)\n")
    parts.append(f"**{stats['total']:,}** pairs from **{stats['num_repos']}** repos\n")

    # --- Stats ---
    parts.append("## Summary Statistics\n")

    parts.append("### Top Repositories\n")
    parts.append(fmt_table(["Repo", "Count"], stats["top_repos"]))
    parts.append("")

    parts.append("### Frameworks\n")
    parts.append(fmt_table(["Framework", "Count"], stats["frameworks"]))
    parts.append("")

    parts.append("### Test Types\n")
    parts.append(fmt_table(["Type", "Count"], stats["test_types"]))
    parts.append("")

    parts.append("### Cut Kinds\n")
    parts.append(fmt_table(["Kind", "Count"], stats["cut_kinds"]))
    parts.append("")

    parts.append("### Prefix Length (chars)\n")
    parts.append(fmt_stat(stats["prefix_chars"]))
    parts.append("")

    parts.append("### Prefix Length (lines)\n")
    parts.append(fmt_stat(stats["prefix_lines"]))
    parts.append("")

    parts.append("### Target Length (chars)\n")
    parts.append(fmt_stat(stats["target_chars"]))
    parts.append("")

    parts.append("### Target Length (lines)\n")
    parts.append(fmt_stat(stats["target_lines"]))
    parts.append("")

    parts.append(f"### Prefix Trimming\n")
    parts.append(f"{stats['prefix_trimmed_count']:,} / {stats['total']:,} "
                 f"({stats['prefix_trimmed_pct']:.1f}%) prefixes were trimmed\n")

    # --- Pairs ---
    parts.append(f"---\n\n## Sample Pairs (showing {min(max_display, len(records))} of {stats['total']:,})\n")

    for i, r in enumerate(records[:max_display]):
        meta = r.get("metadata", {})
        prefix = r.get("prefix", "")
        target = r.get("target", "")

        parts.append(f"### Pair #{i+1}\n")
        parts.append(f"- **Repo:** `{r.get('repo','')}`")
        parts.append(f"- **File:** `{meta.get('file','')}`")
        parts.append(f"- **Function:** `{meta.get('function','') or 'N/A'}`")
        parts.append(f"- **Cut:** `{meta.get('cut_kind','')}` @ line {meta.get('cut_line','?')}")
        parts.append(f"- **Test types:** {', '.join(r.get('test_type', []))}")
        parts.append(f"- **Prefix:** {len(prefix)} chars, {prefix.count(chr(10))+1} lines  "
                     f"| **Target:** {len(target)} chars, {target.count(chr(10))+1} lines")
        parts.append(f"- **Prefix trimmed:** {meta.get('prefix_trimmed', False)}\n")

        parts.append("**PREFIX** (last 25 lines):\n")
        parts.append("```python")
        parts.append(last_n_lines(prefix, 25))
        parts.append("```\n")

        parts.append("**TARGET** (expected output):\n")
        parts.append("```python")
        parts.append(target)
        parts.append("```\n")

        parts.append("---\n")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Visualize test_next_block training pairs (Markdown)")
    parser.add_argument("jsonl_path", help="Path to the JSONL file")
    parser.add_argument("--max-samples", type=int, default=30,
                        help="Max sample pairs to include (default: 30)")
    parser.add_argument("--framework", type=str, default=None,
                        help="Filter by framework (e.g. pytest)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .md file (default: /tmp/<name>_report.md)")
    args = parser.parse_args()

    print(f"Loading {args.jsonl_path} ...")
    records = load_data(args.jsonl_path)
    print(f"  {len(records):,} records loaded.")

    if args.framework:
        records = [r for r in records if r.get("framework", "").lower() == args.framework.lower()]
        print(f"  Filtered to framework='{args.framework}': {len(records):,} remain.")

    stats = compute_stats(records)
    md = generate_md(records, stats, max_display=args.max_samples)

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.jsonl_path))[0]
        out_path = f"/tmp/{base}_report.md"

    with open(out_path, "w") as f:
        f.write(md)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
