#!/usr/bin/env python3
"""
Compute detailed dataset statistics for the EMNLP paper.
Fills in the dataset_details table: assertion types, difficulty, lengths, DRC coverage.

No GPU needed.

Usage:
    python analysis/compute_dataset_stats.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def classify_target(target: str) -> str:
    """Classify assertion target by difficulty/type."""
    t = target.strip()
    if t in ("True", "False"):
        return "bool_literal"
    if t == "None":
        return "none_literal"
    if t.startswith(("'", '"', "b'", 'b"', "f'", 'f"')):
        return "string_literal"
    try:
        int(t)
        return "numeric_literal"
    except ValueError:
        pass
    try:
        float(t)
        return "numeric_literal"
    except ValueError:
        pass
    if t.startswith(("[", "(", "{")):
        return "collection"
    if "(" in t and ")" in t:
        return "func_call"
    if t.isidentifier() or "." in t:
        return "variable"
    return "complex_expr"


def classify_assertion_type(prefix: str) -> str:
    """Classify the assertion type from prefix."""
    lines = prefix.strip().split("\n")
    last_line = lines[-1].strip() if lines else ""
    if "self.assert" in last_line or "self.Assert" in last_line:
        return "self.assert*"
    if "pytest." in last_line:
        return "pytest.*"
    return "assert"


def analyze_split(split_path: Path, oracle_cache_dir: Path | None = None) -> dict:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})

    target_types = Counter()
    assertion_types = Counter()
    target_lengths = []
    prefix_lengths = []
    n_total = 0
    n_with_oracle = 0
    pairs_per_repo = []

    for repo_name, r in repos.items():
        repo_pairs = 0
        oracle_path = oracle_cache_dir / f"{repo_name.replace('/', '_')}.json" if oracle_cache_dir else None
        has_oracle_cache = oracle_path and oracle_path.exists()

        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue

            n_total += 1
            repo_pairs += 1

            target_types[classify_target(target)] += 1
            assertion_types[classify_assertion_type(prefix)] += 1
            target_lengths.append(len(target))
            prefix_lengths.append(len(prefix))

            if has_oracle_cache:
                n_with_oracle += 1

        pairs_per_repo.append(repo_pairs)

    return {
        "n_repos": len(repos),
        "n_pairs": n_total,
        "target_types": dict(target_types.most_common()),
        "assertion_types": dict(assertion_types.most_common()),
        "avg_target_length_chars": sum(target_lengths) / len(target_lengths) if target_lengths else 0,
        "avg_prefix_length_chars": sum(prefix_lengths) / len(prefix_lengths) if prefix_lengths else 0,
        "median_pairs_per_repo": sorted(pairs_per_repo)[len(pairs_per_repo) // 2] if pairs_per_repo else 0,
        "min_pairs_per_repo": min(pairs_per_repo) if pairs_per_repo else 0,
        "max_pairs_per_repo": max(pairs_per_repo) if pairs_per_repo else 0,
        "oracle_coverage": n_with_oracle / n_total if n_total > 0 else 0,
    }


def fmt_pct(count, total):
    return f"{100.0 * count / total:.1f}%" if total > 0 else "0%"


def main():
    splits_dir = Path(os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))) / "REPO_DATASET"
    oracle_cache = Path(os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))) / "ORACLE_CONTEXT_CACHE"

    if not splits_dir.exists():
        print(f"Splits dir not found: {splits_dir}")
        return

    oracle_dir = oracle_cache if oracle_cache.exists() else None

    splits = ["train", "cr_test", "ir_test", "cr_val", "ir_val"]

    all_stats = {}
    for split in splits:
        path = splits_dir / f"{split}.json"
        if not path.exists():
            print(f"  {split}.json not found, skipping")
            continue
        print(f"Analyzing {split}...")
        all_stats[split] = analyze_split(path, oracle_dir)

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for split, stats in all_stats.items():
        print(f"\n--- {split} ---")
        print(f"  Repos: {stats['n_repos']}, Pairs: {stats['n_pairs']}")
        print(f"  Pairs/repo: median={stats['median_pairs_per_repo']}, min={stats['min_pairs_per_repo']}, max={stats['max_pairs_per_repo']}")
        print(f"  Avg target length: {stats['avg_target_length_chars']:.1f} chars")
        print(f"  Avg prefix length: {stats['avg_prefix_length_chars']:.0f} chars")
        if oracle_dir:
            print(f"  Oracle/DRC coverage: {stats['oracle_coverage']:.1%}")

        print(f"  Target types:")
        for t, c in stats["target_types"].items():
            print(f"    {t:<20} {c:>6} ({fmt_pct(c, stats['n_pairs'])})")

        print(f"  Assertion types:")
        for t, c in stats["assertion_types"].items():
            print(f"    {t:<20} {c:>6} ({fmt_pct(c, stats['n_pairs'])})")

    # LaTeX-ready output for paper
    print("\n" + "=" * 80)
    print("LATEX TABLE (copy into paper)")
    print("=" * 80)
    if "train" in all_stats and "cr_test" in all_stats and "ir_test" in all_stats:
        s = {k: all_stats[k] for k in ["train", "cr_test", "ir_test"]}
        target_cats = ["numeric_literal", "string_literal", "bool_literal", "none_literal",
                       "variable", "func_call", "collection", "complex_expr"]
        for cat in target_cats:
            vals = [fmt_pct(s[sp]["target_types"].get(cat, 0), s[sp]["n_pairs"]) for sp in ["train", "cr_test", "ir_test"]]
            print(f"\\quad \\texttt{{{cat}}} & {' & '.join(vals)} \\\\")

    output_path = Path("analysis/output/dataset_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
