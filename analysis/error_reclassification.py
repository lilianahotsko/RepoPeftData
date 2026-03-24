#!/usr/bin/env python3
"""
LiveCodeBench-style error reclassification for Code2LoRA CR-test predictions.

Adapts LiveCodeBench's 4-category error taxonomy (syntax, runtime, wrong answer,
time-limit) to assertion completion, producing a more principled error breakdown.

Categories:
  1. Syntax Error      — prediction is not a valid Python expression
  2. Near-Miss         — EditSim > 0.8 but not exact match (whitespace, parens, quotes)
  3. Wrong Identifier  — correct structure, wrong variable/function/attribute name
  4. Wrong Literal     — correct structure, wrong constant value
  5. Type Mismatch     — structurally different type (e.g., string vs number)
  6. Hallucination     — long/repetitive output (>3x expected length or >100 chars)
  7. Empty/Truncated   — empty or very short output

Run:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/error_reclassification.py
"""

import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "/scratch/lhotsko/BASELINES/hypernet_no_oracle_cr_test.json"
OUTPUT_DIR   = Path("/home/lhotsko/RepoPeftData/analysis/figures")
PAPER_DIR    = Path("/home/lhotsko/RepoPeftData/RepoPeft_Paper/figures")


def is_valid_python_expr(s: str) -> bool:
    """Check if string is a syntactically valid Python expression."""
    try:
        ast.parse(s.strip(), mode="eval")
        return True
    except (SyntaxError, ValueError):
        return False


def get_expr_type(s: str) -> str:
    """Classify a Python expression by its structural type."""
    s = s.strip()
    if not s:
        return "empty"
    try:
        tree = ast.parse(s, mode="eval")
        node = tree.body
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, (int, float, complex)):
                return "number"
            if isinstance(node.value, str):
                return "string"
            if node.value is None:
                return "none"
            return "constant"
        if isinstance(node, ast.Name):
            return "identifier"
        if isinstance(node, ast.Attribute):
            return "attribute"
        if isinstance(node, ast.Call):
            return "call"
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return "collection"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare)):
            return "expression"
        if isinstance(node, ast.Subscript):
            return "subscript"
        return "other"
    except (SyntaxError, ValueError):
        return "unparseable"


def extract_identifiers(s: str) -> set:
    """Extract all Name nodes from a Python expression."""
    try:
        tree = ast.parse(s.strip(), mode="eval")
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    except (SyntaxError, ValueError):
        return set()


def extract_literals(s: str) -> set:
    """Extract all Constant nodes from a Python expression."""
    try:
        tree = ast.parse(s.strip(), mode="eval")
        return {repr(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    except (SyntaxError, ValueError):
        return set()


def classify_error(expected: str, got: str, edit_sim: float) -> str:
    """Classify a prediction error using LiveCodeBench-inspired taxonomy."""
    expected = expected.strip()
    got = got.strip()

    # 1. Empty/Truncated
    if not got or len(got.strip()) == 0:
        return "Empty/Truncated"

    # 2. Hallucination: overly long or repetitive output
    if len(got) > max(100, 3 * len(expected)):
        return "Hallucination"
    # Check for repetitive patterns
    if len(got) > 30:
        tokens = got.split()
        if len(tokens) > 5 and len(set(tokens)) < len(tokens) / 2:
            return "Hallucination"

    # 3. Near-miss: very close but not exact (EditSim > 0.8)
    if edit_sim > 0.8:
        return "Near-Miss"

    # 4. Syntax error: prediction is not valid Python
    if not is_valid_python_expr(got):
        return "Syntax Error"

    # 5. Structural type comparison
    exp_type = get_expr_type(expected)
    got_type = get_expr_type(got)

    # 6. Wrong identifier: same structure, different name
    exp_ids = extract_identifiers(expected)
    got_ids = extract_identifiers(got)
    if exp_ids and got_ids and exp_ids != got_ids:
        # Check if the structural pattern is similar (same number of AST nodes)
        exp_lits = extract_literals(expected)
        got_lits = extract_literals(got)
        if exp_lits == got_lits:
            return "Wrong Identifier"

    # 7. Wrong literal: same structure, different constant
    exp_lits = extract_literals(expected)
    got_lits = extract_literals(got)
    if exp_lits and got_lits and exp_lits != got_lits and exp_ids == got_ids:
        return "Wrong Literal"

    # 8. Type mismatch
    if exp_type != got_type and exp_type != "unparseable" and got_type != "unparseable":
        return "Type Mismatch"

    # 9. If identifiers differ at all, call it wrong identifier
    if exp_ids != got_ids and (exp_ids or got_ids):
        return "Wrong Identifier"

    # 10. If literals differ at all, call it wrong literal
    if exp_lits != got_lits:
        return "Wrong Literal"

    # Fallback
    return "Type Mismatch"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading predictions...")
    data = json.load(open(RESULTS_FILE))
    entries = data["entries"]

    failures = [e for e in entries if not e.get("exact_match", False)]
    print(f"Total entries: {len(entries)}, Failures: {len(failures)}")

    # Classify each failure
    categories = Counter()
    examples = {}

    for e in failures:
        cat = classify_error(e["expected"], e["got"], e.get("edit_similarity", 0))
        categories[cat] += 1
        if cat not in examples:
            examples[cat] = {"expected": e["expected"], "got": e["got"], "repo": e["repo"]}

    # Print results
    print(f"\n{'='*60}")
    print(f"ERROR CLASSIFICATION (LiveCodeBench-style)")
    print(f"{'='*60}")
    total = sum(categories.values())
    for cat, count in categories.most_common():
        pct = 100.0 * count / total
        ex = examples.get(cat, {})
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)  e.g. expected={ex.get('expected','')!r:.40s}  got={ex.get('got','')!r:.40s}")

    # Generate figure
    labels = []
    sizes = []
    colors_map = {
        "Wrong Identifier": "#e74c3c",
        "Wrong Literal":    "#3498db",
        "Near-Miss":        "#2ecc71",
        "Type Mismatch":    "#f39c12",
        "Syntax Error":     "#9b59b6",
        "Hallucination":    "#e67e22",
        "Empty/Truncated":  "#95a5a6",
    }

    for cat, count in categories.most_common():
        pct = 100.0 * count / total
        labels.append(f"{cat}\n({pct:.1f}%)")
        sizes.append(count)

    colors = [colors_map.get(cat, "#bdc3c7") for cat, _ in categories.most_common()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), sizes, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Predictions", fontsize=12)
    ax.set_title(f"Error Classification ({total:,} incorrect CR-test predictions)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Add count labels on bars
    for bar, count in zip(bars, sizes):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "error_reclassification.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")

    if PAPER_DIR.exists():
        import shutil
        shutil.copy2(out_path, PAPER_DIR / "error_reclassification.pdf")
        print(f"Copied to {PAPER_DIR / 'error_reclassification.pdf'}")

    plt.close()


if __name__ == "__main__":
    main()
