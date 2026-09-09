#!/usr/bin/env python3
"""Base-LLM (and Code2LoRA) EM broken down by target TYPE and target LENGTH.

Answers reviewer v1i3-W2 ("targets are short / task is local") with the specific
angle the professor asked for: numbers (not tables) showing the base LLM without
repository context is far weaker on the harder target categories (complex
expressions, function calls, collections) and on longer targets (>=20 chars),
i.e. the benchmark is not dominated by trivial short literals.

Small stored-prediction reanalysis; safe to run on the login node.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.compute_dataset_stats import classify_target  # noqa: E402

METHODS = {
    "Pretrained": "pretrained",
    "FFT": "fft_no_oracle",
    "sLoRA": "single_lora_r16_tuned",
    "DRC": "oracle_context_v2",
    "RAG (k=3)": "rag_top3",
    "Code2LoRA-Static": "hypernet_no_oracle",
}
REFERENCE = "Pretrained"

# pretty label + display order for the Table-6 target-type taxonomy
TYPE_ORDER = [
    ("numeric_literal", "Numeric literal"),
    ("string_literal", "String literal"),
    ("bool_literal", "Bool literal"),
    ("none_literal", "None literal"),
    ("variable", "Variable"),
    ("collection", "Collection"),
    ("func_call", "Function call"),
    ("complex_expr", "Complex expression"),
]


def _as_bool(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def load_method(baselines_dir: Path, stem: str, split: str):
    path = baselines_dir / f"{stem}_{split}.json"
    if not path.exists():
        return None
    entries = json.load(open(path))["entries"]
    return [
        {
            "repo": e["repo"],
            "expected": e["expected"],
            "em": 1.0 if _as_bool(e["exact_match"]) else 0.0,
        }
        for e in entries
    ]


def analyze_split(baselines_dir: Path, split: str):
    data = {lbl: load_method(baselines_dir, stem, split) for lbl, stem in METHODS.items()}
    data = {k: v for k, v in data.items() if v is not None}
    ref = data[REFERENCE]
    ref_key = [(r["repo"], r["expected"]) for r in ref]
    aligned = {}
    for lbl, rows in data.items():
        if len(rows) == len(ref) and [(r["repo"], r["expected"]) for r in rows] == ref_key:
            aligned[lbl] = rows
        else:
            print(f"  [skip] {lbl}: misaligned ({len(rows)} vs {len(ref)})", file=sys.stderr)
    data = aligned

    types = np.array([classify_target(r["expected"]) for r in ref])
    chars = np.array([len(r["expected"].strip()) for r in ref])
    em = {lbl: np.array([r["em"] for r in rows]) for lbl, rows in data.items()}

    n = len(ref)
    per_type = []
    for key, pretty in TYPE_ORDER:
        m = types == key
        if not m.any():
            continue
        row = {"type": pretty, "n": int(m.sum()), "pct": 100.0 * m.sum() / n}
        for lbl in data:
            row[lbl] = 100.0 * em[lbl][m].mean()
        per_type.append(row)

    # collapse the "hard" (non-literal, non-variable) buckets for a headline number
    hard = np.isin(types, ["collection", "func_call", "complex_expr"])
    easy = np.isin(types, ["numeric_literal", "string_literal", "bool_literal", "none_literal"])
    groups = {
        "Literals (num/str/bool/None)": easy,
        "Variables": types == "variable",
        "Structured (collection/call/complex)": hard,
    }
    per_group = []
    for name, m in groups.items():
        if not m.any():
            continue
        row = {"group": name, "n": int(m.sum()), "pct": 100.0 * m.sum() / n}
        for lbl in data:
            row[lbl] = 100.0 * em[lbl][m].mean()
        per_group.append(row)

    # length: short (<20) vs long (>=20 chars)
    per_len = []
    for name, m in [("<20 chars", chars < 20), (">=20 chars", chars >= 20)]:
        row = {"bin": name, "n": int(m.sum()), "pct": 100.0 * m.sum() / n}
        for lbl in data:
            row[lbl] = 100.0 * em[lbl][m].mean()
        per_len.append(row)

    return {"split": split, "n": n, "methods": list(data.keys()),
            "per_type": per_type, "per_group": per_group, "per_len": per_len}


def _md_table(rows, key, methods):
    head = f"| {key} | n | % | " + " | ".join(methods) + " |"
    sep = "|---|---|---|" + "|".join(["---"] * len(methods)) + "|"
    out = [head, sep]
    for r in rows:
        cells = " | ".join(f"{r[m]:.1f}" for m in methods)
        out.append(f"| {r[key]} | {r['n']} | {r['pct']:.1f} | {cells} |")
    return "\n".join(out)


def to_markdown(results):
    lines = ["# Base-LLM accuracy by target type and length (static track)\n",
             "Base LLM = pretrained Qwen2.5-Coder-1.5B, prefix only, no repository context.\n"]
    for split, res in results.items():
        methods = res["methods"]
        lines.append(f"\n## {split} (n={res['n']})\n")
        lines.append("### By target type (Table 6 taxonomy)\n")
        lines.append(_md_table(res["per_type"], "type", methods))
        lines.append("\n### Collapsed groups\n")
        lines.append(_md_table(res["per_group"], "group", methods))
        lines.append("\n### By target length\n")
        lines.append(_md_table(res["per_len"], "bin", methods))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    baselines_dir = Path(os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES"))
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ["cr_test", "ir_test"]:
        print(f"[target-type] {split} ...", file=sys.stderr)
        results[split] = analyze_split(baselines_dir, split)
    (out_dir / "target_type_analysis.json").write_text(json.dumps(results, indent=2))
    (out_dir / "target_type_analysis.md").write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
