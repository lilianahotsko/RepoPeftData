#!/usr/bin/env python3
"""Aggregate human-eval annotation sheets and report agreement.

Usage:
  python evaluation/score_human_eval.py sheetA.csv [sheetB.csv]

Each sheet is a copy of qa_human_eval_sample.csv with the correctness and
grounding columns filled in. With two sheets, also reports Cohen's kappa.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter


def load(path: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cols = list(row.keys())
            corr_col = next(c for c in cols if c.startswith("correctness"))
            grnd_col = next(c for c in cols if c.startswith("grounding"))
            c, g = row[corr_col].strip(), row[grnd_col].strip()
            if c == "" or g == "":
                continue
            out[row["item_id"]] = (int(c), int(g))
    return out


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(name: str, ann: dict[str, tuple[int, int]]) -> None:
    corr = [v[0] for v in ann.values()]
    grnd = [v[1] for v in ann.values()]
    n = len(corr)
    dist = Counter(corr)
    print(f"\n== {name} ({n} rated items) ==")
    print(
        f"correctness: 2(correct)={dist[2]} ({dist[2]/n:.0%})  "
        f"1(partial)={dist[1]} ({dist[1]/n:.0%})  "
        f"0(incorrect)={dist[0]} ({dist[0]/n:.0%})  mean={sum(corr)/n:.2f}"
    )
    print(f"grounding:   repo-specific={sum(grnd)}/{n} ({sum(grnd)/n:.0%})")


def main() -> None:
    sheets = sys.argv[1:]
    if not sheets:
        sys.exit(__doc__)
    anns = [load(p) for p in sheets]
    for path, ann in zip(sheets, anns):
        report(path, ann)

    if len(anns) == 2:
        common = sorted(set(anns[0]) & set(anns[1]))
        if not common:
            print("\nno overlapping rated items; cannot compute agreement")
            return
        a_c = [anns[0][i][0] for i in common]
        b_c = [anns[1][i][0] for i in common]
        a_g = [anns[0][i][1] for i in common]
        b_g = [anns[1][i][1] for i in common]
        print(f"\n== agreement on {len(common)} shared items ==")
        print(f"correctness kappa: {cohen_kappa(a_c, b_c):.3f}")
        print(f"grounding kappa:   {cohen_kappa(a_g, b_g):.3f}")
        hard = [i for i, x, y in zip(common, a_c, b_c) if abs(x - y) == 2]
        if hard:
            print(f"adjudicate (correctness 0 vs 2): {', '.join(hard)}")


if __name__ == "__main__":
    main()
