#!/usr/bin/env python3
"""Recompute the local-solvability / repository-knowledge-required subset split
using the *corrected* (FIM-ban and/or repo_fim) RAG and DRC predictions, so W1
and W2 use the same in-context baselines.

Methods are aligned positionally against the pretrained reference (same V1 split
ordering); alignment is verified by matching the (repo, expected) sequence.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES")
SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")
RAG_FIM = f"{SCRATCH}/RAG_V1_FIM_BAN"
DRC_FIM = f"{SCRATCH}/DRC_V1_FIM_BAN"

# label -> (path template) ; {split} filled per split
SOURCES = {
    "Pretrained":                 f"{BASE}/pretrained_{{split}}.json",
    "RAG naive-prepend (paper)":  f"{BASE}/rag_top3_{{split}}.json",
    "RAG +FIM-ban":               f"{RAG_FIM}/rag_top3_{{split}}_fim_ban.json",
    "RAG +FIM-ban +repo_fim":     f"{RAG_FIM}/rag_top3_repofim_{{split}}_fim_ban.json",
    "DRC no-ban (paper)":         f"{BASE}/oracle_context_v2_{{split}}.json",
    "DRC +FIM-ban":               f"{DRC_FIM}/drc_{{split}}_fim_ban.json",
    "DRC +FIM-ban +repo_fim":     f"{DRC_FIM}/drc_repofim_{{split}}_fim_ban.json",
}


def _as_bool(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def load(path: Path):
    if not path.exists():
        return None
    entries = json.load(open(path)).get("entries")
    if not entries:
        return None
    return [
        {"repo": e["repo"], "expected": e["expected"],
         "em": 1.0 if _as_bool(e["exact_match"]) else 0.0}
        for e in entries
    ]


def analyze(split: str):
    ref = load(Path(SOURCES["Pretrained"].format(split=split)))
    if ref is None:
        print(f"[{split}] no pretrained reference", file=sys.stderr)
        return
    ref_key = [(r["repo"], r["expected"]) for r in ref]
    em = {}
    for lbl, tmpl in SOURCES.items():
        rows = load(Path(tmpl.format(split=split)))
        if rows is None:
            print(f"  [skip] {lbl}: file missing", file=sys.stderr)
            continue
        if len(rows) != len(ref) or [(r["repo"], r["expected"]) for r in rows] != ref_key:
            print(f"  [skip] {lbl}: misaligned/partial ({len(rows)} vs {len(ref)})", file=sys.stderr)
            continue
        em[lbl] = np.array([r["em"] for r in rows])

    solved = em["Pretrained"] == 1.0
    req = ~solved
    print(f"\n## {split}  (n={len(ref)}, locally-solvable={solved.mean()*100:.1f}%, "
          f"repo-required={int(req.sum())})")
    print(f"{'Method':32s} {'overall EM':>10s} {'repo-req EM':>12s} {'local-solv EM':>14s}")
    for lbl, e in em.items():
        print(f"{lbl:32s} {e.mean()*100:10.1f} {e[req].mean()*100:12.1f} {e[solved].mean()*100:14.1f}")


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["cr_test", "ir_test"]):
        analyze(sp)
