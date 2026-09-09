#!/usr/bin/env python3
"""Recompute the headline paired comparisons for Code2LoRA-Static vs. the
*corrected* context baselines (new RAG = repo-FIM, new DRC = FIM-ban), aligned
per-instance to Code2LoRA-Static. Prints delta/CI/McNemar/P(Δ>0)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.rebuttal.reanalysis import (  # noqa: E402
    load_method, mcnemar, paired_bootstrap_diff,
)

SCRATCH = Path(os.environ["SCRATCH"])
BASELINES = Path(os.environ.get("BASELINES_DIR", str(SCRATCH / "BASELINES")))

# (label, dir, stem) for the corrected baselines; split appended as *_fim_ban
NEW = {
    "DRC (new, FIM-ban)": (SCRATCH / "DRC_V1_FIM_BAN", "drc"),
    "RAG (new, repo-FIM)": (SCRATCH / "RAG_V1_FIM_BAN", "rag_top3_repofim"),
}


def repo_clustered_paired(a, b, repos, n_resamples=5000, ci=0.95, seed=0):
    """Paired bootstrap of mean(a)-mean(b) resampling whole REPOSITORIES
    (conservative: respects within-repo correlation)."""
    uniq = np.unique(repos)
    by_repo = {r: (a[repos == r], b[repos == r]) for r in uniq}
    rng = np.random.default_rng(seed)
    k = len(uniq)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, k, size=k)
        aa = np.concatenate([by_repo[uniq[j]][0] for j in pick])
        bb = np.concatenate([by_repo[uniq[j]][1] for j in pick])
        diffs[i] = aa.mean() - bb.mean()
    alpha = (1 - ci) / 2
    return {
        "delta_pp": float((a.mean() - b.mean()) * 100),
        "low_pp": float(np.quantile(diffs, alpha) * 100),
        "high_pp": float(np.quantile(diffs, 1 - alpha) * 100),
        "frac_gt0": float(np.mean(diffs > 0)),
        "n_clusters": int(k),
    }


def aligned_em(ref_rows, other_rows):
    """Return (em_ref, em_other) numpy arrays aligned by (repo, expected).
    Falls back to positional pairing if the sequences already match."""
    ref_key = [(r["repo"], r["expected"]) for r in ref_rows]
    oth_key = [(r["repo"], r["expected"]) for r in other_rows]
    if ref_key == oth_key:
        return (np.array([r["em"] for r in ref_rows]),
                np.array([r["em"] for r in other_rows]), "positional")
    # key-based pairing (consume duplicates in order)
    from collections import defaultdict, deque
    idx = defaultdict(deque)
    for j, k in enumerate(oth_key):
        idx[k].append(j)
    ea, eb, missing = [], [], 0
    for i, k in enumerate(ref_key):
        if idx[k]:
            j = idx[k].popleft()
            ea.append(ref_rows[i]["em"]); eb.append(other_rows[j]["em"])
        else:
            missing += 1
    return np.array(ea), np.array(eb), f"key-matched ({missing} unmatched)"


def main():
    for split in ("cr_test", "ir_test"):
        c2l = load_method(BASELINES, "hypernet_no_oracle", split)
        if c2l is None:
            print(f"[{split}] Code2LoRA-Static missing", file=sys.stderr); continue
        c2l_em_all = np.array([r["em"] for r in c2l])
        repos_all = np.array([r["repo"] for r in c2l])
        print(f"\n===== {split}  (n={len(c2l)}) =====")
        print(f"Code2LoRA-Static EM = {c2l_em_all.mean()*100:.2f}%")
        for lbl, (d, stem) in NEW.items():
            rows = load_method(d, stem, f"{split}_fim_ban")
            if rows is None:
                print(f"  [skip] {lbl}: file missing"); continue
            a, b, how = aligned_em(c2l, rows)
            other_em = np.array([r["em"] for r in rows]).mean() * 100
            pb = paired_bootstrap_diff(a, b)
            mc = mcnemar(a, b)
            rc = repo_clustered_paired(a, b, repos_all) if how == "positional" else None
            print(f"\n  Code2LoRA-Static vs {lbl}  [{how}, n_paired={len(a)}]")
            print(f"    {lbl} EM            = {other_em:.2f}%")
            print(f"    delta (paired)      = {pb['delta_pp']:+.1f} pp  "
                  f"95% CI [{pb['low_pp']:+.1f}, {pb['high_pp']:+.1f}]")
            print(f"    P(delta>0)          = {pb['frac_gt0']:.3f}")
            print(f"    McNemar b/c         = {mc['b']}/{mc['c']}  "
                  f"chi2={mc['chi2']:.1f}  p={mc['p']:.2e}")
            if rc is not None:
                print(f"    repo-clustered      = {rc['delta_pp']:+.1f} pp  "
                      f"95% CI [{rc['low_pp']:+.1f}, {rc['high_pp']:+.1f}]  "
                      f"P(delta>0)={rc['frac_gt0']:.3f}  (clusters={rc['n_clusters']})")


if __name__ == "__main__":
    main()
