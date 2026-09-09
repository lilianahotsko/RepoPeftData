#!/usr/bin/env python3
"""Rebuttal reanalysis over stored static-track predictions (no GPU).

Covers, from the per-instance V1 prediction JSONs in $BASELINES_DIR:
  * Fraction solvable from the local prefix (pretrained EM)            [v1i3-W2]
  * Repo-knowledge-required subset gap (pretrained-fail subset vs.
    locally-solvable complement), per method                          [v1i3-W2]
  * EM vs. target length (char and token bins)                        [v1i3-W2]
  * Strict-EM (no overgeneration truncation) ranking check            [v1i3-W2]
  * 95% bootstrap CIs (instance-level and repo-clustered) per method  [v1i3-W3]
  * Paired significance for headline comparisons: McNemar + paired
    bootstrap of the EM difference                                    [v1i3-W3]

All methods are aligned positionally; alignment is verified by requiring the
(repo, expected) sequence to match the pretrained reference for each split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    normalize_for_match,
    strip_fim_tokens,
)

# label -> filename stem (confirmed against the paper's Table 1)
METHODS = {
    "Pretrained": "pretrained",
    "RAG (k=3)": "rag_top3",
    "DRC": "oracle_context_v2",
    "FFT": "fft_no_oracle",
    "sLoRA": "single_lora_r16_tuned",
    "Code2LoRA-Static": "hypernet_no_oracle",
    "Code2LoRA-PAW": "hypernet_paw_no_oracle",
}
REFERENCE = "Pretrained"
# headline comparisons (A vs B): is A significantly better than B?
COMPARISONS = [
    ("Code2LoRA-Static", "FFT"),
    ("Code2LoRA-Static", "DRC"),
    ("Code2LoRA-Static", "sLoRA"),
    ("Code2LoRA-Static", "Pretrained"),
    ("Code2LoRA-Static", "RAG (k=3)"),
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
            "got_raw": e.get("got_raw", e.get("got", "")),
            "em": 1.0 if _as_bool(e["exact_match"]) else 0.0,
            "es": float(e.get("edit_similarity", 0.0)),
            "cb": float(e.get("code_bleu", 0.0)),
        }
        for e in entries
    ]


def strict_em(got_raw: str, expected: str) -> float:
    """EM with NO overgeneration truncation: first line of the raw generation
    (FIM sentinels removed) must equal the target after whitespace/punct
    normalization. Applied identically to every method."""
    p = strip_fim_tokens(got_raw or "")
    if "\n" in p:
        p = p.split("\n", 1)[0]
    return 1.0 if normalize_for_match(p) == normalize_for_match(expected or "") else 0.0


def repo_clustered_ci(values: np.ndarray, repos: np.ndarray, n_resamples=5000, ci=0.95, seed=0):
    """Bootstrap CI resampling whole repositories (cluster = repo)."""
    uniq = np.unique(repos)
    by_repo = {r: values[repos == r] for r in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    k = len(uniq)
    for i in range(n_resamples):
        pick = rng.integers(0, k, size=k)
        cat = np.concatenate([by_repo[uniq[j]] for j in pick])
        means[i] = cat.mean()
    a = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, a)),
        "high": float(np.quantile(means, 1 - a)),
        "n_clusters": int(k),
    }


def mcnemar(a: np.ndarray, b: np.ndarray):
    """McNemar test on paired binary outcomes (a,b in {0,1})."""
    b_only = int(np.sum((a == 1) & (b == 0)))  # A right, B wrong
    c_only = int(np.sum((a == 0) & (b == 1)))  # A wrong, B right
    n = b_only + c_only
    if n == 0:
        return {"b": b_only, "c": c_only, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b_only - c_only) - 1) ** 2 / n  # continuity correction
    p = math.erfc(math.sqrt(chi2 / 2.0))  # chi2 df=1 survival function
    return {"b": b_only, "c": c_only, "chi2": float(chi2), "p": float(p)}


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_resamples=5000, ci=0.95, seed=0):
    """Bootstrap CI for mean(a)-mean(b) over paired instances."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    alpha = (1 - ci) / 2
    return {
        "delta_pp": float((a.mean() - b.mean()) * 100),
        "low_pp": float(np.quantile(diffs, alpha) * 100),
        "high_pp": float(np.quantile(diffs, 1 - alpha) * 100),
        "frac_gt0": float(np.mean(diffs > 0)),
    }


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

    repos = np.array([r["repo"] for r in ref])
    tgt_chars = np.array([len(r["expected"]) for r in ref])
    tgt_toks = np.array([len(r["expected"].split()) for r in ref])
    em = {lbl: np.array([r["em"] for r in rows]) for lbl, rows in data.items()}
    strict = {lbl: np.array([strict_em(r["got_raw"], r["expected"]) for r in rows]) for lbl, rows in data.items()}

    # ---- per-method summary: relaxed EM, strict EM, CIs ----
    summary = {}
    for lbl, rows in data.items():
        e = em[lbl]
        summary[lbl] = {
            "em_pct": float(e.mean() * 100),
            "strict_em_pct": float(strict[lbl].mean() * 100),
            "ci_instance": {k: (v if k in ("n", "n_resamples", "ci") else v) for k, v in bootstrap_ci(e).items()},
            "ci_repo_clustered": repo_clustered_ci(e, repos),
        }

    # ---- fraction solvable / repo-knowledge-required subset ----
    solved = em[REFERENCE] == 1.0
    req = ~solved  # repo-knowledge-required = pretrained fails locally
    subset = {
        "n_total": int(len(ref)),
        "frac_local_solvable_pct": float(solved.mean() * 100),
        "n_repo_required": int(req.sum()),
        "per_method": {},
    }
    for lbl in data:
        subset["per_method"][lbl] = {
            "em_on_repo_required_pct": float(em[lbl][req].mean() * 100),
            "em_on_local_solvable_pct": float(em[lbl][solved].mean() * 100),
        }

    # ---- EM by target length ----
    def bin_em(sizes, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sizes >= lo) & (sizes < hi)
            row = {"bin": f"[{lo},{hi})", "n": int(m.sum())}
            for lbl in data:
                row[lbl] = float(em[lbl][m].mean() * 100) if m.any() else None
            out.append(row)
        return out

    length_analysis = {
        "by_target_chars": bin_em(tgt_chars, [0, 3, 6, 10, 20, 10 ** 6]),
        "by_target_tokens": bin_em(tgt_toks, [1, 2, 3, 5, 10 ** 6]),
    }

    # ---- paired significance for headline comparisons ----
    comps = {}
    for A, B in COMPARISONS:
        if A in data and B in data:
            comps[f"{A} vs {B}"] = {
                "mcnemar": mcnemar(em[A], em[B]),
                "paired_bootstrap": paired_bootstrap_diff(em[A], em[B]),
            }

    return {
        "split": split,
        "n": int(len(ref)),
        "methods": list(data.keys()),
        "summary": summary,
        "fraction_and_subset": subset,
        "length_analysis": length_analysis,
        "headline_comparisons": comps,
    }


def to_markdown(results: dict) -> str:
    lines = ["# Rebuttal reanalysis (static track, stored predictions)\n"]
    for split, res in results.items():
        lines.append(f"\n## {split}  (n={res['n']})\n")
        lines.append("### Per-method EM, strict-EM, and 95% CIs\n")
        lines.append("| Method | EM % | strict-EM % | 95% CI (instance) | 95% CI (repo-clustered) |")
        lines.append("|---|---|---|---|---|")
        for lbl, s in res["summary"].items():
            ci = s["ci_instance"]; rc = s["ci_repo_clustered"]
            lines.append(
                f"| {lbl} | {s['em_pct']:.1f} | {s['strict_em_pct']:.1f} | "
                f"[{ci['low']*100:.1f}, {ci['high']*100:.1f}] | "
                f"[{rc['low']*100:.1f}, {rc['high']*100:.1f}] |"
            )
        sub = res["fraction_and_subset"]
        lines.append(
            f"\n### Fraction solvable from local prefix: **{sub['frac_local_solvable_pct']:.1f}%** "
            f"(repo-knowledge-required subset: {sub['n_repo_required']}/{sub['n_total']})\n"
        )
        lines.append("| Method | EM on repo-required % | EM on local-solvable % |")
        lines.append("|---|---|---|")
        for lbl, v in sub["per_method"].items():
            lines.append(f"| {lbl} | {v['em_on_repo_required_pct']:.1f} | {v['em_on_local_solvable_pct']:.1f} |")

        lines.append("\n### EM by target length (chars)\n")
        methods = res["methods"]
        lines.append("| bin | n | " + " | ".join(methods) + " |")
        lines.append("|---|---|" + "|".join(["---"] * len(methods)) + "|")
        for row in res["length_analysis"]["by_target_chars"]:
            cells = " | ".join(f"{row[m]:.1f}" if row[m] is not None else "-" for m in methods)
            lines.append(f"| {row['bin']} | {row['n']} | {cells} |")

        lines.append("\n### Headline comparisons (paired)\n")
        lines.append("| Comparison | delta EM (pp) | 95% CI (pp) | P(delta>0) | McNemar b/c | McNemar p |")
        lines.append("|---|---|---|---|---|---|")
        for name, c in res["headline_comparisons"].items():
            pb = c["paired_bootstrap"]; mc = c["mcnemar"]
            p = mc["p"]
            pstr = "<1e-12" if p < 1e-12 else f"{p:.2e}"
            lines.append(
                f"| {name} | {pb['delta_pp']:+.1f} | [{pb['low_pp']:+.1f}, {pb['high_pp']:+.1f}] | "
                f"{pb['frac_gt0']:.3f} | {mc['b']}/{mc['c']} | {pstr} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-dir", default=os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES"))
    ap.add_argument("--splits", nargs="+", default=["cr_test", "ir_test"])
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    baselines_dir = Path(args.baselines_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in args.splits:
        print(f"[reanalysis] {split} ...", file=sys.stderr)
        results[split] = analyze_split(baselines_dir, split)

    (out_dir / "reanalysis.json").write_text(json.dumps(results, indent=2))
    (out_dir / "reanalysis.md").write_text(to_markdown(results))
    print(f"[reanalysis] wrote {out_dir/'reanalysis.json'} and {out_dir/'reanalysis.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
