#!/usr/bin/env python3
"""EM vs. repository size ablation (no GPU)  [covers yeRc-W3].

Tests the concern that a single fixed-size mean+max-pooled embedding cannot
scale to large repositories: we bin CR/IR-test instances by the total source
size of their repository and report each method's EM per bin. If Code2LoRA held
up only on small repos, its advantage would shrink in the largest bins.

Repository size = total characters of non-test ``*.py`` files under
``$REPOS_ROOT/<repo_id>`` (test files excluded by path). Reported in characters
and approximate tokens (chars / 4).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.rebuttal.reanalysis import METHODS, _as_bool, load_method  # noqa: E402

TEST_HINTS = ("test", "test_hypernet")
PRUNE_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
              ".mypy_cache", ".pytest_cache", ".tox", "build", "dist", ".eggs"}


def repo_source_chars(repos_root: Path, repo_id: str) -> int:
    d = repos_root / repo_id
    if not d.is_dir():
        return -1
    total = 0
    for root, dirnames, filenames in os.walk(d):
        dirnames[:] = [dn for dn in dirnames if dn not in PRUNE_DIRS]
        low_root = root.lower()
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            low = low_root + "/" + fn.lower()
            if any(h in low for h in TEST_HINTS):
                continue
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def analyze(baselines_dir: Path, repos_root: Path, split: str):
    labels = ["Pretrained", "DRC", "FFT", "sLoRA", "Code2LoRA-Static"]
    data = {}
    for lbl in labels:
        rows = load_method(baselines_dir, METHODS[lbl], split)
        if rows is not None:
            data[lbl] = rows
    ref = data["Pretrained"]
    n = len(ref)
    # alignment guard
    for lbl, rows in list(data.items()):
        if len(rows) != n or [(r["repo"], r["expected"]) for r in rows] != [(r["repo"], r["expected"]) for r in ref]:
            print(f"  [skip] {lbl}: misaligned", file=sys.stderr)
            data.pop(lbl)

    repos = [r["repo"] for r in ref]
    uniq = sorted(set(repos))
    size_chars = {rp: repo_source_chars(repos_root, rp) for rp in uniq}
    n_missing = sum(1 for v in size_chars.values() if v < 0)
    inst_size = np.array([size_chars[r] for r in repos], dtype=float)
    em = {lbl: np.array([r["em"] for r in rows]) for lbl, rows in data.items()}

    # size bins in approx tokens (chars/4)
    tok = inst_size / 4.0
    edges = [0, 25_000, 75_000, 150_000, 300_000, 1e12]
    edge_lbls = ["<25K", "25-75K", "75-150K", "150-300K", ">300K"]
    bins = []
    for (lo, hi), name in zip(zip(edges[:-1], edges[1:]), edge_lbls):
        m = (tok >= lo) & (tok < hi) & (inst_size >= 0)
        row = {"bin_tokens": name, "n_instances": int(m.sum()),
               "n_repos": len(set(np.array(repos)[m]))}
        for lbl in data:
            row[lbl] = float(em[lbl][m].mean() * 100) if m.any() else None
        bins.append(row)

    # per-repo EM vs size: correlation of repo-level EM with log size
    per_repo = {}
    for lbl in data:
        arr = em[lbl]
        rp_em, rp_sz = [], []
        for rp in uniq:
            if size_chars[rp] <= 0:
                continue
            mask = np.array([r == rp for r in repos])
            rp_em.append(arr[mask].mean())
            rp_sz.append(np.log10(size_chars[rp]))
        if len(rp_em) > 2:
            per_repo[lbl] = float(np.corrcoef(rp_sz, rp_em)[0, 1])
    return {
        "split": split,
        "n": n,
        "n_repos": len(uniq),
        "n_repos_missing_size": n_missing,
        "bins": bins,
        "corr_repoEM_vs_log_size": per_repo,
    }


DEFAULT_GRU_DIR = os.environ.get(
    "GRU_EVAL_DIR",
    "/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_best_sharded")


def analyze_evo(gru_dir: Path, repos_root: Path, split: str, file_prefix: str = "gru_v2"):
    """Code2LoRA-Evo (GRU) EM vs. repository size on the per-commit (evolution)
    track. GRU eval stores per-commit aggregates (repo_id, n_qnas, exact_match),
    so we bin commits by their repository's source size and report the
    QnA-weighted EM per bin — the evolution-track analogue of ``analyze``."""
    path = Path(gru_dir) / f"{file_prefix}_{split}.json"
    if not path.exists():
        print(f"  [evo skip] {path} missing", file=sys.stderr)
        return None
    doc = json.load(open(path))
    rows = doc.get("per_commit") or []
    if not rows:
        print(f"  [evo skip] {path} has no per_commit rows", file=sys.stderr)
        return None

    uniq = sorted({r["repo_id"] for r in rows})
    size_chars = {rp: repo_source_chars(repos_root, rp) for rp in uniq}
    n_missing = sum(1 for v in size_chars.values() if v < 0)
    tok = {rp: (size_chars[rp] / 4.0 if size_chars[rp] >= 0 else -1.0) for rp in uniq}

    edges = [0, 25_000, 75_000, 150_000, 300_000, 1e12]
    edge_lbls = ["<25K", "25-75K", "75-150K", "150-300K", ">300K"]
    bins = []
    for (lo, hi), name in zip(zip(edges[:-1], edges[1:]), edge_lbls):
        em_sum, n_sum, repos_in = 0.0, 0, set()
        for r in rows:
            rp = r["repo_id"]
            t = tok[rp]
            if t < 0 or not (lo <= t < hi):
                continue
            n = int(r.get("n_qnas", 0) or 0)
            em_sum += float(r.get("exact_match", 0.0)) * n
            n_sum += n
            repos_in.add(rp)
        bins.append({"bin_tokens": name, "n_instances": n_sum, "n_repos": len(repos_in),
                     "Code2LoRA-Evo": (100.0 * em_sum / n_sum if n_sum else None)})

    # per-repo QnA-weighted EM vs log10(size) correlation
    per_repo_em, per_repo_n = {}, {}
    for r in rows:
        rp = r["repo_id"]
        n = int(r.get("n_qnas", 0) or 0)
        per_repo_em[rp] = per_repo_em.get(rp, 0.0) + float(r.get("exact_match", 0.0)) * n
        per_repo_n[rp] = per_repo_n.get(rp, 0) + n
    xs, ys = [], []
    for rp in uniq:
        if size_chars[rp] <= 0 or per_repo_n.get(rp, 0) == 0:
            continue
        xs.append(np.log10(size_chars[rp]))
        ys.append(per_repo_em[rp] / per_repo_n[rp])
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None

    return {"split": split, "n": sum(per_repo_n.values()), "n_repos": len(uniq),
            "n_repos_missing_size": n_missing, "bins": bins,
            "corr_repoEM_vs_log_size": corr, "source": str(path)}


def evo_to_md(results: dict) -> str:
    lines = ["\n# Code2LoRA-Evo (GRU) — EM vs. repository size (evolution / per-commit track, yeRc-W3)\n",
             "\nSame fixed-size repository embedding, evaluated on the per-commit evolution track. "
             "EM is QnA-weighted across all commits whose repository falls in each size bin.\n"]
    for split, res in results.items():
        if res is None:
            continue
        lines.append(f"\n## {split}  (n={res['n']} QnAs, repos={res['n_repos']}, "
                     f"missing size={res['n_repos_missing_size']})\n")
        lines.append("| repo size (approx tokens) | n QnAs | n repos | Code2LoRA-Evo |")
        lines.append("|---|---|---|---|")
        for b in res["bins"]:
            v = b["Code2LoRA-Evo"]
            lines.append(f"| {b['bin_tokens']} | {b['n_instances']} | {b['n_repos']} | "
                         + (f"{v:.1f}" if v is not None else "-") + " |")
        c = res["corr_repoEM_vs_log_size"]
        if c is not None:
            lines.append(f"\nPer-repo EM vs. log10(repo size) Pearson r: **{c:+.3f}**\n")
    return "\n".join(lines) + "\n"


def to_md(results: dict) -> str:
    lines = ["# EM vs. repository size (yeRc-W3)\n"]
    for split, res in results.items():
        lines.append(f"\n## {split}  (n={res['n']}, repos={res['n_repos']}, missing size={res['n_repos_missing_size']})\n")
        methods = [k for k in res["bins"][0] if k not in ("bin_tokens", "n_instances", "n_repos")]
        lines.append("| repo size (approx tokens) | n inst | n repos | " + " | ".join(methods) + " |")
        lines.append("|---|---|---|" + "|".join(["---"] * len(methods)) + "|")
        for b in res["bins"]:
            cells = " | ".join(f"{b[m]:.1f}" if b[m] is not None else "-" for m in methods)
            lines.append(f"| {b['bin_tokens']} | {b['n_instances']} | {b['n_repos']} | {cells} |")
        lines.append("\n### Correlation of per-repo EM with log10(repo size)\n")
        lines.append("| Method | Pearson r |")
        lines.append("|---|---|")
        for m, r in res["corr_repoEM_vs_log_size"].items():
            lines.append(f"| {m} | {r:+.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-dir", default=os.environ.get("BASELINES_DIR", "/scratch/lhotsko/BASELINES"))
    ap.add_argument("--repos-root", default=os.environ.get("REPOS_ROOT", "/scratch/lhotsko/REPO_DATASET/repositories"))
    ap.add_argument("--splits", nargs="+", default=["cr_test", "ir_test"])
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--gru-eval-dir", default=DEFAULT_GRU_DIR,
                    help="dir with per-commit GRU eval JSONs (set '' to skip Evo)")
    ap.add_argument("--gru-file-prefix", default="gru_v2")
    args = ap.parse_args()
    out, evo = {}, {}
    for sp in args.splits:
        print(f"[repo_size] {sp} ...", file=sys.stderr)
        out[sp] = analyze(Path(args.baselines_dir), Path(args.repos_root), sp)
        if args.gru_eval_dir:
            print(f"[repo_size] {sp} (Evo/GRU) ...", file=sys.stderr)
            evo[sp] = analyze_evo(Path(args.gru_eval_dir), Path(args.repos_root), sp,
                                  args.gru_file_prefix)
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "repo_size_ablation.json").write_text(json.dumps({"static": out, "evo": evo}, indent=2))
    md = to_md(out)
    if any(v for v in evo.values()):
        md += evo_to_md(evo)
    (od / "repo_size_ablation.md").write_text(md)
    print(f"[repo_size] wrote {od/'repo_size_ablation.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
