#!/usr/bin/env python3
"""Aggregate per-repo LoRA eval JSONs into a single paper-style summary.

Each per-repo LoRA adapter writes its eval payload at
``<repo_dir>_results/<split>/lora_results.json`` (see
``baselines/lora_per_repo/test_lora.py``). This script walks the entire
output base directory, concatenates the per-entry metric records, and
produces a single summary JSON with EM / EditSim / CodeBLEU means and 95%
bootstrap CIs (matching the schema of the merged shard outputs under
``evaluation/merge_eval_shards.py``).

Usage::

    python baselines/lora_per_repo/aggregate_results.py \
        --base $CKPT_DIR/PER_REPO_LORA_GRU \
        --split ir_test \
        --output $BASELINES_DIR/per_repo_lora_gru_ir_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import aggregate_metrics_with_ci


def _walk_results(base: Path, split: str) -> list[Path]:
    """Return all <author>/<repo>_results/<split>/lora_results.json files."""
    results: list[Path] = []
    if not base.exists():
        return results
    for author_dir in sorted(base.iterdir()):
        if not author_dir.is_dir():
            continue
        for repo_results in sorted(author_dir.glob("*_results")):
            cand = repo_results / split / "lora_results.json"
            if cand.is_file():
                results.append(cand)
    return results


def _format_summary(summary: dict) -> str:
    def fmt(key: str, scale: float = 1.0, digits: int = 4) -> str:
        if key not in summary:
            return f"{key}=?"
        mean = summary[key]
        ci = summary.get(f"{key}_ci")
        if ci:
            return (f"{key}={mean * scale:.{digits}f} "
                    f"({ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f})")
        return f"{key}={mean * scale:.{digits}f}"

    return "  ".join([
        f"n={summary['n_qnas']:,}",
        f"repos={summary.get('n_repos', '?')}",
        fmt("exact_match", scale=100.0, digits=2) + "%",
        fmt("edit_similarity"),
        fmt("code_bleu"),
    ])


def main() -> None:
    default_base = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "TRAINING_CHECKPOINTS",
        "PER_REPO_LORA",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=str, default=default_base,
                    help="Output base dir written by run_all_repos.py "
                         "(e.g. $CKPT_DIR/PER_REPO_LORA_GRU)")
    ap.add_argument("--split", type=str, default="ir_test",
                    help="Eval split to aggregate (default: ir_test)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path. Defaults to "
                         "<base>/aggregate_<split>_summary.json")
    ap.add_argument("--bootstrap", type=int, default=5000,
                    help="Number of bootstrap resamples (default: 5000)")
    ap.add_argument("--label", type=str, default="per_repo_lora",
                    help="Method label to embed in the output JSON")
    ap.add_argument("--suite", type=str, default=None,
                    help="Suite name override (default: <split>)")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    output = Path(args.output) if args.output else base / f"aggregate_{args.split}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = _walk_results(base, args.split)
    if not paths:
        raise SystemExit(f"No lora_results.json found under {base}/*/*_results/{args.split}/")

    print(f"Found {len(paths)} per-repo result files under {base}")

    records: list[dict] = []
    per_repo: dict[str, dict] = {}
    for p in paths:
        # p = <base>/<author>/<repo>_results/<split>/lora_results.json
        author = p.parents[2].name
        repo_name = p.parents[1].name.removesuffix("_results")
        repo_id = f"{author}/{repo_name}"
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  [skip] {p}: corrupt JSON ({e})")
            continue
        entries = payload.get("entries") or []
        if not entries:
            continue
        em_count = 0
        ed_sum = 0.0
        cb_sum = 0.0
        for ent in entries:
            em = bool(ent.get("exact_match"))
            ed = float(ent.get("edit_similarity", 0.0))
            cb = float(ent.get("code_bleu", 0.0))
            records.append({"exact_match": em,
                            "edit_similarity": ed,
                            "code_bleu": cb})
            if em:
                em_count += 1
            ed_sum += ed
            cb_sum += cb
        n = len(entries)
        per_repo[repo_id] = {
            "n": n,
            "exact_match_pct": 100.0 * em_count / n,
            "exact_match_count": em_count,
            "edit_similarity": ed_sum / n,
            "code_bleu": cb_sum / n,
        }

    if not records:
        raise SystemExit(f"All result files under {base} were empty.")

    agg = aggregate_metrics_with_ci(records, n_resamples=int(args.bootstrap))
    summary: dict = {"n_qnas": len(records),
                     "n_repos": len(per_repo),
                     "n_result_files": len(paths),
                     "suite": args.suite or args.split,
                     "method": args.label}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            summary[k] = float(v["mean"])
            summary[f"{k}_ci"] = [float(v.get("low", 0.0)),
                                  float(v.get("high", 0.0))]
            summary[f"{k}_n"] = int(v.get("n", 0))
        else:
            summary[k] = v

    payload = {
        "finalized": True,
        "merged_from": [str(p.relative_to(base)) for p in paths],
        "summary": summary,
        "per_repo": per_repo,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"-> {output}")
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
