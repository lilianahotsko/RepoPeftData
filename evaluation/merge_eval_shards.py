#!/usr/bin/env python3
"""Aggregate sharded eval JSONs (from ``run_baselines_v2.py``,
``run_code2lora_gru_v2_eval.py``, or ``eval_code2lora_gru_commits_metrics.py``)
into a single per-suite summary JSON with correct bootstrap CIs over the
**union** of the shards' raw samples.

Two shard formats are supported:

* **Baselines / v2 GRU**::

    {
        "finalized": bool, "shard_i": int, "num_shards": int,
        "summary": {...},
        "per_commit": [{repo_id, commit_sha, commit_index, n_qnas,
                        exact_match, edit_similarity, code_bleu}, ...],
        "raw_samples": {"exact_match": [...], "edit_similarity": [...],
                        "code_bleu": [...]}
    }

* **Legacy GRU**::

    {"<suite>": {
        "finalized": bool, "shard_i": int, "num_shards": int,
        "final": {n, exact_match, edit_similarity, code_bleu},
        "timeline": [{commit_index, n, exact_match, ...}],
        "per_repo": {repo_id: {final, timeline, sanity}},
        "sanity": {...}
    }}

Usage::

    # Merge baselines / v2 GRU shards (one suite at a time):
    python evaluation/merge_eval_shards.py \\
        --pattern '/path/to/baseline_pretrained_cr_val_shard*of*.json' \\
        --output  /path/to/baseline_pretrained_cr_val.json

    # Or auto-discover & merge every suite in a directory:
    python evaluation/merge_eval_shards.py \\
        --input-dir /path/to/run_dir --auto-detect
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import aggregate_metrics_with_ci  # noqa: E402


def _bootstrap_summary(em: List[float], ed: List[float], cb: List[float],
                       n_resamples: int = 5000) -> Dict[str, Any]:
    n = len(em)
    if n == 0:
        return {"n_qnas": 0, "exact_match": 0.0,
                "edit_similarity": 0.0, "code_bleu": 0.0}
    metric_dicts = [
        {"exact_match": bool(em[i]), "edit_similarity": float(ed[i]),
         "code_bleu": float(cb[i])}
        for i in range(n)
    ]
    agg = aggregate_metrics_with_ci(metric_dicts, n_resamples=int(n_resamples))
    out: Dict[str, Any] = {"n_qnas": n}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            out[k] = float(v["mean"])
            out[f"{k}_ci"] = [float(v.get("low", 0.0)),
                              float(v.get("high", 0.0))]
            out[f"{k}_n"] = int(v.get("n", 0))
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Format A: baselines / v2 GRU
# ---------------------------------------------------------------------------

def merge_baselines_like(shard_paths: List[Path], output: Path,
                         bootstrap: int = 5000) -> Dict[str, Any]:
    if not shard_paths:
        raise SystemExit("merge_baselines_like: no shard paths given")

    per_commit_all: List[Dict[str, Any]] = []
    em: List[float] = []
    ed: List[float] = []
    cb: List[float] = []
    suite_name: Optional[str] = None
    n_unfinalized = 0
    n_total_groups = 0

    for p in shard_paths:
        obj = json.loads(p.read_text())
        if not obj.get("finalized"):
            n_unfinalized += 1
        s = obj.get("summary", {})
        if s.get("suite") and suite_name is None:
            suite_name = s["suite"]
        per_commit_all.extend(obj.get("per_commit", []))
        rs = obj.get("raw_samples") or {}
        em.extend(rs.get("exact_match", []))
        ed.extend(rs.get("edit_similarity", []))
        cb.extend(rs.get("code_bleu", []))
        n_total_groups = max(n_total_groups, int(obj.get("n_total_groups", 0)))

    if n_unfinalized:
        print(f"  [warn] {n_unfinalized} of {len(shard_paths)} shards are "
              f"not finalized -- merging anyway with whatever raw_samples / "
              f"per_commit they had at last write", flush=True)

    summary = _bootstrap_summary(em, ed, cb, n_resamples=bootstrap)
    summary["suite"] = suite_name or "unknown"
    summary["n_scored_commits"] = len(per_commit_all)
    summary["n_repos"] = len({r["repo_id"] for r in per_commit_all})
    summary["n_shards"] = len(shard_paths)

    payload = {
        "finalized": True,
        "merged_from": [str(p.name) for p in shard_paths],
        "summary": summary,
        "per_commit": per_commit_all,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"  [{summary['suite']}] -> {output}  "
          f"n_qnas={summary['n_qnas']:,}  EM={summary['exact_match']:.4f}  "
          f"commits={summary['n_scored_commits']:,}  "
          f"repos={summary['n_repos']}  shards={len(shard_paths)}",
          flush=True)
    return payload


# ---------------------------------------------------------------------------
# Format B: legacy GRU
# ---------------------------------------------------------------------------

def merge_legacy_gru(shard_paths: List[Path], output: Path,
                     suite_name: str, bootstrap: int = 5000) -> Dict[str, Any]:
    """Combine per-shard legacy GRU JSONs (one suite per file) into a unified
    one. Recomputes the suite-level FINAL by summing per-repo finals from all
    shards, and the per-commit-index timeline by summing weighted per-repo
    timelines."""
    if not shard_paths:
        raise SystemExit("merge_legacy_gru: no shard paths given")

    per_repo: Dict[str, Any] = {}
    final_n = 0
    final_em = 0.0
    final_ed = 0.0
    final_cb = 0.0
    timeline_acc: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "em": 0.0, "ed": 0.0, "cb": 0.0}
    )

    for p in shard_paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        suite = obj.get(suite_name) or next(iter(obj.values()), {})
        if not suite:
            continue
        per_repo.update(suite.get("per_repo", {}) or {})
        fin = suite.get("final") or {}
        n = int(fin.get("n", 0))
        final_n += n
        final_em += float(fin.get("exact_match", 0.0)) * n
        final_ed += float(fin.get("edit_similarity", 0.0)) * n
        final_cb += float(fin.get("code_bleu", 0.0)) * n
        for row in (suite.get("timeline") or []):
            ci = int(row["commit_index"])
            rn = int(row.get("n", 0))
            acc = timeline_acc[ci]
            acc["n"] += rn
            acc["em"] += float(row.get("exact_match", 0.0)) * rn
            acc["ed"] += float(row.get("edit_similarity", 0.0)) * rn
            acc["cb"] += float(row.get("code_bleu", 0.0)) * rn

    def _finalize(n: int, em: float, ed: float, cb: float) -> Dict[str, float]:
        d = max(1, n)
        return {"n": n, "exact_match": em / d, "edit_similarity": ed / d,
                "code_bleu": cb / d}

    suite_out: Dict[str, Any] = {
        "finalized": True,
        "merged_from": [str(p.name) for p in shard_paths],
        "final": _finalize(final_n, final_em, final_ed, final_cb),
        "timeline": [
            {"commit_index": int(ci),
             **_finalize(acc["n"], acc["em"], acc["ed"], acc["cb"])}
            for ci, acc in sorted(timeline_acc.items(), key=lambda x: x[0])
        ],
        "per_repo": per_repo,
        "n_repos": len(per_repo),
        "n_shards": len(shard_paths),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({suite_name: suite_out}, indent=2),
                      encoding="utf-8")
    print(f"  [{suite_name}] -> {output}  "
          f"n={suite_out['final']['n']:,}  "
          f"EM={suite_out['final']['exact_match']:.4f}  "
          f"repos={suite_out['n_repos']}  shards={len(shard_paths)}",
          flush=True)
    return suite_out


# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------

SHARD_RE = re.compile(r"_shard(\d+)of(\d+)\.json$")


def _group_shards(input_dir: Path) -> Dict[str, List[Path]]:
    """Return {prefix_without_shard_suffix: [shard_paths sorted by shard_i]}.

    For example:
      baseline_pretrained_cr_val_shard0of4.json
      baseline_pretrained_cr_val_shard1of4.json
      baseline_pretrained_cr_val_shard2of4.json
      baseline_pretrained_cr_val_shard3of4.json
        -> {"baseline_pretrained_cr_val": [shard0..shard3]}
    """
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in sorted(input_dir.glob("*.json")):
        m = SHARD_RE.search(p.name)
        if not m:
            continue
        prefix = p.name[: m.start()]
        groups[prefix].append((int(m.group(1)), p))
    return {k: [p for _, p in sorted(v)] for k, v in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", type=str, default=None,
                    help="Glob pattern matching the shards to merge "
                         "(used when --auto-detect is off).")
    ap.add_argument("--output", type=str, default=None,
                    help="Where to write the merged JSON (used with --pattern).")
    ap.add_argument("--input-dir", type=str, default=None)
    ap.add_argument("--auto-detect", action="store_true",
                    help="Walk --input-dir and merge all shard groups it finds.")
    ap.add_argument("--legacy-gru", action="store_true",
                    help="Treat shards as legacy-GRU-eval format (single-suite "
                         "wrapped JSON with per_repo dict + timeline rows).")
    ap.add_argument("--suite-name", type=str, default=None,
                    help="Required with --legacy-gru when not --auto-detect.")
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()

    if args.auto_detect:
        if not args.input_dir:
            raise SystemExit("--auto-detect requires --input-dir")
        groups = _group_shards(Path(args.input_dir))
        if not groups:
            print(f"no shard groups found in {args.input_dir}")
            return
        for prefix, paths in groups.items():
            out_path = Path(args.input_dir) / f"{prefix}.json"
            if args.legacy_gru:
                # Try to extract suite from prefix: e.g.
                # 'eval_results__cross_repo_cr_val' -> 'cross_repo_cr_val'
                suite = prefix.split("__")[-1] if "__" in prefix else prefix
                merge_legacy_gru(paths, out_path, suite_name=suite,
                                 bootstrap=args.bootstrap)
            else:
                merge_baselines_like(paths, out_path, bootstrap=args.bootstrap)
        return

    if not args.pattern or not args.output:
        raise SystemExit("either --auto-detect (with --input-dir) or "
                         "--pattern + --output are required")

    paths = sorted(Path(p) for p in glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no shards matched {args.pattern}")
    out = Path(args.output)
    if args.legacy_gru:
        suite = args.suite_name or out.stem
        merge_legacy_gru(paths, out, suite_name=suite, bootstrap=args.bootstrap)
    else:
        merge_baselines_like(paths, out, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
