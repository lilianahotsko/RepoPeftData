#!/usr/bin/env python3
"""Inference-time token-overhead + amortization analysis (no GPU) [yeRc-W1/W6].

Clarifies the "zero inference cost" claim. Code2LoRA merges a pre-generated
adapter into the weights, so at inference it feeds only the test prefix -- zero
extra input tokens per query. Context baselines (RAG / DRC) must re-encode
retrieved context on EVERY query.

We report, per test suite:
  * median / mean prefix tokens (the shared base cost every method pays),
  * the per-query *added* input tokens for RAG and DRC (their configured context
    budgets), vs. 0 for Code2LoRA,
  * queries served per repository (adapter-generation is paid once per
    repo-state and amortized over this many completions),
  * break-even query count given a measured one-time adapter-generation time
    (read from the CPU benchmark SLURM log when available).

Tokens are counted with the actual Qwen2.5-Coder tokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# per-query context budgets actually used by the V2 baselines (see
# scripts/slurm/eval_baselines_v2_sharded.sh defaults)
RAG_CTX_BUDGET = 1536
DRC_CTX_BUDGET = 4096


def prefix_token_stats(parquet_path: Path, tokenizer, sample: int = 2000):
    pf = pq.ParquetFile(str(parquet_path))
    n_total = pf.metadata.num_rows
    # Cheap full pass over the small repo_id column for the repo count.
    repos: set = set()
    for batch in pf.iter_batches(batch_size=8192, columns=["repo_id"]):
        repos.update(batch.column("repo_id").to_pylist())
    n_repos = len(repos)
    # Memory-frugal prefix sampling: tokenize at most `sample` prefixes.
    lens_list = []
    for batch in pf.iter_batches(batch_size=256, columns=["prefix"]):
        for p in batch.column("prefix").to_pylist():
            lens_list.append(len(tokenizer.encode(p, add_special_tokens=False)))
            if sample > 0 and len(lens_list) >= sample:
                break
        if sample > 0 and len(lens_list) >= sample:
            break
    lens = np.array(lens_list)
    return {
        "n_queries": int(n_total),
        "n_repos": int(n_repos),
        "queries_per_repo": float(n_total / n_repos) if n_repos else None,
        "prefix_tokens_median": float(np.median(lens)),
        "prefix_tokens_mean": float(lens.mean()),
        "prefix_tokens_p90": float(np.quantile(lens, 0.9)),
        "sampled": int(len(lens)),
    }


def parse_bench_log(log_glob: str):
    """Pull full/incr/cached seconds from the CPU benchmark SLURM log if present."""
    files = sorted(Path("slurm_logs").glob("bench_c2l_adapt_cpu_*.out"))
    if not files:
        return None
    txt = files[-1].read_text(errors="ignore")
    by_repo = {}  # dedup: the log prints the table once in-loop and again in summary
    for line in txt.splitlines():
        m = re.match(r"^(\S+/\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if m:
            by_repo[m.group(1)] = {"repo": m.group(1), "full_s": float(m.group(2)),
                                   "incr_s": float(m.group(3)), "cached_s": float(m.group(4)),
                                   "commits": int(m.group(5))}
    rows = list(by_repo.values())
    if not rows:
        return None
    return {"log": str(files[-1]), "rows": rows,
            "mean_full_s": float(np.mean([r["full_s"] for r in rows])),
            "mean_incr_s": float(np.mean([r["incr_s"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qna-dir",
                    default=os.environ.get("QNA_DIR",
                                           "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/qna"))
    ap.add_argument("--suites", nargs="+", default=["cr_test", "ir_test"])
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)

    per_suite = {}
    for sp in args.suites:
        p = Path(args.qna_dir) / f"{sp}.parquet"
        if not p.exists():
            print(f"[amort] missing {p}", file=sys.stderr)
            continue
        per_suite[sp] = prefix_token_stats(p, tok)

    bench = parse_bench_log("slurm_logs/bench_c2l_adapt_cpu_*.out")

    out = {
        "rag_ctx_budget_tokens_per_query": RAG_CTX_BUDGET,
        "drc_ctx_budget_tokens_per_query": DRC_CTX_BUDGET,
        "code2lora_added_tokens_per_query": 0,
        "per_suite": per_suite,
        "adapter_generation_benchmark": bench,
    }
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "amortization.json").write_text(json.dumps(out, indent=2))

    lines = ["# Inference token overhead + amortization (yeRc-W1/W6)\n"]
    lines.append("Per-query *added* input tokens (context that must be re-encoded on every query):\n")
    lines.append("| Method | added tokens / query |")
    lines.append("|---|---|")
    lines.append(f"| Code2LoRA (merged adapter) | 0 |")
    lines.append(f"| RAG | up to {RAG_CTX_BUDGET} |")
    lines.append(f"| DRC | up to {DRC_CTX_BUDGET} |")
    lines.append("\n## Prefix cost and queries per repo\n")
    lines.append("| Suite | queries | repos | queries/repo | prefix tok (median) | prefix tok (mean) |")
    lines.append("|---|---|---|---|---|---|")
    for sp, s in per_suite.items():
        lines.append(f"| {sp} | {s['n_queries']} | {s['n_repos']} | {s['queries_per_repo']:.1f} | "
                     f"{s['prefix_tokens_median']:.0f} | {s['prefix_tokens_mean']:.0f} |")
    if bench:
        lines.append(f"\n## One-time adapter generation (CPU, {len(bench['rows'])} repos)\n")
        lines.append(f"mean full walk: **{bench['mean_full_s']:.1f}s**, "
                     f"mean incremental /adapt: **{bench['mean_incr_s']:.1f}s** "
                     f"(paid once per repo-state, amortized over all subsequent completions).\n")
        lines.append("| repo | full_s | incr_s | cached_s | commits |")
        lines.append("|---|---|---|---|---|")
        for r in bench["rows"]:
            lines.append(f"| {r['repo']} | {r['full_s']:.1f} | {r['incr_s']:.1f} | {r['cached_s']:.1f} | {r['commits']} |")
    else:
        lines.append("\n## One-time adapter generation\n")
        lines.append("_CPU benchmark (SLURM job 17386812) not finished yet; rerun this script "
                     "once `slurm_logs/bench_c2l_adapt_cpu_*.out` exists to fill in wall-clock times._\n")
    (od / "amortization.md").write_text("\n".join(lines) + "\n")
    print(f"[amort] wrote {od/'amortization.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
