#!/usr/bin/env python3
"""Per-commit decay evaluation for Code2LoRA-direct (static hypernetwork).

Mirrors the schema of ``hypernetwork/eval_code2lora_gru_commits_metrics.py``
so the same plotters (``analysis/plot_per_commit_decay_all_suites.py``) can
overlay static and GRU curves on the same figure.

For each suite JSON (one item per (repo, commit), shape produced by
``create_dataset/build_static_commit_all_splits.py``)::

    for repo_commit_key in suite["repositories"]:
        emb = entry["embedding"]                # 2048-d
        qna_pairs = entry["qna_pairs"]          # already v2-extracted at this commit
        lora_params = hypernetwork(emb)
        score(qna_pairs, lora_params)
        # accumulate EM / EditSim / CodeBLEU

Output JSON shape (per suite)::

    {
      "final": {em_pct, edit_similarity, code_bleu, n, ci_low, ci_high},
      "per_repo": {
        repo_id: {
            "n_assertions_total": int,
            "per_commit_timeline": [
                {"commit_index": int,
                 "commit_sha": str,
                 "n_commits_after_anchor": int,
                 "n_files_changed_since_anchor": int,
                 "n": int,
                 "exact_match": float,    (0..1)
                 "edit_similarity": float,
                 "code_bleu": float},
                ...
            ],
        }, ...
      },
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "hypernetwork"))

# Reuse the same scoring helpers as the GRU eval driver.
from hypernetwork.eval_code2lora_gru_commits_metrics import (  # noqa: E402
    apply_lora_hooks, remove_lora_hooks,
    _score_assertions, _acc_init, _acc_merge, _acc_finalize,
    get_bos_id,
)
from evaluation.run_repopeft_bench import build_direct_method  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def _bootstrap_ci(values: List[float], weights: List[int],
                  n_boot: int, alpha: float = 0.05) -> Dict[str, float]:
    if not values or sum(weights) == 0:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(1234)
    n = len(values)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        w = sum(weights[i] for i in idx)
        if w == 0:
            continue
        m = sum(values[i] * weights[i] for i in idx) / w
        means.append(m)
    means.sort()
    return {
        "low": means[int(len(means) * (alpha / 2))],
        "high": means[int(len(means) * (1 - alpha / 2))],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to hypernetwork_best.pt for the static model.")
    ap.add_argument("--splits-dir", required=True,
                    help="Directory containing <suite>.json files.")
    ap.add_argument("--suites", nargs="+",
                    default=["ir_val", "ir_test", "cr_val", "cr_test", "ood_test"])
    ap.add_argument("--base-model",
                    default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--target-module-names", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-assertions-per-commit", type=int, default=64,
                    help="Sub-sample QnAs per (repo,commit) to bound runtime.")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-entries", type=int, default=0)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Splits dir  : {args.splits_dir}")
    print(f"Suites      : {args.suites}")
    print(f"Output      : {args.output_json}")

    rng = random.Random(args.seed)

    print("Loading base model + tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()
    bos_id = get_bos_id(tok)

    print("Loading static hypernetwork ...", flush=True)
    install_fn, tmd, scaling = build_direct_method(
        checkpoint=Path(args.checkpoint),
        base_model=base_model,
        target_module_names=args.target_module_names,
        device=device,
    )

    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "splits_dir": str(args.splits_dir),
        "suites": list(args.suites),
        "base_model": args.base_model,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_assertions_per_commit": args.max_assertions_per_commit,
        "bootstrap": args.bootstrap,
    }

    splits_dir = Path(args.splits_dir)
    for suite in args.suites:
        path = splits_dir / f"{suite}.json"
        if not path.exists():
            print(f"  [{suite}] missing {path}, skipping")
            results[suite] = None
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("repositories") or {}
        keys = sorted(entries.keys())
        if args.limit_entries > 0:
            keys = keys[: args.limit_entries]
        print(f"\n=== Suite: {suite} ({len(keys)} entries) ===", flush=True)

        suite_acc = _acc_init()
        per_repo: Dict[str, Any] = defaultdict(lambda: {
            "n_assertions_total": 0,
            "per_commit_timeline": [],
        })

        for i, key in enumerate(keys, 1):
            entry = entries[key]
            md = entry.get("metadata") or {}
            repo_id = md.get("repo_id") or key.split("@@")[0]
            sha = md.get("commit_sha") or (key.split("@@")[1] if "@@" in key else "")
            ci = int(md.get("commit_index", 0))

            qna_pairs = entry.get("qna_pairs") or []
            if not qna_pairs:
                continue

            # Build assertion tuples (prefix, target) and optionally subsample.
            pairs = [(q.get("prefix") or "", q.get("target") or "")
                     for q in qna_pairs if q.get("prefix") and q.get("target")]
            if not pairs:
                continue
            if args.max_assertions_per_commit > 0 and len(pairs) > args.max_assertions_per_commit:
                pairs = rng.sample(pairs, args.max_assertions_per_commit)

            # Generate LoRA for this snapshot, install, score, uninstall.
            lora_params = install_fn(entry)
            if lora_params is None:
                continue
            handles = apply_lora_hooks(tmd, lora_params, scaling)
            acc = _score_assertions(
                pairs, base_model, tok, device, bos_id,
                args.max_input_tokens, args.max_new_tokens,
            )
            remove_lora_hooks(handles)

            row = {
                "commit_index": ci,
                "commit_sha": sha,
                "n_commits_after_anchor": int(md.get("n_commits_after_anchor", 0)),
                "n_files_changed_since_anchor":
                    int(md.get("n_files_changed_since_anchor", 0)),
                **_acc_finalize(acc),
            }
            per_repo[repo_id]["per_commit_timeline"].append(row)
            per_repo[repo_id]["n_assertions_total"] += int(row["n"])
            _acc_merge(suite_acc, acc)

            if i % 50 == 0 or i == len(keys):
                fin = _acc_finalize(suite_acc)
                print(f"  [{suite}] [{i}/{len(keys)}] "
                      f"EM={fin['em_pct']:.2f}% n={fin['n']:,}", flush=True)

        fin = _acc_finalize(suite_acc)
        # Bootstrap CI over per-(repo,commit) means.
        rows_flat = [r for v in per_repo.values()
                     for r in v["per_commit_timeline"]]
        if rows_flat:
            em_vals = [r["em_pct"] for r in rows_flat]
            ns = [r["n"] for r in rows_flat]
            ci = _bootstrap_ci(em_vals, ns, n_boot=args.bootstrap)
        else:
            ci = {"low": 0.0, "high": 0.0}
        suite_obj = {
            "final": {**fin, "ci_low": ci["low"], "ci_high": ci["high"]},
            "per_repo": {k: dict(v) for k, v in per_repo.items()},
            "n_repos_scored": len(per_repo),
            "n_entries_scored": sum(len(v["per_commit_timeline"])
                                    for v in per_repo.values()),
        }
        results[suite] = suite_obj

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    print("\n=== HEADLINE ===")
    for suite in args.suites:
        s = results.get(suite)
        if not isinstance(s, dict):
            continue
        f = s.get("final") or {}
        print(f"  {suite:<12}  EM={f.get('em_pct', 0):>6.2f}%  "
              f"[{f.get('ci_low', 0):>5.2f}, {f.get('ci_high', 0):>5.2f}]  "
              f"n={f.get('n', 0):,}  repos={s.get('n_repos_scored', 0)}")


if __name__ == "__main__":
    main()
