#!/usr/bin/env python3
"""
Train and evaluate per-repo LoRA adapters for all repos in the training split.
Aggregates results across all repos for the IR (in-repo) evaluation.

Usage:
    python baselines/lora_per_repo/run_all_repos.py
    python baselines/lora_per_repo/run_all_repos.py --limit-repos 10  # first 10 repos
    python baselines/lora_per_repo/run_all_repos.py --eval-only        # skip training, just evaluate
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    default_output = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "TRAINING_CHECKPOINTS", "PER_REPO_LORA",
    )

    ap = argparse.ArgumentParser(description="Train+eval per-repo LoRA for all repos")
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--output-base", type=str, default=default_output)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    ap.add_argument("--eval-split", type=str, default="ir_test")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Pass through to train_lora.py")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Pass through to train_lora.py")
    ap.add_argument("--oracle-cache-dir", type=str, default=None,
                    help="Pass through to train_lora.py")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()

    train_path = splits_dir / "train.json"
    if not train_path.exists():
        raise FileNotFoundError(f"train.json not found at {splits_dir}")
    data = json.loads(train_path.read_text(encoding="utf-8"))
    repo_names = sorted(data.get("repositories", {}).keys())
    if args.limit_repos:
        repo_names = repo_names[:args.limit_repos]

    print(f"Processing {len(repo_names)} repos")

    script_dir = Path(__file__).resolve().parent
    train_script = str(script_dir / "train_lora.py")
    test_script = str(script_dir / "test_lora.py")

    all_results = {}
    total_em = 0
    total_n = 0

    for i, repo in enumerate(repo_names):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(repo_names)}] {repo}")
        print("=" * 60)

        adapter_path = Path(args.output_base) / repo / "adapter"

        # Train
        if not args.eval_only:
            if adapter_path.exists():
                print(f"  Adapter already exists, skipping training")
            else:
                cmd = [
                    sys.executable, train_script,
                    "--from-split", "train",
                    "--splits-dir", str(splits_dir),
                    "--repo-name", repo,
                    "--epochs", str(args.epochs),
                    "--output-dir", str(Path(args.output_base) / repo),
                ]
                if args.no_wandb:
                    cmd.append("--no-wandb")
                if args.use_oracle:
                    cmd.append("--use-oracle")
                if args.oracle_cache_dir:
                    cmd.extend(["--oracle-cache-dir", args.oracle_cache_dir])
                if args.max_seq_length:
                    cmd.extend(["--max-seq-length", str(args.max_seq_length)])
                print(f"  Training LoRA for {repo}...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  TRAINING FAILED: {result.stderr[-500:]}")
                    continue

        # Evaluate
        if not adapter_path.exists():
            print(f"  No adapter found at {adapter_path}, skipping eval")
            continue

        cmd = [
            sys.executable, test_script,
            "--adapter", str(adapter_path),
            "--splits-dir", str(splits_dir),
            "--split", args.eval_split,
            "--repo", repo,
        ]

        print(f"  Evaluating on {args.eval_split}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  EVAL FAILED: {result.stderr[-500:]}")
            continue

        # Read results -- test_lora.py saves to {adapter_parent}_results/{split}/
        repo_dir = adapter_path.parent
        results_path = Path(str(repo_dir) + "_results") / args.eval_split / "lora_results.json"
        if not results_path.exists():
            for rp in [
                repo_dir / "adapter_results" / args.eval_split / "lora_results.json",
                Path(str(adapter_path) + "_results") / args.eval_split / "lora_results.json",
            ]:
                if rp.exists():
                    results_path = rp
                    break

        if results_path.exists():
            repo_results = json.loads(results_path.read_text(encoding="utf-8"))
            em_pct = repo_results.get("exact_match_pct", 0)
            n = repo_results.get("n", 0)
            em_count = repo_results.get("exact_match_count", 0)
            edit_sim = repo_results.get("edit_similarity", 0)
            bleu = repo_results.get("code_bleu", 0)
            print(f"  EM: {em_pct:.2f}% ({em_count}/{n})  EditSim: {edit_sim:.4f}  BLEU: {bleu:.4f}")
            all_results[repo] = repo_results
            total_em += em_count
            total_n += n
        else:
            print(f"  Results not found at {results_path}")
            if result.stderr:
                print(f"  Eval stderr (last 300): {result.stderr[-300:]}")
            if result.stdout:
                print(f"  Eval stdout (last 300): {result.stdout[-300:]}")

    # Aggregate
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({args.eval_split})")
    print("=" * 60)
    if total_n > 0:
        print(f"  Repos evaluated: {len(all_results)}")
        print(f"  Total examples: {total_n}")
        print(f"  Overall Exact Match: {100.0 * total_em / total_n:.2f}% ({total_em}/{total_n})")
        total_edit = sum(r.get("edit_similarity", 0) * r.get("n", 0) for r in all_results.values())
        print(f"  Overall Edit Similarity: {total_edit / total_n:.4f}")
    else:
        print("  No results collected")

    # Save aggregate
    agg_path = Path(args.output_base) / f"aggregate_{args.eval_split}.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg = {
        "method": "lora_per_repo",
        "split": args.eval_split,
        "exact_match_pct": 100.0 * total_em / total_n if total_n > 0 else 0,
        "exact_match_count": total_em,
        "n": total_n,
        "n_repos": len(all_results),
        "per_repo": {r: {"exact_match_pct": v.get("exact_match_pct", 0), "n": v.get("n", 0)}
                     for r, v in all_results.items()},
    }
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nAggregate results saved to {agg_path}")


if __name__ == "__main__":
    main()
