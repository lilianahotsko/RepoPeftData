#!/usr/bin/env python3
"""Produce per-repo train/test splits for the GRU OOD suite (Table 3 per-repo LoRA).

Each OOD repo in ``gru_ood_test.json`` is a single bundle of qna_pairs across one
or more commits. To train a per-repo LoRA adapter and still evaluate it fairly
on held-out data from the *same* repo, we split each repo's qna_pairs by
``commit_idx`` (temporal split, default 80% earliest commits -> train,
20% latest -> test). For repos with only one unique commit_idx we fall back to
a deterministic random 80/20 split of the qna_pairs themselves so the per-repo
LoRA pipeline still has training data.

The output mirrors the structure of ``gru_train.json`` / ``gru_ir_test.json`` so
that the existing ``baselines/lora_per_repo`` pipeline can be reused unchanged:

    <out-dir>/
        train.json       # early-commit qna_pairs per repo (training set)
        ir_test.json     # late-commit qna_pairs per repo  (eval set)
        manifest.json    # per-repo split metadata (debug / audit)

Usage:
    python scripts/preprocess_gru_ood_temporal_split.py \
        --input  $SCRATCH/REPO_DATASET/gru_ood_test.json \
        --output $SCRATCH/REPO_DATASET_GRU_OOD \
        --train-frac 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def split_repo_by_commit(
    qna_pairs: list[dict],
    train_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict]:
    """Return (train_pairs, test_pairs, info).

    Splits pairs by ``commit_idx`` whenever the repo has >=2 unique commits;
    otherwise falls back to a deterministic random pair split using ``rng``.
    """
    commit_idxs = sorted({p.get("commit_idx") for p in qna_pairs if p.get("commit_idx") is not None})

    if len(commit_idxs) >= 2:
        cutoff = max(1, int(round(len(commit_idxs) * train_frac)))
        train_commits = set(commit_idxs[:cutoff])
        test_commits = set(commit_idxs[cutoff:])
        # Guarantee non-empty test side
        if not test_commits and train_commits:
            shifted = sorted(train_commits)[-1]
            train_commits.discard(shifted)
            test_commits.add(shifted)
        train_pairs = [p for p in qna_pairs if p.get("commit_idx") in train_commits]
        test_pairs = [p for p in qna_pairs if p.get("commit_idx") in test_commits]
        if not train_pairs or not test_pairs:
            # If a side is empty (degenerate distribution) fall back to random.
            return _random_split(qna_pairs, train_frac, rng) + (
                {"mode": "random_fallback", "n_commits": len(commit_idxs)},
            )
        return (
            train_pairs,
            test_pairs,
            {
                "mode": "commit",
                "n_commits": len(commit_idxs),
                "n_train_commits": len(train_commits),
                "n_test_commits": len(test_commits),
            },
        )

    train_pairs, test_pairs = _random_split(qna_pairs, train_frac, rng)
    return (
        train_pairs,
        test_pairs,
        {"mode": "random", "n_commits": len(commit_idxs)},
    )


def _random_split(
    qna_pairs: list[dict],
    train_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    shuffled = list(qna_pairs)
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_frac)))
    if n_train >= len(shuffled):
        n_train = len(shuffled) - 1
    return shuffled[:n_train], shuffled[n_train:]


def main() -> None:
    default_input = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
        "gru_ood_test.json",
    )
    default_output = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET_GRU_OOD",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default=default_input)
    ap.add_argument("--output", type=str, default=default_output)
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of (sorted) commits assigned to training (default 0.8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-train-pairs", type=int, default=2,
                    help="Skip repos with fewer than this many train pairs")
    ap.add_argument("--min-test-pairs", type=int, default=1)
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    print(f"Loaded {len(repos)} OOD repos")

    train_root: dict[str, dict] = {"repositories": {}}
    test_root: dict[str, dict] = {"split": "ood_test", "repositories": {}}
    manifest: dict[str, dict] = {}
    skipped: list[str] = []
    n_train_total = 0
    n_test_total = 0

    rng = random.Random(args.seed)
    for repo_name in sorted(repos.keys()):
        r = repos[repo_name]
        pairs = list(r.get("qna_pairs", []))
        if len(pairs) < (args.min_train_pairs + args.min_test_pairs):
            skipped.append(repo_name)
            continue
        # Use a per-repo seeded RNG so the random fallback is deterministic
        # and independent of repo iteration order.
        local_rng = random.Random(f"{args.seed}:{repo_name}")
        train_pairs, test_pairs, info = split_repo_by_commit(
            pairs, args.train_frac, local_rng,
        )
        if len(train_pairs) < args.min_train_pairs or len(test_pairs) < args.min_test_pairs:
            skipped.append(repo_name)
            continue

        train_root["repositories"][repo_name] = {
            "qna_pairs": train_pairs,
            "embedding": r.get("embedding"),
            "commit_history": r.get("commit_history"),
        }
        test_root["repositories"][repo_name] = {
            "qna_pairs": test_pairs,
            "embedding": r.get("embedding"),
            "commit_history": r.get("commit_history"),
        }
        manifest[repo_name] = {
            **info,
            "n_total_pairs": len(pairs),
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
        }
        n_train_total += len(train_pairs)
        n_test_total += len(test_pairs)

    train_path = output_dir / "train.json"
    test_path = output_dir / "ir_test.json"
    manifest_path = output_dir / "manifest.json"

    print(f"Writing {train_path}  ({n_train_total} pairs across {len(manifest)} repos)")
    train_path.write_text(json.dumps(train_root), encoding="utf-8")
    print(f"Writing {test_path}  ({n_test_total} pairs across {len(manifest)} repos)")
    test_path.write_text(json.dumps(test_root), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "train_frac": args.train_frac,
                "seed": args.seed,
                "n_repos_kept": len(manifest),
                "n_repos_skipped": len(skipped),
                "skipped": skipped,
                "n_train_pairs_total": n_train_total,
                "n_test_pairs_total": n_test_total,
                "per_repo": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote manifest -> {manifest_path}")
    if skipped:
        print(f"Skipped {len(skipped)} repos (too few pairs): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Mode breakdown
    mode_counts: dict[str, int] = {}
    for info in manifest.values():
        mode_counts[info["mode"]] = mode_counts.get(info["mode"], 0) + 1
    print(f"Split modes: {mode_counts}")


if __name__ == "__main__":
    main()
