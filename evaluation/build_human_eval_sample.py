#!/usr/bin/env python3
"""Build a stratified human-evaluation sample of repository-QA answers.

Reads the scored QA records (records_qa_scored.jsonl), stratifies by
qa_quality quartile and deduplicates by repo, and emits:

  - qa_human_eval_sample.csv   blind annotation sheet (no automatic scores)
  - qa_human_eval_key.jsonl    hidden key with metrics for later analysis

Usage:
  python evaluation/build_human_eval_sample.py \
      --records .../records_qa_scored.jsonl \
      --out-dir Code2LoRADemo/human_eval \
      --n 100 --seed 13
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-per-repo", type=int, default=2)
    args = ap.parse_args()

    records = []
    with open(args.records) as f:
        for line in f:
            r = json.loads(line)
            if r.get("prediction") and r.get("target"):
                records.append(r)
    print(f"loaded {len(records)} scored QA records")

    records.sort(key=lambda r: r["qa_quality"])
    n_q = len(records) // 4
    quartiles = [
        records[:n_q],
        records[n_q : 2 * n_q],
        records[2 * n_q : 3 * n_q],
        records[3 * n_q :],
    ]

    rng = random.Random(args.seed)
    per_stratum = args.n // 4
    sample: list[dict] = []
    repo_counts: defaultdict[str, int] = defaultdict(int)
    for q in quartiles:
        rng.shuffle(q)
        taken = 0
        for r in q:
            if taken >= per_stratum:
                break
            if repo_counts[r["repo_id"]] >= args.max_per_repo:
                continue
            repo_counts[r["repo_id"]] += 1
            sample.append(r)
            taken += 1
    rng.shuffle(sample)  # blind annotators to the quality stratum

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = out_dir / "qa_human_eval_sample.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_id",
                "repo",
                "question",
                "reference_answer",
                "model_answer",
                "correctness (2=correct, 1=partially, 0=incorrect)",
                "grounding (1=repo-specific, 0=generic/hallucinated)",
                "notes",
            ]
        )
        for i, r in enumerate(sample, 1):
            w.writerow(
                [
                    f"Q{i:03d}",
                    r["repo_id"],
                    r["prefix"].strip(),
                    r["target"].strip(),
                    r["prediction"].strip(),
                    "",
                    "",
                    "",
                ]
            )

    key = out_dir / "qa_human_eval_key.jsonl"
    with open(key, "w") as f:
        for i, r in enumerate(sample, 1):
            f.write(
                json.dumps(
                    {
                        "item_id": f"Q{i:03d}",
                        "repo_id": r["repo_id"],
                        "commit_sha": r.get("commit_sha"),
                        "suite": r.get("suite"),
                        "qa_quality": r["qa_quality"],
                        "rougeL": r.get("rougeL"),
                        "token_f1": r.get("token_f1"),
                    }
                )
                + "\n"
            )

    print(f"wrote {len(sample)} items -> {sheet}")
    print(f"hidden key            -> {key}")
    print(f"repos covered: {len(repo_counts)}")


if __name__ == "__main__":
    main()
