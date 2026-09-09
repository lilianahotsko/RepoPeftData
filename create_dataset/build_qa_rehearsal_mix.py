#!/usr/bin/env python
"""Build the qa + assert_rhs rehearsal-mix training parquet.

Motivation (EMNLP demo paper): the qa-only finetune of the multi-task head
drops aggregate assertion EM (rehearsal-free drift). This script builds a
mixed train set = ALL qa QnAs + a small rehearsal sample of assert_rhs QnAs,
so `train_c2l_evo_multitask.sh` can re-run with QNAS_DIR pointed here and
quantify how much retention a light rehearsal buys at equal budget.

Sampling protocol (deterministic, seed 3407):
  * assert_rhs source: the smart-capped training QnAs
    (in_repo_split == 'train' rows only -- never rehearse on ir_val/ir_test).
  * group by (repo_id, commit_sha); sample REHEARSAL_COMMIT_FRAC of the
    groups; keep the first QnA of each sampled group (1/commit).
    This touches many repos/commits shallowly instead of few deeply, matching
    how the GRU walk consumes data.

Output schema is the intersection the trainer's loader needs; assert_rhs rows
get an explicit task column so the multi-task head routes them to index 0.

Usage:
    python create_dataset/build_qa_rehearsal_mix.py \
        --qa-parquet   $SCRATCH/REPO_DATASET/qna_qa_hf/qna/train.parquet \
        --assert-parquet $SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet \
        --out-dir      $SCRATCH/REPO_DATASET/qna_qa_mix_rehearsal_hf \
        --commit-frac  0.25
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = [
    "repo_id", "cross_repo_split", "commit_index", "commit_sha",
    "in_repo_split", "task", "prefix", "target",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-parquet", required=True)
    ap.add_argument("--assert-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--commit-frac", type=float, default=0.25,
                    help="fraction of assert_rhs (repo, commit) groups to sample")
    ap.add_argument("--per-commit", type=int, default=1,
                    help="rehearsal QnAs kept per sampled commit")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- qa side: keep every row, project to the shared schema ----
    qa = pq.read_table(args.qa_parquet)
    qa_cols = {}
    for c in COLUMNS:
        qa_cols[c] = qa.column(c) if c in qa.column_names else \
            pa.array(["qa"] * qa.num_rows, type=pa.string())
    qa_t = pa.table(qa_cols)
    print(f"qa rows          : {qa_t.num_rows}")

    # ---- assert side: train-split rows only, sample by commit group ----
    # Two streaming passes keep memory bounded (prefixes are large; the full
    # 1.2M-row table OOMs a login node).
    pf = pq.ParquetFile(args.assert_parquet)

    # Pass 1: light columns only -> global row index of the first
    # `per_commit` rows of every train-split (repo, commit) group.
    groups: dict[tuple, list[int]] = {}
    gidx = 0
    for batch in pf.iter_batches(batch_size=65536,
                                 columns=["repo_id", "commit_sha",
                                          "in_repo_split"]):
        repo = batch.column(0).to_pylist()
        sha = batch.column(1).to_pylist()
        irs = batch.column(2).to_pylist()
        for r, s, sp in zip(repo, sha, irs):
            if sp == "train":
                lst = groups.setdefault((r, s), [])
                if len(lst) < args.per_commit:
                    lst.append(gidx)
            gidx += 1
    keys = sorted(groups)  # deterministic order before sampling
    n_sample = int(len(keys) * args.commit_frac)
    sampled = rng.sample(keys, n_sample)
    take = set()
    for k in sampled:
        take.update(groups[k])

    # Pass 2: stream full columns, keep only the selected global rows.
    kept_batches = []
    gidx = 0
    read_cols = [c for c in COLUMNS if c != "task"]
    for batch in pf.iter_batches(batch_size=65536, columns=read_cols):
        n = batch.num_rows
        local = [i for i in range(n) if (gidx + i) in take]
        if local:
            kept_batches.append(
                pa.Table.from_batches([batch]).take(pa.array(local)))
        gidx += n
    at_s = pa.concat_tables(kept_batches)
    at_cols = {c: at_s.column(c) for c in at_s.column_names}
    at_cols["task"] = pa.array(["assert_rhs"] * at_s.num_rows, type=pa.string())
    at_t = pa.table({c: at_cols[c] for c in COLUMNS})
    print(f"assert groups    : {len(keys)} total, {n_sample} sampled "
          f"({args.commit_frac:.0%}), {at_t.num_rows} rehearsal rows")

    at_t = at_t.cast(qa_t.schema)  # unify string vs large_string columns
    mixed = pa.concat_tables([qa_t, at_t])
    print(f"mixed rows       : {mixed.num_rows} "
          f"(rehearsal share {at_t.num_rows / mixed.num_rows:.1%})")

    out = Path(args.out_dir) / "qna"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(mixed, out / "train.parquet")
    print(f"wrote {out / 'train.parquet'}")


if __name__ == "__main__":
    main()
