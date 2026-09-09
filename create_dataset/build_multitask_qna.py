#!/usr/bin/env python3
"""Build the multi-task QnA dataset for ``train_code2lora_gru_v2.py``.

Merges the canonical ``assert_rhs`` QnAs with the generated ``qa`` QnAs into a
single qna dir whose parquets carry a ``task`` column so the multi-task head
routes each example via ``task_to_idx``.

Output layout (mirrors what the trainer expects)::

    <out>/qna/train.parquet     assert_rhs[train] + qa[train]   (training set)
    <out>/qna/cr_val.parquet    assert_rhs + qa  (cr_val eval suite)
    <out>/qna/cr_test.parquet   assert_rhs + qa  (cr_test eval suite)

Only the columns the trainer's ``load_qna_rows`` actually reads are kept, so
the two source schemas (which differ in their extra columns) can be unioned
cleanly. The trainer trains exclusively on commits whose *commit-level*
``in_repo_split == "train"``, so the train shard is filtered to
``in_repo_split == "train"`` to stay lean; eval shards keep all rows.

Usage::

    python create_dataset/build_multitask_qna.py \
        --assert-train  $SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet \
        --assert-eval-dir $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna \
        --qa-dir        $SCRATCH/REPO_DATASET/qna_qa_hf/qna \
        --out-dir       $SCRATCH/REPO_DATASET/qna_multitask_assert_qa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Columns the trainer's loader consumes (everything else is dropped). ``task``
# is appended explicitly. Keeping this minimal lets the two schemas union.
KEEP_COLS = [
    "repo_id", "commit_sha", "commit_index",
    "in_repo_split", "cross_repo_split", "prefix", "target",
]

OUT_SCHEMA = pa.schema([
    ("repo_id", pa.string()),
    ("commit_sha", pa.string()),
    ("commit_index", pa.int64()),
    ("in_repo_split", pa.string()),
    ("cross_repo_split", pa.string()),
    ("prefix", pa.string()),
    ("target", pa.string()),
    ("task", pa.string()),
])


def _normalize(table: pa.Table, task: str) -> pa.Table:
    """Project ``table`` onto OUT_SCHEMA, casting and stamping the task."""
    cols = {}
    n = table.num_rows
    for name in KEEP_COLS:
        if name in table.column_names:
            arr = table.column(name)
        else:
            arr = pa.nulls(n, type=OUT_SCHEMA.field(name).type)
        cols[name] = arr.cast(OUT_SCHEMA.field(name).type)
    cols["task"] = pa.array([task] * n, type=pa.string())
    return pa.table(cols, schema=OUT_SCHEMA)


def _load(path: Path, task: str, only_train: bool) -> pa.Table:
    dset = ds.dataset(str(path), format="parquet")
    flt = (pc.field("in_repo_split") == "train") if only_train else None
    cols = [c for c in KEEP_COLS if c in dset.schema.names]
    table = dset.to_table(columns=cols, filter=flt)
    out = _normalize(table, task)
    print(f"    {path.name}: task={task} only_train={only_train} -> {out.num_rows} rows",
          flush=True)
    return out


def _build_shard(out_path: Path, parts) -> None:
    tables = [t for t in parts if t is not None and t.num_rows > 0]
    merged = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, str(out_path), compression="zstd")
    # Per-task row summary.
    import collections
    c = collections.Counter(merged.column("task").to_pylist())
    print(f"  wrote {out_path}  rows={merged.num_rows}  by_task={dict(c)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assert-train", required=True,
                    help="assert_rhs training qna parquet (smart-cap).")
    ap.add_argument("--assert-eval-dir", required=True,
                    help="dir with assert_rhs {cr_val,cr_test}.parquet eval suites.")
    ap.add_argument("--qa-dir", required=True,
                    help="dir with qa {train,cr_val,cr_test}.parquet.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-suites", nargs="+", default=["cr_val", "cr_test"])
    args = ap.parse_args()

    assert_train = Path(args.assert_train)
    assert_eval_dir = Path(args.assert_eval_dir)
    qa_dir = Path(args.qa_dir)
    out_qna = Path(args.out_dir) / "qna"

    print("Building multi-task train shard ...", flush=True)
    _build_shard(out_qna / "train.parquet", [
        _load(assert_train, "assert_rhs", only_train=True),
        _load(qa_dir / "train.parquet", "qa", only_train=True),
    ])

    for suite in args.eval_suites:
        a_path = assert_eval_dir / f"{suite}.parquet"
        q_path = qa_dir / f"{suite}.parquet"
        parts = []
        if a_path.exists():
            parts.append(_load(a_path, "assert_rhs", only_train=False))
        else:
            print(f"  [skip] missing {a_path}", flush=True)
        if q_path.exists():
            parts.append(_load(q_path, "qa", only_train=False))
        else:
            print(f"  [skip] missing {q_path}", flush=True)
        if parts:
            print(f"Building multi-task {suite} eval shard ...", flush=True)
            _build_shard(out_qna / f"{suite}.parquet", parts)

    print(f"\nDone. Multi-task qna dir: {out_qna}", flush=True)


if __name__ == "__main__":
    main()
