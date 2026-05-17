#!/usr/bin/env python3
"""Build the ``repopeft-code2lora-snapshots`` HuggingFace dataset (Dataset B).

This is the static Code2LoRA dataset: every example is a single
``(repo, commit, repo_state_embedding, qna)`` row -- no GRU rollout needed.

Inputs (already produced by the rest of the pipeline):
    --v2-commits-dir   commit_parquet_hf_v2/commits/{train,cr_val,cr_test}.parquet
                       (must contain ``repo_state_embedding`` -- run
                       ``merge_gru_v2_embeddings.py`` first)
    --base-qna-dir     commit_parquet_hf/qna/{train,cr_val,cr_test}.parquet
                       (the canonical, unfiltered RepoPeftBench QnAs; used for
                       all eval QnAs except for ``train``)
    --train-qnas-json  static_commit/splits_at_anchor/train.json
                       (re-extracted v2 QnAs at the anchor commit -- one
                       repository entry per train repo)

Outputs (written under --out-dir):

    commits/
        train.parquet      400 rows: anchor commits only (latest in_repo='train'
                           per train repo) with repo_state_embedding.
        ir_val.parquet     train repos x in_repo_split=='val' commits.
        ir_test.parquet    train repos x in_repo_split=='test' commits.
        cr_val.parquet     all cr_val commits.
        cr_test.parquet    all cr_test commits.
    qna/
        train.parquet      ~44 k re-extracted v2 QnAs at anchor (from JSON).
        ir_val.parquet     base qna/train.parquet rows w/ in_repo_split=='val'.
        ir_test.parquet    base qna/train.parquet rows w/ in_repo_split=='test'.
        cr_val.parquet     copy of base qna/cr_val.parquet.
        cr_test.parquet    copy of base qna/cr_test.parquet.
    SNAPSHOTS_README.json  provenance and row counts.

Schema notes:
    * The five commits parquets share the schema of v2's commits parquet,
      minus ``diff_embedding`` (which is GRU-only): repo_id, cross_repo_split,
      commit_index, commit_sha, commit_timestamp, in_repo_split,
      production_code_diff (dropped to save space), n_*_assertions, plus
      ``repo_state_embedding`` (list<float16>[2048]).
    * The qna parquets keep the canonical column set verbatim. The train
      qna parquet adds: ``anchor_sha``, ``anchor_index``, ``qna_source``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


DEFAULT_V2_COMMITS_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2/commits"
DEFAULT_BASE_QNA_DIR = "/scratch/lhotsko/REPO_DATASET/commit_parquet_hf/qna"
DEFAULT_TRAIN_QNAS_JSON = "/scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor/train.json"
DEFAULT_OUT_DIR = "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf"
EMBED_DIM = 2048

# Columns to keep in the commits parquets (drop production_code_diff).
COMMIT_OUT_COLS = [
    "repo_id", "cross_repo_split", "commit_index", "commit_sha",
    "commit_timestamp", "in_repo_split",
    "n_new_assertions", "n_added_assertions", "n_modified_assertions",
    "repo_state_embedding",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_table(table: pa.Table, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    out_path.unlink(missing_ok=True)
    tmp.rename(out_path)


def _select_anchor_rows(v2_train: pa.Table) -> pa.Table:
    """Pick the row with maximum commit_index per repo_id among in_repo_split=='train'."""
    mask = pc.equal(v2_train["in_repo_split"], pa.scalar("train"))
    sub = v2_train.filter(mask)
    n = sub.num_rows
    repos = sub.column("repo_id").to_pylist()
    idxs = sub.column("commit_index").to_pylist()
    best: Dict[str, int] = {}
    for i, (r, c) in enumerate(zip(repos, idxs)):
        if r not in best or c > idxs[best[r]]:
            best[r] = i
    keep_idx = sorted(best.values())
    return sub.take(keep_idx)


def _filter_by_in_repo_split(v2_train: pa.Table, value: str) -> pa.Table:
    mask = pc.equal(v2_train["in_repo_split"], pa.scalar(value))
    return v2_train.filter(mask)


def _trim_commit_cols(t: pa.Table) -> pa.Table:
    keep = [c for c in COMMIT_OUT_COLS if c in t.schema.names]
    return t.select(keep)


def _build_train_qna_parquet(train_json: Path) -> pa.Table:
    """Read splits_at_anchor/train.json and flatten into a parquet table.

    The JSON layout is::

        {"repositories": {
            "<repo_id>": {
                "embedding": [...2048...],
                "qna_pairs": [{
                    "prefix": str, "target": str,
                    "assertion_type": str, "difficulty": str,
                    "context_lines": int, "metadata": {...}
                }, ...],
                "metadata": {
                    "repo_id", "commit_sha", "commit_index",
                    "anchor_sha", "anchor_index", ...
                }
            },
            ...
        }}
    """
    print(f"  Loading {train_json} ...", flush=True)
    obj = json.loads(train_json.read_text(encoding="utf-8"))
    repos = obj.get("repositories", {})
    rows: List[Dict] = []
    for repo_id, item in repos.items():
        md = item.get("metadata", {})
        anchor_sha = md.get("anchor_sha", md.get("commit_sha", ""))
        anchor_idx = int(md.get("anchor_index", md.get("commit_index", -1)))
        for pair in item.get("qna_pairs", []):
            pmd = pair.get("metadata", {})
            rows.append({
                "repo_id": repo_id,
                "commit_sha": anchor_sha,
                "commit_index": anchor_idx,
                "in_repo_split": "train",
                "cross_repo_split": "train",
                "test_file": pmd.get("test_file", ""),
                "test_function": pmd.get("test_function", ""),
                "assertion_type": pair.get("assertion_type", ""),
                "difficulty": pair.get("difficulty", ""),
                "context_lines": int(pair.get("context_lines", 0)),
                "prefix": pair.get("prefix", ""),
                "target": pair.get("target", ""),
                "anchor_sha": anchor_sha,
                "anchor_index": anchor_idx,
                "qna_source": "v2_extractor_at_anchor",
            })
    print(f"  -> {len(rows)} train QnA rows.", flush=True)
    if not rows:
        raise SystemExit("train.json produced no QnA rows; aborting.")

    schema = pa.schema([
        ("repo_id", pa.string()),
        ("commit_sha", pa.string()),
        ("commit_index", pa.int32()),
        ("in_repo_split", pa.string()),
        ("cross_repo_split", pa.string()),
        ("test_file", pa.string()),
        ("test_function", pa.string()),
        ("assertion_type", pa.string()),
        ("difficulty", pa.string()),
        ("context_lines", pa.int32()),
        ("prefix", pa.large_string()),
        ("target", pa.large_string()),
        ("anchor_sha", pa.string()),
        ("anchor_index", pa.int32()),
        ("qna_source", pa.string()),
    ])
    # Column-major
    cols = {field.name: [r.get(field.name) for r in rows] for field in schema}
    return pa.table(cols, schema=schema)


def _filter_qna_in_repo_split(base_qna_path: Path,
                              value: str,
                              tmp_dir: Path) -> pa.Table:
    """Filter base qna/train.parquet rows where in_repo_split == value.

    Uses pyarrow.dataset streaming so we never materialize the 15-GB file in
    memory all at once.
    """
    print(f"  Streaming {base_qna_path} for in_repo_split=='{value}' ...",
          flush=True)
    ds = pads.dataset(base_qna_path, format="parquet")
    mask = pc.equal(pads.field("in_repo_split"), pa.scalar(value))
    scanner = ds.scanner(filter=mask, batch_size=200_000)
    parts: List[pa.RecordBatch] = []
    n_rows = 0
    t0 = time.time()
    for batch in scanner.to_batches():
        parts.append(batch)
        n_rows += batch.num_rows
        if len(parts) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    batches={len(parts)} rows={n_rows} ({elapsed:.0f}s)",
                  flush=True)
    if not parts:
        # Return an empty table with schema matching the source.
        return ds.schema.empty_table()
    table = pa.Table.from_batches(parts)
    print(f"  -> {table.num_rows} rows", flush=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-commits-dir", default=DEFAULT_V2_COMMITS_DIR)
    ap.add_argument("--base-qna-dir", default=DEFAULT_BASE_QNA_DIR)
    ap.add_argument("--train-qnas-json", default=DEFAULT_TRAIN_QNAS_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip-large-qna", action="store_true",
                    help="Skip producing ir_val/ir_test (which require reading "
                         "the 15-GB base qna/train.parquet). For smoke tests.")
    args = ap.parse_args()

    v2 = Path(args.v2_commits_dir)
    base_qna = Path(args.base_qna_dir)
    train_json = Path(args.train_qnas_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "commits").mkdir(exist_ok=True)
    (out_dir / "qna").mkdir(exist_ok=True)

    summary: Dict[str, Dict] = {}

    # ----- 1. Commits -----
    print("\n[commits] reading v2 train ...", flush=True)
    v2_train = pq.read_table(v2 / "train.parquet", memory_map=True)
    print(f"  v2 train rows = {v2_train.num_rows}", flush=True)

    anchor = _select_anchor_rows(v2_train)
    print(f"  train (anchors): {anchor.num_rows}", flush=True)
    p = out_dir / "commits" / "train.parquet"
    _write_table(_trim_commit_cols(anchor), p)
    summary["commits/train"] = {"rows": anchor.num_rows,
                                "sha256": _sha256(p), "path": str(p)}

    ir_val = _filter_by_in_repo_split(v2_train, "val")
    print(f"  ir_val: {ir_val.num_rows}", flush=True)
    p = out_dir / "commits" / "ir_val.parquet"
    _write_table(_trim_commit_cols(ir_val), p)
    summary["commits/ir_val"] = {"rows": ir_val.num_rows,
                                 "sha256": _sha256(p), "path": str(p)}

    ir_test = _filter_by_in_repo_split(v2_train, "test")
    print(f"  ir_test: {ir_test.num_rows}", flush=True)
    p = out_dir / "commits" / "ir_test.parquet"
    _write_table(_trim_commit_cols(ir_test), p)
    summary["commits/ir_test"] = {"rows": ir_test.num_rows,
                                  "sha256": _sha256(p), "path": str(p)}
    del v2_train, anchor, ir_val, ir_test

    for split in ("cr_val", "cr_test"):
        print(f"\n[commits] {split} ...", flush=True)
        t = pq.read_table(v2 / f"{split}.parquet", memory_map=True)
        t = _trim_commit_cols(t)
        p = out_dir / "commits" / f"{split}.parquet"
        _write_table(t, p)
        summary[f"commits/{split}"] = {"rows": t.num_rows,
                                       "sha256": _sha256(p), "path": str(p)}
        print(f"  {split}: {t.num_rows} rows", flush=True)
        del t

    # ----- 2. QnAs -----
    print("\n[qna] train (re-extracted v2 at anchor) ...", flush=True)
    train_qna = _build_train_qna_parquet(train_json)
    p = out_dir / "qna" / "train.parquet"
    _write_table(train_qna, p)
    summary["qna/train"] = {"rows": train_qna.num_rows,
                            "sha256": _sha256(p), "path": str(p)}
    del train_qna

    for split in ("cr_val", "cr_test"):
        print(f"\n[qna] copying base qna/{split}.parquet ...", flush=True)
        src = base_qna / f"{split}.parquet"
        if src.is_symlink():
            src = src.resolve()
        dst = out_dir / "qna" / f"{split}.parquet"
        dst.unlink(missing_ok=True)
        shutil.copy2(src, dst)
        md = pq.read_metadata(dst)
        summary[f"qna/{split}"] = {"rows": md.num_rows,
                                   "sha256": _sha256(dst), "path": str(dst)}

    if not args.skip_large_qna:
        for value, name in (("val", "ir_val"), ("test", "ir_test")):
            print(f"\n[qna] filtering base qna/train.parquet -> {name} ...",
                  flush=True)
            t = _filter_qna_in_repo_split(
                base_qna / "train.parquet", value, out_dir)
            p = out_dir / "qna" / f"{name}.parquet"
            _write_table(t, p)
            summary[f"qna/{name}"] = {"rows": t.num_rows,
                                      "sha256": _sha256(p), "path": str(p)}
            del t
    else:
        print("\n[qna] --skip-large-qna; ir_val/ir_test NOT generated.",
              flush=True)

    readme = {
        "v2_commits_dir": str(v2),
        "base_qna_dir": str(base_qna),
        "train_qnas_json": str(train_json),
        "out_dir": str(out_dir),
        "embed_dim": EMBED_DIM,
        "splits": summary,
    }
    (out_dir / "SNAPSHOTS_README.json").write_text(
        json.dumps(readme, indent=2), encoding="utf-8")
    print("\n=== Summary ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: rows={v['rows']:>8d}  sha={v['sha256'][:12]}...",
              flush=True)
    print(f"\nWrote {out_dir / 'SNAPSHOTS_README.json'}", flush=True)


if __name__ == "__main__":
    main()
