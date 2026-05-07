#!/usr/bin/env python3
"""Build a "smart-capped" snapshot of the GRU-commit QnA parquet.

The Code2LoRA-GRU<sub>commit</sub> training set is heavy-tailed: a few
commits contribute thousands of QnA rows, but most contribute one or two.
The current trainer side-steps this with ``--max-assertions-per-commit 8``,
which uniformly samples 8 rows per commit. That cap is *file-blind*: a
commit that touches ``tests/test_a.py`` with 200 assertions and
``tests/test_b.py`` with 8 assertions will spend most of its 8-budget on
``test_a.py``.

This script writes a smaller, fixed training-set snapshot that:

1. drops trivial-target rows (``len(target.strip()) < 4`` and the bare
   ``")"`` / leading-comma cases);
2. round-robin samples within each commit across (test_file, test_function)
   groups so file/function diversity is preserved;
3. caps each commit at ``--max-per-commit`` and each
   (commit, file) at ``--max-per-file``;
4. only touches rows with ``in_repo_split=='train'``: ``val`` / ``test``
   rows are passed through verbatim, so cross-repo evaluation is identical.

Output layout mirrors ``commit_parquet_hf``:

    <out-dir>/commits/{train,cr_val,cr_test}.parquet     (symlinks)
    <out-dir>/qna/train.parquet                           (rewritten)
    <out-dir>/qna/cr_val.parquet                          (symlinks)
    <out-dir>/qna/cr_test.parquet                         (symlinks)
    <out-dir>/splits/...                                  (symlinks)
    <out-dir>/SMART_CAP_README.json                       (provenance)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


SPLIT_FILES = {
    "train": "train.parquet",
    "cr_val": "cr_val.parquet",
    "cr_test": "cr_test.parquet",
}


def is_trivial_target(target: Optional[str], min_chars: int) -> bool:
    """Match the trivial-target heuristics quantified in
    ``analysis/smart_cap_qna_analysis.py`` (drops ~20% of train rows on
    ``commit_parquet_hf``: bare ``)``, ``,``, ``,)``, etc.).
    """
    if target is None:
        return True
    t = target.strip()
    if len(t) < min_chars:
        return True
    # Closing-paren-only / leading-comma trivia.
    if len(t) <= 6 and (t.endswith(")") or t.startswith(",")):
        return True
    return False


def smart_cap_one_commit(
    rows: List[Dict[str, Any]],
    max_per_file: int,
    max_per_commit: int,
    rng: random.Random,
) -> List[int]:
    """Return indices into ``rows`` (which all share one commit) selected by:

    1. Group by (test_file, test_function); shuffle each group.
    2. Truncate each group to ``max_per_file`` (treating the file alone, so
       all functions inside one file collectively respect ``max_per_file``).
    3. Round-robin across (file, function) groups, taking 1 from each in a
       randomized file/func order, until the per-commit budget is filled.

    Single-file commits collapse to "random K-of-N" sampling (preserving
    backwards-compatibility with the current cap). Multi-file commits get
    file/function diversity for free.
    """
    if not rows:
        return []
    # Bucket by file -> list of (orig_index, function).
    by_file: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r.get("test_file") or ""].append(
            (i, r.get("test_function") or "")
        )

    # Per-file pre-cap (max_per_file rows per file). Inside the file we still
    # want function diversity, so do a round-robin per (function) within file.
    file_budgets: Dict[str, List[int]] = {}
    for f, items in by_file.items():
        # Shuffle items so the function ordering is random.
        items = list(items)
        rng.shuffle(items)
        # Group by function inside this file.
        by_func: Dict[str, List[int]] = defaultdict(list)
        for i, fn in items:
            by_func[fn].append(i)
        # Round-robin across functions until file budget is filled.
        fn_keys = list(by_func.keys())
        rng.shuffle(fn_keys)
        chosen: List[int] = []
        progressed = True
        while progressed and len(chosen) < max_per_file:
            progressed = False
            for fk in fn_keys:
                if not by_func[fk]:
                    continue
                chosen.append(by_func[fk].pop())
                progressed = True
                if len(chosen) >= max_per_file:
                    break
        file_budgets[f] = chosen

    # Per-commit round-robin across files until commit budget is filled.
    file_keys = list(file_budgets.keys())
    rng.shuffle(file_keys)
    selected: List[int] = []
    progressed = True
    while progressed and len(selected) < max_per_commit:
        progressed = False
        for fk in file_keys:
            if not file_budgets[fk]:
                continue
            selected.append(file_budgets[fk].pop())
            progressed = True
            if len(selected) >= max_per_commit:
                break
    return selected


def stream_qna_in_repo_train(
    qna_path: Path,
    columns: Sequence[str],
    *,
    batch_size: int,
) -> Iterable["pa.RecordBatch"]:
    """Yield only ``in_repo_split == 'train'`` rows so val/test rows
    remain bit-identical via the symlink path.
    """
    ds = pads.dataset(str(qna_path), format="parquet")
    scanner = ds.scanner(
        columns=list(columns),
        filter=pc.field("in_repo_split") == "train",
        batch_size=batch_size,
    )
    yield from scanner.to_batches()


def list_train_qna_columns(qna_path: Path) -> List[str]:
    sch = pq.read_schema(qna_path)
    return [f.name for f in sch]


def smart_cap_train_qna(
    qna_path: Path,
    out_path: Path,
    *,
    max_per_file: int,
    max_per_commit: int,
    min_target_chars: int,
    seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Two-pass writer:

    * Pass 1: copy every row whose ``in_repo_split != 'train'`` verbatim
      (uses Arrow record-batch passthrough; no Python row materialization).
    * Pass 2: stream ``in_repo_split == 'train'`` rows ordered by
      (repo, commit), apply quality filter + smart cap, write capped
      rows. The original QnA parquet is sorted by (repo, commit), so a
      single forward scan is enough.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(qna_path)
    columns = [f.name for f in schema]
    print(f"  schema cols ({len(columns)}): {columns}", flush=True)

    # ``data_page_size`` keeps internal buffers small; ``write_batch_size``
    # makes each ``write_batch`` flush its own row group instead of pinning
    # everything in RAM until ``close()``.
    writer = pq.ParquetWriter(
        str(out_path),
        schema,
        compression="snappy",
        data_page_size=8 * 1024 * 1024,
        write_batch_size=batch_size,
    )
    repos_seen: set = set()
    repos_with_train_kept: set = set()
    commits_with_train_kept: set = set()
    n_in_train = 0
    n_kept_train = 0
    n_dropped_trivial = 0
    n_dropped_smart = 0
    n_eval_rows = 0
    n_total_rows = 0

    # ── Pass 1: passthrough non-train rows ──
    t0 = time.time()
    ds = pads.dataset(str(qna_path), format="parquet")
    eval_filter = pc.not_equal(pc.field("in_repo_split"), "train")
    eval_scanner = ds.scanner(
        columns=columns, filter=eval_filter, batch_size=batch_size,
    )
    import gc
    eval_buffer: List["pa.RecordBatch"] = []
    eval_buffer_rows = 0
    eval_flush_rows = max(batch_size * 4, 8192)

    def flush_eval_buffer() -> None:
        nonlocal eval_buffer_rows
        if not eval_buffer:
            return
        tbl = pa.Table.from_batches(eval_buffer, schema=schema)
        writer.write_table(tbl, row_group_size=tbl.num_rows)
        eval_buffer.clear()
        eval_buffer_rows = 0

    for bi, rb in enumerate(eval_scanner.to_batches()):
        if rb.num_rows == 0:
            continue
        eval_buffer.append(rb)
        eval_buffer_rows += rb.num_rows
        n_eval_rows += rb.num_rows
        n_total_rows += rb.num_rows
        if eval_buffer_rows >= eval_flush_rows:
            flush_eval_buffer()
        if bi % 50 == 0:
            gc.collect()
            print(
                f"  [pass1 eval-passthrough] {n_eval_rows:,} rows in "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    flush_eval_buffer()
    print(
        f"  [pass1] passthrough done: {n_eval_rows:,} rows in "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )

    # ── Pass 2: train rows, smart-capped per commit ──
    rng = random.Random(seed)
    pending_train_rows: List[Dict[str, Any]] = []
    out_buffer: List[Dict[str, Any]] = []
    out_flush_threshold = max(batch_size, 4096)
    current_commit: Optional[Tuple[str, int]] = None

    def flush_out_buffer() -> None:
        if not out_buffer:
            return
        tbl = pa.Table.from_pylist(out_buffer, schema=schema)
        writer.write_table(tbl, row_group_size=tbl.num_rows)
        out_buffer.clear()

    def flush_commit() -> None:
        nonlocal n_dropped_smart, n_kept_train
        if not pending_train_rows:
            return
        kept_idx = smart_cap_one_commit(
            pending_train_rows,
            max_per_file=max_per_file,
            max_per_commit=max_per_commit,
            rng=rng,
        )
        n_dropped_smart += len(pending_train_rows) - len(kept_idx)
        n_kept_train += len(kept_idx)
        if kept_idx:
            for i in sorted(kept_idx):
                out_buffer.append(pending_train_rows[i])
            commits_with_train_kept.add(current_commit)
            if current_commit is not None:
                repos_with_train_kept.add(current_commit[0])
            if len(out_buffer) >= out_flush_threshold:
                flush_out_buffer()
        pending_train_rows.clear()

    train_filter = pc.equal(pc.field("in_repo_split"), "train")
    train_scanner = ds.scanner(
        columns=columns, filter=train_filter, batch_size=batch_size,
    )
    t1 = time.time()
    for bi, rb in enumerate(train_scanner.to_batches()):
        if rb.num_rows == 0:
            continue
        n_in_train += rb.num_rows
        n_total_rows += rb.num_rows
        repo = rb.column("repo_id").to_pylist()
        ci = rb.column("commit_index").to_pylist()
        target = rb.column("target").to_pylist()
        # Materialize the full row dicts only after the trivial-target
        # filter has dropped ~20% of rows.
        for j in range(rb.num_rows):
            repos_seen.add(repo[j])
            if is_trivial_target(target[j], min_target_chars):
                n_dropped_trivial += 1
                continue
            key = (repo[j], int(ci[j]))
            if current_commit is None:
                current_commit = key
            elif key != current_commit:
                flush_commit()
                current_commit = key
            row = {col: rb.column(col)[j].as_py() for col in columns}
            pending_train_rows.append(row)
        if bi % 20 == 0:
            print(
                f"  [pass2 train-cap] {n_in_train:,} rows scanned in "
                f"{time.time() - t1:.0f}s "
                f"(kept={n_kept_train:,}, dropped_trivial={n_dropped_trivial:,}, "
                f"dropped_smart={n_dropped_smart:,})",
                flush=True,
            )

    flush_commit()
    flush_out_buffer()
    writer.close()

    elapsed = time.time() - t0
    return {
        "qna_path_in": str(qna_path),
        "qna_path_out": str(out_path),
        "rows_in_total": n_total_rows,
        "rows_in_train": n_in_train,
        "rows_in_eval": n_eval_rows,
        "rows_out_train_capped": n_kept_train,
        "rows_out_eval_passthrough": n_eval_rows,
        "rows_dropped_trivial_target": n_dropped_trivial,
        "rows_dropped_smart_cap": n_dropped_smart,
        "n_repos_seen": len(repos_seen),
        "n_repos_with_train_kept": len(repos_with_train_kept),
        "n_commits_with_train_kept": len(commits_with_train_kept),
        "max_per_file": max_per_file,
        "max_per_commit": max_per_commit,
        "min_target_chars": min_target_chars,
        "seed": seed,
        "elapsed_sec": elapsed,
    }


def make_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> None:
    default_in = (
        Path(os.environ.get("SCRATCH", str(Path.home() / "scratch")))
        / "REPO_DATASET" / "commit_parquet_hf"
    )
    default_out = (
        Path(os.environ.get("SCRATCH", str(Path.home() / "scratch")))
        / "REPO_DATASET" / "commit_parquet_hf_smartcap"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=default_in)
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--max-per-file", type=int, default=4)
    ap.add_argument("--max-per-commit", type=int, default=8)
    ap.add_argument("--min-target-chars", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument(
        "--splits", nargs="+", default=["train"],
        help="Cross-repo splits whose train assertions to cap "
             "(default: train only; cr_val/cr_test are always symlinked).",
    )
    args = ap.parse_args()

    in_dir = args.in_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Smart-cap GRU-commit QnA parquet ===", flush=True)
    print(f"  in:  {in_dir}", flush=True)
    print(f"  out: {out_dir}", flush=True)
    print(
        f"  max_per_file={args.max_per_file} "
        f"max_per_commit={args.max_per_commit} "
        f"min_target_chars={args.min_target_chars} "
        f"seed={args.seed}",
        flush=True,
    )

    # Symlink commits/, splits/ verbatim (commits aren't capped).
    for sub in ("commits", "splits"):
        src_sub = in_dir / sub
        dst_sub = out_dir / sub
        if not src_sub.exists():
            continue
        if dst_sub.is_symlink() or dst_sub.exists():
            if dst_sub.is_symlink():
                dst_sub.unlink()
        if not dst_sub.exists():
            os.symlink(src_sub.resolve(), dst_sub)

    qna_in = in_dir / "qna"
    qna_out = out_dir / "qna"
    qna_out.mkdir(parents=True, exist_ok=True)

    # Symlink eval splits.
    for split, fname in SPLIT_FILES.items():
        if split in args.splits:
            continue
        src = qna_in / fname
        dst = qna_out / fname
        if src.exists():
            make_link(src, dst)
            print(f"  symlinked {dst.name} -> {src}", flush=True)

    # Cap requested cross-repo splits.
    summary: Dict[str, Any] = {}
    for split in args.splits:
        fname = SPLIT_FILES[split]
        src = qna_in / fname
        if not src.exists():
            print(f"  WARN: missing {src}; skipping", flush=True)
            continue
        dst = qna_out / fname
        print(f"\n  capping {src} -> {dst}", flush=True)
        stat = smart_cap_train_qna(
            src,
            dst,
            max_per_file=args.max_per_file,
            max_per_commit=args.max_per_commit,
            min_target_chars=args.min_target_chars,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        summary[split] = stat
        print(json.dumps(stat, indent=2), flush=True)

    # Provenance file.
    provenance = {
        "type": "code2lora_gru_commit_smartcap",
        "in_dir": str(in_dir),
        "max_per_file": args.max_per_file,
        "max_per_commit": args.max_per_commit,
        "min_target_chars": args.min_target_chars,
        "seed": args.seed,
        "splits_capped": args.splits,
        "splits_passthrough": [
            s for s in SPLIT_FILES if s not in args.splits
        ],
        "per_split": summary,
        "notes": (
            "Only in_repo_split=='train' rows are capped. val / test rows "
            "are passed through verbatim. Quality filter drops trivial "
            "targets (length < min_target_chars or bare ')' / leading-comma "
            "tokens). Smart cap interleaves rows across (test_file, "
            "test_function) groups before truncating to max_per_commit."
        ),
    }
    out_provenance = out_dir / "SMART_CAP_README.json"
    out_provenance.write_text(
        json.dumps(provenance, indent=2), encoding="utf-8",
    )
    print(f"\nWrote {out_provenance}", flush=True)


if __name__ == "__main__":
    main()
