#!/usr/bin/env python3
"""Export the commit-level Parquet dataset to a HuggingFace-ready layout.

This packages the per-repo Parquet shards under
``$PARQUET_DIR/shards/`` plus the cross-repo split JSONs
(``train.json``, ``cr_val.json``, ``cr_test.json``) into a directory
that:

* can be uploaded to the HuggingFace Hub as a dataset repo (``--hub-repo``);
* can be consumed on a new cluster by the unchanged loader in
  ``hypernetwork/parquet_commit_dataset.py`` (``prefer=shards`` or
  ``prefer=concat``).

The output layout mirrors what HF Datasets auto-discovers for multi-config
Parquet repos:

    <output>/
        commits/
            train.parquet
            cr_val.parquet
            cr_test.parquet
        qna/
            train.parquet
            cr_val.parquet
            cr_test.parquet
        shards/             # optional, copied as-is when --keep-shards
            *.commits.parquet
            *.qna.parquet
        splits/
            train.json
            cr_val.json
            cr_test.json
        README.md

The per-split ``commits/*.parquet`` and ``qna/*.parquet`` files are produced
by streaming the shards through a ``pyarrow.dataset`` filter so the exporter
never materializes the entire 15 GB dataset in RAM.

Examples
--------
Save locally (streams, bounded memory)::

    python create_dataset/export_commit_parquet_to_hf.py \\
        --parquet-dir  $SCRATCH/REPO_DATASET/commit_parquet \\
        --splits-dir   $SCRATCH/REPO_DATASET \\
        --output-dir   $SCRATCH/REPO_DATASET/commit_parquet_hf \\
        --keep-shards

Push to the HuggingFace Hub (requires ``huggingface-cli login``)::

    python create_dataset/export_commit_parquet_to_hf.py \\
        --parquet-dir  $SCRATCH/REPO_DATASET/commit_parquet \\
        --splits-dir   $SCRATCH/REPO_DATASET \\
        --output-dir   $SCRATCH/REPO_DATASET/commit_parquet_hf \\
        --keep-shards \\
        --hub-repo     your-username/code2lora-gru-commits \\
        --private

Load on another cluster::

    # after ``huggingface-cli login``
    huggingface-cli download your-username/code2lora-gru-commits \\
        --repo-type dataset \\
        --local-dir $SCRATCH/REPO_DATASET/commit_parquet_hf

    # then train exactly as before -- the existing loader reads
    # either shards/ or per-split files:
    sbatch scripts/slurm/train_code2lora_gru_commits.sh \\
        # with PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required. Run `module load arrow/18.1.0` (CC) or "
        "`pip install pyarrow` first."
    ) from exc


# Columns we export for each config; keep these in lock-step with the
# schema in build_commit_parquet_db.py.
_COMMIT_COLS = [
    "repo_id",
    "cross_repo_split",
    "commit_index",
    "commit_sha",
    "commit_timestamp",
    "in_repo_split",
    "production_code_diff",
    "n_new_assertions",
]
_QNA_COLS = [
    "repo_id",
    "cross_repo_split",
    "commit_index",
    "commit_sha",
    "in_repo_split",
    "test_file",
    "lineno",
    "col_offset",
    "assertion_type",
    "test_function",
    "prefix",
    "target",
]

_SPLITS = ("train", "cr_val", "cr_test")


def _collect_shards(parquet_dir: Path, kind: str) -> List[Path]:
    """Return sorted list of ``*.<kind>.parquet`` shard paths."""
    shards_dir = parquet_dir / "shards"
    if not shards_dir.is_dir():
        raise SystemExit(f"No shards/ under {parquet_dir}")
    files = sorted(shards_dir.glob(f"*.{kind}.parquet"))
    if not files:
        raise SystemExit(f"No {kind}.parquet shards found in {shards_dir}")
    return files


def _stream_split(
    shard_paths: List[Path],
    columns: List[str],
    split_name: str,
    out_path: Path,
    row_group_size: int = 65536,
) -> int:
    """Write one cross-repo split as a single Parquet file, streamed.

    Returns total row count written.
    """
    if not shard_paths:
        raise SystemExit("no shards supplied")

    ds = pads.dataset([str(p) for p in shard_paths], format="parquet")
    filt = pads.field("cross_repo_split") == split_name
    scanner = ds.scanner(columns=columns, filter=filt, batch_size=8192)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    n_rows = 0
    try:
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    str(out_path),
                    batch.schema,
                    compression="snappy",
                )
            writer.write_batch(batch, row_group_size=row_group_size)
            n_rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # No rows matched -- write an empty file with the expected schema so
        # downstream loaders don't choke on missing files.
        schema = ds.schema.select(
            [ds.schema.get_field_index(c) for c in columns]
        )
        pq.write_table(pa.table({c: [] for c in columns}, schema=schema),
                       out_path, compression="snappy")
    return n_rows


def _copy_splits_json(splits_dir: Path, out_dir: Path) -> None:
    """Copy train.json/cr_val.json/cr_test.json into <out>/splits/."""
    dest = out_dir / "splits"
    dest.mkdir(parents=True, exist_ok=True)
    for name in _SPLITS:
        src = splits_dir / f"{name}.json"
        if src.exists():
            shutil.copy2(src, dest / src.name)
            print(f"  copied {src.name}")
        else:
            print(f"  WARN: missing {src} (skipped)")


def _copy_shards(parquet_dir: Path, out_dir: Path) -> None:
    """Mirror shards/ verbatim (so the unchanged loader can still read them)."""
    src_shards = parquet_dir / "shards"
    dst_shards = out_dir / "shards"
    dst_shards.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in sorted(src_shards.iterdir()):
        if fp.suffix == ".parquet":
            shutil.copy2(fp, dst_shards / fp.name)
            n += 1
    print(f"  mirrored {n} shard files to {dst_shards}")


def _write_readme(
    out_dir: Path,
    n_commits_by_split: dict,
    n_qna_by_split: dict,
) -> None:
    """Create a minimal README with the HF Datasets `configs` front matter so
    `load_dataset(repo_id, 'commits', split='train')` works out of the box."""
    total_commits = sum(n_commits_by_split.values())
    total_qna = sum(n_qna_by_split.values())
    yaml_front = [
        "---",
        "license: mit",
        "task_categories:",
        "  - text-generation",
        "  - feature-extraction",
        "language:",
        "  - code",
        "pretty_name: Code2LoRA-GRU Commit-Level Dataset",
        "configs:",
        "  - config_name: commits",
        "    data_files:",
    ]
    for split in _SPLITS:
        yaml_front.append(f"      - split: {split}")
        yaml_front.append(f"        path: commits/{split}.parquet")
    yaml_front.append("  - config_name: qna")
    yaml_front.append("    data_files:")
    for split in _SPLITS:
        yaml_front.append(f"      - split: {split}")
        yaml_front.append(f"        path: qna/{split}.parquet")
    yaml_front.append("---")

    body = [
        "",
        "# Code2LoRA-GRU commit-level dataset",
        "",
        "Per-repository commit sequences with test-file assertion QnA pairs. "
        "Each kept commit introduces at least one new or changed assertion.",
        "",
        "## Splits",
        "",
        "Cross-repository splits group entire repositories; in-repository splits",
        "chronologically partition the commits within each repo (80/10/10).",
        "",
        f"* **commits**: {total_commits:,} rows total",
    ]
    for s, n in n_commits_by_split.items():
        body.append(f"  * {s}: {n:,}")
    body.append(f"* **qna**: {total_qna:,} rows total")
    for s, n in n_qna_by_split.items():
        body.append(f"  * {s}: {n:,}")

    body.extend([
        "",
        "## Schemas",
        "",
        "### commits",
        "",
        "| column | type | description |",
        "|---|---|---|",
        "| repo_id | string | `<owner>/<repo>` |",
        "| cross_repo_split | string | train / cr_val / cr_test |",
        "| commit_index | int32 | 0-based index within the kept sequence |",
        "| commit_sha | string | git SHA of this kept commit |",
        "| commit_timestamp | string | ISO 8601 |",
        "| in_repo_split | string | train / val / test (chronological 80/10/10) |",
        "| production_code_diff | large_string | filtered unified diff vs prev kept commit |",
        "| n_new_assertions | int32 | number of new assertions introduced |",
        "",
        "### qna",
        "",
        "| column | type | description |",
        "|---|---|---|",
        "| repo_id | string | `<owner>/<repo>` |",
        "| cross_repo_split | string | train / cr_val / cr_test |",
        "| commit_index | int32 | kept commit index |",
        "| commit_sha | string | git SHA |",
        "| in_repo_split | string | train / val / test |",
        "| test_file | string | path of the test file |",
        "| lineno | int32 | 1-based line of the assertion |",
        "| col_offset | int32 | column of the assertion |",
        "| assertion_type | string | `assert` / `self.assertX` / `pytest.raises` / ... |",
        "| test_function | string | enclosing test function name |",
        "| prefix | large_string | test file contents up to the assertion cut |",
        "| target | string | assertion RHS (what the model must predict) |",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        "commits = load_dataset('REPO_ID', 'commits', split='train')",
        "qna     = load_dataset('REPO_ID', 'qna',     split='train')",
        "```",
        "",
        "Or download the whole repo and point the in-project loader at it:",
        "",
        "```bash",
        "huggingface-cli download REPO_ID --repo-type dataset \\",
        "    --local-dir $SCRATCH/REPO_DATASET/commit_parquet_hf",
        "# then:",
        "#   PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf \\",
        "#   sbatch scripts/slurm/train_code2lora_gru_commits.sh",
        "```",
        "",
    ])

    (out_dir / "README.md").write_text(
        "\n".join(yaml_front) + "\n" + "\n".join(body),
        encoding="utf-8",
    )
    print(f"  wrote {out_dir / 'README.md'}")


def _push_to_hub(
    out_dir: Path,
    repo_id: str,
    private: bool,
    allow_patterns: Optional[List[str]],
    ignore_patterns: Optional[List[str]],
    num_workers: int,
    use_large_folder: bool,
) -> None:
    """Upload ``out_dir`` to ``repo_id`` as a dataset repo.

    For folders bigger than ~5 GB we prefer ``upload_large_folder`` which
    is resumable, chunked, multi-worker, and checkpoints to
    ``.cache/huggingface/`` under ``out_dir`` so a second run just picks
    up where the first left off.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --hub-repo. "
            "`pip install huggingface_hub`."
        ) from exc

    api = HfApi()
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    print(f"  uploading {out_dir} -> hf://datasets/{repo_id}")

    if use_large_folder and hasattr(api, "upload_large_folder"):
        # Resumable + parallel. Filters only support *ignore_patterns* here.
        api.upload_large_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
            num_workers=num_workers,
            private=private,
            print_report=True,
        )
    else:
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    print(f"  uploaded. https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, required=True,
                    help="Directory containing shards/ (and optionally "
                         "commits.parquet / qna_pairs.parquet)")
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="Directory with train.json / cr_val.json / cr_test.json")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Destination directory for the HF-ready layout")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Mirror the per-repo shards/ under <output>/shards/ "
                         "(let both loading paths work on the new cluster)")
    ap.add_argument("--hub-repo", type=str, default=None,
                    help="If set, also upload to this HF Hub dataset repo "
                         "(e.g. 'user/code2lora-gru-commits').")
    ap.add_argument("--private", action="store_true",
                    help="Create the Hub repo as private.")
    ap.add_argument("--only-hub-upload", action="store_true",
                    help="Skip re-exporting; just upload an existing "
                         "<output-dir> to the Hub.")
    ap.add_argument("--no-large-folder", action="store_true",
                    help="Force the classic (non-resumable) upload_folder "
                         "API instead of upload_large_folder. Default is "
                         "to use upload_large_folder for big uploads.")
    ap.add_argument("--upload-workers", type=int, default=4,
                    help="Parallel upload workers when using "
                         "upload_large_folder (default: 4).")
    ap.add_argument("--exclude-shards", action="store_true",
                    help="Do not upload the shards/ mirror (halves the "
                         "payload; the new cluster will use prefer=hf).")
    args = ap.parse_args()

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_commits_by_split: dict = {}
    n_qna_by_split: dict = {}

    if not args.only_hub_upload:
        print(f"[1/3] Exporting per-split parquets to {out_dir}/...")

        commit_shards = _collect_shards(args.parquet_dir, "commits")
        qna_shards = _collect_shards(args.parquet_dir, "qna")
        print(f"  found {len(commit_shards)} commit shards, "
              f"{len(qna_shards)} qna shards")

        for split in _SPLITS:
            cout = out_dir / "commits" / f"{split}.parquet"
            nc = _stream_split(commit_shards, _COMMIT_COLS, split, cout)
            n_commits_by_split[split] = nc
            print(f"  commits/{split}.parquet  rows={nc:,}")

            qout = out_dir / "qna" / f"{split}.parquet"
            nq = _stream_split(qna_shards, _QNA_COLS, split, qout)
            n_qna_by_split[split] = nq
            print(f"  qna/{split}.parquet      rows={nq:,}")

        print("[2/3] Copying split JSONs and README...")
        _copy_splits_json(args.splits_dir, out_dir)
        if args.keep_shards:
            _copy_shards(args.parquet_dir, out_dir)
        _write_readme(out_dir, n_commits_by_split, n_qna_by_split)
    else:
        print(f"[1/3] --only-hub-upload: reading stats from existing {out_dir}")

    if args.hub_repo:
        print(f"[3/3] Pushing to HuggingFace Hub: {args.hub_repo}")
        ignore = ["**/.ipynb_checkpoints/**", "**/__pycache__/**"]
        if args.exclude_shards:
            ignore.append("shards/**")
        _push_to_hub(
            out_dir,
            args.hub_repo,
            private=args.private,
            allow_patterns=None,
            ignore_patterns=ignore,
            num_workers=args.upload_workers,
            use_large_folder=not args.no_large_folder,
        )
    else:
        print("[3/3] Skipping Hub upload (no --hub-repo).")

    print("\nDone.")
    print(f"  Local HF-ready dir: {out_dir}")
    if args.hub_repo:
        print(f"  Hub dataset:        https://huggingface.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()
