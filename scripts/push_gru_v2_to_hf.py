#!/usr/bin/env python3
"""Push the v2 Code2LoRA-GRU dataset to HuggingFace Hub.

V2 = v1 + ``diff_embedding`` and ``repo_state_embedding`` columns on every
commits parquet. The qna parquets are unchanged (we keep the smart-cap train
filter on ``qna/train.parquet``).

Layout uploaded::

    <repo-id>/
        README.md
        commits/
            train.parquet      (= v1 + diff_embedding + repo_state_embedding)
            cr_val.parquet     (= v1 + diff_embedding + repo_state_embedding)
            cr_test.parquet    (= v1 + diff_embedding + repo_state_embedding)
        qna/
            train.parquet      (smart-cap snapshot, unchanged from v1)
            cr_val.parquet     (unfiltered canonical RepoPeftBench QnAs)
            cr_test.parquet    (unfiltered canonical RepoPeftBench QnAs)
        splits/                (verbatim from v1)
        SMART_CAP_README.json  (verbatim from v1)
        EMBEDDINGS_README.json (encoder hyperparameters + sha256 of each parquet)

Usage::

    python scripts/push_gru_v2_to_hf.py \\
        --repo-id     nanigock/repopeft-gru-commits-v2 \\
        --v2-dir      /scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2 \\
        --v1-dir      /scratch/lhotsko/REPO_DATASET/commit_parquet_hf \\
        --smartcap-dir /scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


README = """---
license: mit
task_categories:
  - text-generation
  - feature-extraction
language:
  - code
pretty_name: Code2LoRA-GRU Commit-Level Dataset v2 (with frozen Qwen3 embeddings)
size_categories:
  - 100K<n<1M
configs:
  - config_name: commits
    data_files:
      - split: train
        path: commits/train.parquet
      - split: cr_val
        path: commits/cr_val.parquet
      - split: cr_test
        path: commits/cr_test.parquet
  - config_name: qna
    data_files:
      - split: train
        path: qna/train.parquet
      - split: cr_val
        path: qna/cr_val.parquet
      - split: cr_test
        path: qna/cr_test.parquet
---

# Code2LoRA-GRU commit-level dataset, v2

This is the **v2** snapshot of
[`nanigock/repopeft-gru-commits`](https://huggingface.co/datasets/nanigock/repopeft-gru-commits)
extended with two precomputed embedding columns on every commit row:

* `diff_embedding`        - 2048-d `float16`, encodes the filtered
  `production_code_diff` for that commit.
* `repo_state_embedding`  - 2048-d `float16`, encodes the entire `.py`
  tree of the repository at that commit.

Both embeddings come from a single frozen model
(`Qwen/Qwen3-Embedding-0.6B`) and are produced with the exact same recipe
used by the published Code2LoRA-GRU and Code2LoRA-direct trainers. See
`EMBEDDINGS_README.json` for the exact hyperparameters and per-split
sha256 sums.

### Why precompute?

The encoder is frozen for both Code2LoRA-GRU<sub>commit</sub> and
Code2LoRA<sub>direct</sub>, so embedding cost is the dominant per-epoch
overhead. Shipping them inside the dataset means the GRU trainer becomes
a pure dataloader -> RNN -> LoRA-head pipeline (no Qwen3 forward pass on
the hot path), and the static model can train without ever touching the
encoder at all.

### Memory-light loading recipe

The two embedding columns add ~8 GB total over the v1 dataset. For
training rigs with limited host RAM, load them on demand::

    import pyarrow.dataset as pads

    ds = pads.dataset("commits/train.parquet", format="parquet")
    scanner = ds.scanner(columns=["repo_id", "commit_index", "commit_sha",
                                  "diff_embedding"],
                         batch_size=2048)
    for batch in scanner.to_batches():
        ...

This streams one row group at a time (~250 MB peak) and never
materializes the 1.4 GB diff column in memory.

## Splits

(Identical to v1.) Repos are partitioned into a cross-repo
train/cr_val/cr_test split; commits inside each repo are then chopped
80 / 10 / 10 chronologically into in-repo train / val / test slices.

## Schemas

### commits

| column | type | description |
|---|---|---|
| repo_id | string | `<owner>/<repo>` |
| cross_repo_split | string | train / cr_val / cr_test |
| commit_index | int32 | 0-based index within the kept sequence |
| commit_sha | string | git SHA of this kept commit |
| commit_timestamp | string | ISO 8601 |
| in_repo_split | string | train / val / test (80/10/10) |
| production_code_diff | large_string | filtered unified diff vs prev kept commit (test hunks removed) |
| n_new_assertions | int32 | number of assertion events introduced |
| n_added_assertions | int32 | events newly added at this commit |
| n_modified_assertions | int32 | events modified at this commit |
| **diff_embedding** | list[float16, 2048] | Qwen3 encoding of `production_code_diff`. concat(MaxPool, MeanPool); not normalized. |
| **repo_state_embedding** | list[float16, 2048] | Qwen3 encoding of the full `.py` tree at this commit. concat(mean_files, max_files), L2-normalized. |

### qna

Unchanged from v1. See the v1 dataset card for the column list.

## Reproducibility

The v2 build pipeline lives under
[`create_dataset/`](https://github.com/lilianahotsko/RepoPeftData) of
the project repository:

* `build_diff_embeddings_shard.py`
* `build_repo_state_embeddings_shard.py`
* `merge_gru_v2_embeddings.py`

Per-shard SLURM launchers are in `scripts/slurm/`.

## Citation

```bibtex
@misc{repopeft_gru_commits_v2_2026,
  title  = {Code2LoRA-GRU commit-level dataset, v2 (Qwen3 diff and repo-state embeddings)},
  year   = {2026},
  author = {RepoPeftData authors},
}
```
"""


def _materialize_dir(src: Path, dst: Path,
                     follow_symlinks: bool = True) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.is_symlink() and follow_symlinks:
            real = entry.resolve()
            if real.is_file():
                shutil.copy2(real, target)
            elif real.is_dir():
                _materialize_dir(real, target, follow_symlinks=True)
        elif entry.is_dir():
            _materialize_dir(entry, target, follow_symlinks=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


def _stage(v2_dir: Path, v1_dir: Path,
           smartcap_dir: Path, staging: Path) -> None:
    print(f"Staging at {staging}", flush=True)
    staging.mkdir(parents=True, exist_ok=True)

    # 1. commits (v2; includes diff_embedding + repo_state_embedding).
    print(f"\n[commits] copying from {v2_dir / 'commits'} ...", flush=True)
    _materialize_dir(v2_dir / "commits", staging / "commits")

    # 2. qna (v1 smart-cap; unchanged).
    qna_src = smartcap_dir / "qna"
    print(f"\n[qna] copying from {qna_src} ...", flush=True)
    _materialize_dir(qna_src, staging / "qna")

    # 3. splits/ if present in smart-cap.
    src_splits = smartcap_dir / "splits"
    if src_splits.is_symlink():
        src_splits = src_splits.resolve()
    if src_splits.is_dir():
        print(f"\n[splits] copying from {src_splits} ...", flush=True)
        _materialize_dir(src_splits, staging / "splits")

    # 4. Provenance JSONs.
    for fname in ("SMART_CAP_README.json", "EMBEDDINGS_README.json"):
        for candidate in (v2_dir / fname, smartcap_dir / fname):
            if candidate.exists():
                shutil.copy2(candidate, staging / fname)
                print(f"  copied {candidate.name}", flush=True)
                break

    # 5. README.
    (staging / "README.md").write_text(README, encoding="utf-8")


def _push(staging: Path, repo_id: str, token: str | None,
          private: bool, commit_message: str) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"\nCreating dataset repo: {repo_id} (private={private})", flush=True)
    create_repo(repo_id=repo_id, repo_type="dataset", token=token,
                exist_ok=True, private=private)
    print(f"\nUploading {staging} -> {repo_id}", flush=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print(f"\nDone. https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True,
                    help="HF dataset repo, e.g. 'nanigock/repopeft-gru-commits-v2'.")
    ap.add_argument("--v2-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_v2")
    ap.add_argument("--v1-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf")
    ap.add_argument("--smartcap-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap")
    ap.add_argument("--staging-dir", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message",
                    default="Initial upload: v2 (with diff/repo-state embeddings).")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    v2 = Path(args.v2_dir)
    v1 = Path(args.v1_dir)
    sc = Path(args.smartcap_dir)
    if not (v2 / "commits").is_dir():
        sys.exit(f"v2 dir missing commits/: {v2}")
    if not (sc / "qna").is_dir():
        sys.exit(f"smartcap dir missing qna/: {sc}")

    if args.staging_dir:
        staging = Path(args.staging_dir)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _stage(v2, v1, sc, staging)
        if not args.no_upload:
            _push(staging, args.repo_id, args.token,
                  args.private, args.commit_message)
        print(f"\nStaging tree preserved at {staging}")
    else:
        with tempfile.TemporaryDirectory(prefix="hf_push_v2_") as tmp:
            staging = Path(tmp) / "tree"
            _stage(v2, v1, sc, staging)
            if not args.no_upload:
                _push(staging, args.repo_id, args.token,
                      args.private, args.commit_message)


if __name__ == "__main__":
    main()
